"""
Evaluate a ConditionedS4DMeanFlowNetV4 (v4 backbone + extremity adapter)
checkpoint with a classifier-free-guidance sweep. The key diagnostic is
whether `sample_max` grows monotonically with guidance scale w — if yes,
the adapter learned useful conditioning; if no, it is still at zero.

Also emits tail metrics for each (w, tq) pair so we can build a Pareto
plot against v4-base + twCRPS-retraining.
"""
from __future__ import annotations
import os, sys, json, time, argparse, logging
import numpy as np
import torch
import torch.nn as nn

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator, AddTimeFeatures, AsNumpyArray,
    Chain, InstanceSplitter, TestSplitSampler,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v4 import (
    S4DMeanFlowNetV4, get_lag_indices_v4, extract_lags_v4, RobustNorm,
)
from meanflow_ts.model_v4_tail import (
    ConditionedS4DMeanFlowNetV4, guided_sample_v4,
)
from meanflow_ts.tail_metrics import compute_all_tail_metrics, compute_train_thresholds
from experiments.train_v4 import CONFIGS, LAG_MAP

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class V4CFGForecaster(nn.Module):
    def __init__(self, cond_net, ctx_len, pred_len, num_samples, freq,
                 guidance_scale, target_extremity, noise_scale=1.0):
        super().__init__()
        self.cond = cond_net
        self.context_length = ctx_len
        self.prediction_length = pred_len
        self.num_samples = num_samples
        self.freq = freq
        self.w = float(guidance_scale)
        self.tq = float(target_extremity)
        self.noise = float(noise_scale)
        self.norm = RobustNorm()

    def forward(self, past_target, past_observed_values, **kw):
        d = past_target.device
        B = past_target.shape[0]
        ctx = past_target[:, -self.context_length:]
        ctx_n, loc, scale = self.norm(ctx)
        lags = extract_lags_v4(past_target.float(), self.context_length, self.freq)
        lags_n = (lags - loc.unsqueeze(1)) / scale.unsqueeze(1)

        all_preds = []
        for _ in range(self.num_samples):
            ext_q = torch.full((B,), self.tq, device=d)
            pred_n = guided_sample_v4(
                self.cond, lags_n, ext_q,
                (B, self.prediction_length), d,
                guidance_scale=self.w, noise_scale=self.noise,
            )
            pred = self.norm.inverse(pred_n.float(), loc, scale).float()
            all_preds.append(pred.cpu())
        return torch.stack(all_preds, dim=1)


def load_cond_v4(ckpt_path, cfg, device):
    n_lags = len(get_lag_indices_v4(cfg["freq"]))
    base = S4DMeanFlowNetV4(
        pred_len=cfg["pred"], ctx_len=cfg["ctx"],
        d_model=192, n_s4d_blocks=6, ssm_dim=64,
        n_lags=n_lags, freq=cfg["freq"],
    ).to(device)
    cond = ConditionedS4DMeanFlowNetV4(base, cfg_drop_prob=0.2).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ck.get("cond_ema") or ck.get("cond_net")
    cond.load_state_dict(state)
    cond.eval()
    return cond, ck


def run_sweep(name, ckpt_path, num_samples, tq, noise, w_grid, device):
    cfg = CONFIGS[name]
    freq, ctx_len, pred_len = cfg["freq"], cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP[freq]
    cond, ck = load_cond_v4(ckpt_path, cfg, device)
    logger.info(f"[{name}] loaded {os.path.basename(ckpt_path)}  epoch={ck.get('epoch')}")

    dataset = get_dataset(name)
    train_thr = compute_train_thresholds(dataset.train, quantiles=(0.9, 0.95, 0.99))
    tr = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(start_field="start", target_field="target", output_field="time_feat",
                        time_features=time_features_from_frequency_str(freq), pred_length=pred_len),
    ])
    test_t = tr.apply(dataset.test, is_train=False)
    sp = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
        past_length=ctx_len + max_lag, future_length=pred_len,
        time_series_fields=["time_feat", "observed_values"],
    )

    results = []
    for w in w_grid:
        fc = V4CFGForecaster(cond, ctx_len, pred_len, num_samples, freq,
                              guidance_scale=w, target_extremity=tq, noise_scale=noise).to(device)
        pr = PyTorchPredictor(
            prediction_length=pred_len,
            input_names=["past_target", "past_observed_values"],
            prediction_net=fc, batch_size=16, input_transform=sp, device=device,
        )
        t0 = time.time()
        fi, ti = make_evaluation_predictions(dataset=test_t, predictor=pr,
                                              num_samples=num_samples)
        forecasts = list(fi); tss = list(ti)
        gluon, _ = Evaluator(num_workers=0)(tss, forecasts)
        samples = np.stack([f.samples for f in forecasts], axis=0).astype(np.float32)
        targets = np.stack([ts.values[-pred_len:].flatten() for ts in tss], axis=0).astype(np.float32)
        m = compute_all_tail_metrics(samples, targets, train_thresholds=train_thr)
        m["gluonts_mean_wQuantileLoss"] = float(gluon["mean_wQuantileLoss"])
        m["w"] = float(w)
        m["tq"] = float(tq)
        m["noise"] = float(noise)
        m["sample_max_mean"] = float(samples.max(axis=(1, 2)).mean())
        m["time_s"] = time.time() - t0
        logger.info(
            f"  w={w:.1f} tq={tq:.2f} n={noise:.1f}: CRPS={m['gluonts_mean_wQuantileLoss']:.4f}  "
            f"twCRPS95={m['twCRPS_q095']:.4f}  twCRPS99={m['twCRPS_q099']:.4f}  "
            f"sample_max={m['sample_max_mean']:.2f}  ({m['time_s']:.1f}s)"
        )
        results.append(m)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--tq", type=float, default=0.9)
    p.add_argument("--noise", type=float, default=1.0)
    p.add_argument("--w-grid", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 2.0, 4.0, 6.0])
    p.add_argument("--output", default="results_v4_cfg_sweep.json")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = run_sweep(args.dataset, args.ckpt, args.num_samples,
                     args.tq, args.noise, args.w_grid, device)
    out = {}
    if os.path.exists(args.output):
        try: out = json.load(open(args.output))
        except Exception: out = {}
    out[f"{args.dataset}_tq{args.tq}_n{args.noise}"] = rows
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=float)
    logger.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
