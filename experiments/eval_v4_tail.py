"""
Evaluate a v4 S4DMeanFlowNetV4 checkpoint on a target dataset with our
tail metrics (twCRPS, qloss, Winkler, coverage). Supports the 7 configs
registered in experiments/train_v4.py's CONFIGS dict. Noise scaling is
supported via --noise.

Also computes GluonTS mean_wQuantileLoss for comparability to the leaderboard.
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
from meanflow_ts.tail_metrics import compute_all_tail_metrics
from experiments.train_v4 import CONFIGS, LAG_MAP

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class V4NoiseScaleForecaster(nn.Module):
    """Same as V4Forecaster but lets caller set an inference noise scale."""

    def __init__(self, net, ctx_len, pred_len, num_samples, freq, noise=1.0,
                 clamp_min=None):
        super().__init__()
        self.net = net
        self.context_length = ctx_len
        self.prediction_length = pred_len
        self.num_samples = num_samples
        self.freq = freq
        self.noise = noise
        self.clamp_min = clamp_min
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
            z = torch.randn(B, self.prediction_length, device=d) * self.noise
            t = torch.ones(B, device=d)
            h = torch.ones(B, device=d)
            u = self.net(z, (t, h), lags_n)
            pred = self.norm.inverse((z - u).float(), loc, scale).float()
            if self.clamp_min is not None:
                pred = pred.clamp(min=self.clamp_min)
            all_preds.append(pred.cpu())
        return torch.stack(all_preds, dim=1)


def load_v4_net(ckpt_path, cfg, device):
    freq = cfg["freq"]
    n_lags = len(get_lag_indices_v4(freq))
    net = S4DMeanFlowNetV4(
        pred_len=cfg["pred"], ctx_len=cfg["ctx"], d_model=192, n_s4d_blocks=6,
        ssm_dim=64, n_lags=n_lags, freq=freq,
    ).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ck.get("net_ema") or ck.get("net")
    net.load_state_dict(state)
    net.eval()
    return net, ck


def evaluate(name, ckpt_path, num_samples=200, noise=1.0, clamp_min=None,
              device=None):
    cfg = CONFIGS[name]
    freq, ctx_len, pred_len = cfg["freq"], cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP.get(freq, 672)
    device = device or torch.device("cuda")
    net, ck = load_v4_net(ckpt_path, cfg, device)
    logger.info(f"[{name}] loaded {os.path.basename(ckpt_path)} "
                f"(reported_CRPS={ck.get('crps','?')})")

    dataset = get_dataset(name)
    tr = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(start_field="start", target_field="target", output_field="time_feat",
                        time_features=time_features_from_frequency_str(freq), pred_length=pred_len),
    ])
    test_transform = tr.apply(dataset.test, is_train=False)
    sp = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
        past_length=ctx_len + max_lag, future_length=pred_len,
        time_series_fields=["time_feat", "observed_values"],
    )
    fc = V4NoiseScaleForecaster(net, ctx_len, pred_len, num_samples, freq,
                                 noise=noise, clamp_min=clamp_min).to(device)
    pr = PyTorchPredictor(
        prediction_length=pred_len,
        input_names=["past_target", "past_observed_values"],
        prediction_net=fc, batch_size=16, input_transform=sp, device=device,
    )
    t0 = time.time()
    fi, ti = make_evaluation_predictions(dataset=test_transform, predictor=pr,
                                          num_samples=num_samples)
    forecasts = list(fi)
    tss = list(ti)
    logger.info(f"  sampled in {time.time()-t0:.1f}s")

    gluon, _ = Evaluator(num_workers=0)(tss, forecasts)
    samples = np.stack([f.samples for f in forecasts], axis=0).astype(np.float32)
    targets = np.stack([ts.values[-pred_len:].flatten() for ts in tss], axis=0).astype(np.float32)
    tail_m = compute_all_tail_metrics(samples, targets)
    tail_m["gluonts_mean_wQuantileLoss"] = float(gluon["mean_wQuantileLoss"])
    tail_m["noise_scale"] = noise
    return tail_m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--noise", type=float, default=1.0)
    p.add_argument("--clamp-min", type=float, default=None)
    p.add_argument("--output", default="results_v4_tail_eval.json")
    args = p.parse_args()
    m = evaluate(args.dataset, args.ckpt, num_samples=args.num_samples,
                  noise=args.noise, clamp_min=args.clamp_min)
    logger.info(f"  gluonts_CRPS={m['gluonts_mean_wQuantileLoss']:.4f}  "
                f"twCRPS95={m['twCRPS_q095']:.4f}  twCRPS99={m['twCRPS_q099']:.4f}  "
                f"wqloss95={m['wqloss_q095']:.4f}  cov90={m['coverage_90']:.3f}")
    existing = {}
    if os.path.exists(args.output):
        try:
            existing = json.load(open(args.output))
        except Exception:
            existing = {}
    existing[f"{args.dataset}_n{args.noise}"] = m
    with open(args.output, "w") as f:
        json.dump(existing, f, indent=2, default=float)
    logger.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
