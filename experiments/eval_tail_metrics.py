"""
Evaluate tail-aware metrics (twCRPS, quantile loss at extremes, Winkler,
coverage, exceedance Brier) for the extremity-conditioned model.

Two evaluations:
  1. Tail metric table: phase2 baseline vs self-training round 2 at
     target_extremity=0.5 (marginal), guidance=1.0. Standard reviewer
     comparison.
  2. CFG controllability sweep: fix self-training round 2, sweep guidance
     scale w ∈ {0.0, 0.5, 1.0, 2.0, 4.0, 6.0} at target extremity 0.9.
     Report (a) mean sample max value (upward shift), (b) twCRPS on tail
     subset, (c) overall CRPS. Saves a JSON per dataset.

This script reuses the ConditionedMeanFlowNet / TailFlowForecaster from
train_tail_v2 and pulls the frozen quantile mapper from the saved
checkpoints.
"""
import os, sys, json, argparse, logging, pickle, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator, AddTimeFeatures, AsNumpyArray,
    Chain, InstanceSplitter, TestSplitSampler,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model import ConditionalMeanFlowNet
from meanflow_ts.tail_metrics import compute_all_tail_metrics
from experiments.train_tail_v2 import (
    ConditionedMeanFlowNet, TailFlowForecaster, CONFIGS, LAG_MAP,
)

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def build_net(device):
    base = ConditionalMeanFlowNet(
        pred_len=24, ctx_len=24,
        model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
    )
    cond = ConditionedMeanFlowNet(base, cfg_drop_prob=0.2)
    return cond.to(device)


def build_net_for_exchange(device):
    base = ConditionalMeanFlowNet(
        pred_len=30, ctx_len=30,
        model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
    )
    cond = ConditionedMeanFlowNet(base, cfg_drop_prob=0.2)
    return cond.to(device)


def load_cond_net(ckpt_path, cfg, device):
    if cfg["pred"] == 30:
        net = build_net_for_exchange(device)
    else:
        net = build_net(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ck.get("cond_ema", ck.get("cond_net"))
    if state is None:
        raise ValueError(f"no cond_ema/cond_net key in {ckpt_path}")
    net.load_state_dict(state)
    net.eval()
    return net, ck


def collect_samples_and_targets(net, cfg, dataset, device,
                                 num_samples=200, guidance_scale=1.0,
                                 target_extremity=0.5, batch_size=128):
    """Run inference once, return (samples, targets) numpy arrays."""
    freq, ctx_len, pred_len = cfg["freq"], cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP.get(freq, 672)

    tr = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(start_field="start", target_field="target", output_field="time_feat",
                        time_features=time_features_from_frequency_str(freq), pred_length=pred_len),
    ])
    test_transform = tr.apply(dataset.test, is_train=False)
    splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
        past_length=ctx_len + max_lag, future_length=pred_len,
        time_series_fields=["time_feat", "observed_values"],
    )

    forecaster = TailFlowForecaster(
        net, ctx_len, pred_len, num_samples=num_samples,
        guidance_scale=guidance_scale, target_extremity=target_extremity,
    ).to(device)
    predictor = PyTorchPredictor(
        prediction_length=pred_len,
        input_names=["past_target", "past_observed_values"],
        prediction_net=forecaster, batch_size=batch_size,
        input_transform=splitter, device=device,
    )

    fi, ti = make_evaluation_predictions(
        dataset=test_transform, predictor=predictor, num_samples=num_samples,
    )
    forecasts = list(fi)
    tss = list(ti)

    # Stack into arrays: (N, S, T) and (N, T)
    samples = np.stack([f.samples for f in forecasts], axis=0)  # (N, S, T)
    targets = np.stack([ts.values[-pred_len:].flatten() for ts in tss], axis=0)  # (N, T)
    return samples, targets, forecasts, tss


def evaluate_checkpoint(name, ckpt_path, stage_label, num_samples, device,
                        thresholds=(0.9, 0.95, 0.99)):
    cfg = CONFIGS[name]
    dataset = get_dataset(name)
    net, _ = load_cond_net(ckpt_path, cfg, device)
    samples, targets, forecasts, tss = collect_samples_and_targets(
        net, cfg, dataset, device,
        num_samples=num_samples, guidance_scale=1.0, target_extremity=0.5,
    )
    metrics = compute_all_tail_metrics(samples, targets, thresholds=thresholds,
                                        quantiles=thresholds)
    # Also report GluonTS mean_wQuantileLoss for comparability to our RESULTS
    gluon_metrics, _ = Evaluator(num_workers=0)(tss, forecasts)
    metrics["gluonts_mean_wQuantileLoss"] = float(gluon_metrics["mean_wQuantileLoss"])
    metrics["stage"] = stage_label
    metrics["ckpt"] = os.path.basename(ckpt_path)
    return metrics


def cfg_sweep_for_controllability(name, ckpt_path, num_samples, device,
                                   w_grid=(0.0, 0.5, 1.0, 2.0, 4.0, 6.0),
                                   target_extremity=0.9):
    """
    Show 'the knob works':
      - mean sample max value (should increase with w)
      - twCRPS at q=0.95 on the tail subset (10% most extreme test windows)
      - overall CRPS
    """
    cfg = CONFIGS[name]
    dataset = get_dataset(name)
    net, _ = load_cond_net(ckpt_path, cfg, device)
    # Pre-compute tail subset indices using a target-extremity score on GT
    _, targets_ref, _, _ = collect_samples_and_targets(
        net, cfg, dataset, device, num_samples=1,
        guidance_scale=1.0, target_extremity=0.5,
    )
    gt_norm = targets_ref - targets_ref.mean(axis=1, keepdims=True)
    gt_scores = np.abs(gt_norm).max(axis=1)  # max deviation per window
    tail_thr = np.quantile(gt_scores, 0.9)
    tail_mask = gt_scores >= tail_thr
    logger.info(f"[{name}] tail subset size = {tail_mask.sum()}/{len(tail_mask)}")

    threshold_abs = float(np.quantile(targets_ref, 0.95))
    results = []
    for w in w_grid:
        samples, targets, _, _ = collect_samples_and_targets(
            net, cfg, dataset, device,
            num_samples=num_samples, guidance_scale=float(w),
            target_extremity=float(target_extremity),
        )
        overall_crps_m = compute_all_tail_metrics(samples, targets, thresholds=(0.95,),
                                                    quantiles=(0.95,))
        # tail subset
        sub_s = samples[tail_mask]
        sub_t = targets[tail_mask]
        tail_m = compute_all_tail_metrics(sub_s, sub_t, thresholds=(0.95,),
                                           quantiles=(0.95,))
        mean_sample_max = float(samples.max(axis=(1, 2)).mean())
        row = {
            "w": float(w),
            "overall_crps": overall_crps_m["crps"],
            "overall_twCRPS_q095": overall_crps_m["twCRPS_q095"],
            "tail_subset_crps": tail_m["crps"],
            "tail_subset_twCRPS_q095": tail_m["twCRPS_q095"],
            "mean_sample_max": mean_sample_max,
        }
        logger.info(f"  w={w}: crps={row['overall_crps']:.4f}  "
                    f"twCRPS95={row['overall_twCRPS_q095']:.4f}  "
                    f"tail_crps={row['tail_subset_crps']:.4f}  "
                    f"sample_max={row['mean_sample_max']:.2f}")
        results.append(row)
    return {
        "dataset": name,
        "target_extremity": float(target_extremity),
        "tail_threshold_abs": threshold_abs,
        "w_sweep": results,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["solar_nips", "electricity_nips", "traffic_nips",
                            "exchange_rate_nips"])
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--do-tail-table", action="store_true")
    p.add_argument("--do-cfg-sweep", action="store_true")
    p.add_argument("--output", default="results_tail_eval.json")
    args = p.parse_args()

    if not (args.do_tail_table or args.do_cfg_sweep):
        args.do_tail_table = True
        args.do_cfg_sweep = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {"datasets": {}}

    base = os.path.join(os.path.dirname(__file__), "..", "results")
    for name in args.datasets:
        logger.info(f"{'='*60}\n{name}\n{'='*60}")
        ds_out = {}
        phase2 = os.path.join(base, name, "phase2_best.pt")
        st2 = os.path.join(base, name, "st_round_2.pt")
        if not os.path.exists(phase2):
            logger.warning(f"missing {phase2}")
            continue

        if args.do_tail_table:
            t0 = time.time()
            ds_out["phase2"] = evaluate_checkpoint(
                name, phase2, "phase2_best", args.num_samples, device)
            logger.info(f"phase2 done in {time.time()-t0:.1f}s")
            if os.path.exists(st2):
                t0 = time.time()
                ds_out["self_train_round_2"] = evaluate_checkpoint(
                    name, st2, "st_round_2", args.num_samples, device)
                logger.info(f"st_round_2 done in {time.time()-t0:.1f}s")

        if args.do_cfg_sweep:
            sweep_ckpt = st2 if os.path.exists(st2) else phase2
            t0 = time.time()
            ds_out["cfg_sweep"] = cfg_sweep_for_controllability(
                name, sweep_ckpt, args.num_samples, device)
            logger.info(f"cfg sweep done in {time.time()-t0:.1f}s")

        out["datasets"][name] = ds_out
        # Write intermediate in case of interruption
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, default=float)

    logger.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
