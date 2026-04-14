"""Load saved best checkpoints and compute ALL GluonTS metrics for each dataset."""
import os, sys, tempfile, json, time
import torch
import numpy as np
from tqdm.auto import tqdm

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator, AddTimeFeatures, AsNumpyArray,
    Chain, InstanceSplitter, TestSplitSampler,
)

try:
    import pykeops
    tmp = tempfile.mkdtemp(prefix="pykeops_build_")
    pykeops.set_build_folder(tmp)
    pykeops.clean_pykeops()
except: pass

sys.path.insert(0, os.path.dirname(__file__))
from meanflow_ts.model import ConditionalMeanFlowNet, MeanFlowForecaster

DATASETS = {
    "electricity_nips": {"freq": "H", "ctx": 24, "pred": 24, "ckpt": "best_cond_meanflow.pt"},
    "solar_nips":       {"freq": "H", "ctx": 24, "pred": 24, "ckpt": "best_meanflow_solar_nips.pt"},
    "traffic_nips":     {"freq": "H", "ctx": 24, "pred": 24, "ckpt": "best_meanflow_traffic_nips.pt"},
    "exchange_rate_nips": {"freq": "B", "ctx": 30, "pred": 30, "ckpt": "best_meanflow_exchange_rate_nips.pt"},
    "m4_hourly":        {"freq": "H", "ctx": 48, "pred": 48, "ckpt": "best_meanflow_m4_hourly.pt"},
}
LAG_MAP = {"H": 672, "B": 750}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_results = {}

for name, cfg in DATASETS.items():
    ctx, pred, freq = cfg["ctx"], cfg["pred"], cfg["freq"]
    ckpt_path = cfg["ckpt"]
    if not os.path.exists(ckpt_path):
        print(f"SKIP {name}: {ckpt_path} not found")
        continue

    print(f"\n{'='*60}")
    print(f"Evaluating {name}")
    print(f"{'='*60}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = ConditionalMeanFlowNet(
        pred_len=pred, ctx_len=ctx, model_channels=128,
        num_res_blocks=4, time_emb_dim=64, dropout=0.1,
    ).to(device)
    net.load_state_dict(ckpt['net_ema'])
    net.eval()
    print(f"  Loaded from epoch {ckpt.get('epoch', '?')}, training CRPS={ckpt.get('crps', '?')}")

    dataset = get_dataset(name)
    max_lag = LAG_MAP.get(freq, 672)

    transformation = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(
            start_field="start", target_field="target", output_field="time_feat",
            time_features=time_features_from_frequency_str(freq), pred_length=pred,
        ),
    ])
    test_transform = transformation.apply(dataset.test, is_train=False)
    test_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
        past_length=ctx + max_lag, future_length=pred,
        time_series_fields=["time_feat", "observed_values"],
    )

    num_samples = 50
    forecaster = MeanFlowForecaster(net, ctx, pred, num_samples=num_samples).to(device)

    t0 = time.time()
    predictor = PyTorchPredictor(
        prediction_length=pred,
        input_names=["past_target", "past_observed_values"],
        prediction_net=forecaster, batch_size=256,
        input_transform=test_splitter, device=device,
    )
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_transform, predictor=predictor, num_samples=num_samples,
    )
    forecasts = list(tqdm(forecast_it, total=len(test_transform), desc=name))
    tss = list(ts_it)
    inference_time = time.time() - t0

    metrics, per_ts = Evaluator(num_workers=0)(tss, forecasts)

    # Add CRPS alias
    metrics["CRPS"] = metrics["mean_wQuantileLoss"]

    # Select key metrics
    key_metrics = [
        "CRPS", "ND", "NRMSE", "MSE", "MASE",
        "sMAPE", "MSIS", "mean_absolute_QuantileLoss",
        "Coverage[0.5]", "Coverage[0.9]",
    ]

    print(f"\n  Key Metrics ({num_samples} samples, inference: {inference_time:.1f}s):")
    print(f"  {'Metric':<35} | {'Value':>12}")
    print(f"  {'-'*50}")
    result = {"inference_time_s": inference_time, "num_samples": num_samples}
    for k in key_metrics:
        if k in metrics:
            v = metrics[k]
            print(f"  {k:<35} | {v:>12.6f}")
            result[k] = float(v)

    # Also dump ALL metrics
    result["all_metrics"] = {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                             for k, v in metrics.items()}
    all_results[name] = result

# Summary table
print(f"\n\n{'='*80}")
print(f"FINAL SUMMARY — MeanFlow-TS (1-step inference, 100 samples)")
print(f"{'='*80}")
print(f"{'Dataset':<25} | {'CRPS':>8} | {'ND':>8} | {'NRMSE':>8} | {'MASE':>8} | {'Time(s)':>8}")
print(f"{'-'*80}")
for name, r in all_results.items():
    print(f"{name:<25} | {r.get('CRPS',0):>8.4f} | {r.get('ND',0):>8.4f} | {r.get('NRMSE',0):>8.4f} | {r.get('MASE','N/A'):>8} | {r['inference_time_s']:>8.1f}")
print(f"{'='*80}")

with open('final_all_metrics.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print("Saved to final_all_metrics.json")
