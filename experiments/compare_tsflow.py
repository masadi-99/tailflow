"""
Run TSFlow's saved electricity_nips model through the same GluonTS evaluator
to get all metrics for fair comparison with MeanFlow.
"""
import os, sys, json, time, tempfile
import torch
import numpy as np
from tqdm.auto import tqdm

TSFLOW_PATH = os.environ.get("TSFLOW_PATH", os.path.join(os.path.dirname(__file__), '..', '..', 'TSFlow'))
sys.path.insert(0, TSFLOW_PATH)

try:
    os.environ["TSFLOW_NO_KEOPS"] = "1"
    import pykeops
    tmp = tempfile.mkdtemp(prefix="pykeops_build_")
    pykeops.set_build_folder(tmp)
    pykeops.clean_pykeops()
except: pass

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.time_feature import time_features_from_frequency_str
from tsflow.dataset import get_gts_dataset
from tsflow.model import TSFlowCond
from tsflow.utils import create_transforms
from tsflow.utils.util import create_splitter
from tsflow.utils.variables import get_season_length

device = torch.device("cuda")

# Load TSFlow model
print("Loading TSFlow electricity_nips model...")
model = TSFlowCond(
    setting="univariate", target_dim=1,
    context_length=24, prediction_length=24,
    backbone_params=dict(
        input_dim=1, output_dim=1, step_emb=64, num_residual_blocks=3,
        residual_block="s4", hidden_dim=64, dropout=0.0, init_skip=False, feature_skip=True,
    ),
    prior_params=dict(kernel="ou", gamma=1, context_freqs=14),
    optimizer_params=dict(lr=1e-3),
    ema_params=dict(beta=0.9999, update_after_step=128, update_every=1),
    frequency="H", normalization="longmean",
    use_lags=True, use_ema=True, num_steps=32, solver="euler", matching="random",
).to(device)

ckpt_path = os.environ.get("TSFLOW_CKPT", os.path.join(TSFLOW_PATH, "logs/tsflow/20260325_213559/best_checkpoint.ckpt"))
state = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(state, strict=True)
model.eval()
print("TSFlow loaded.")

# Evaluate with different sample counts
dataset = get_gts_dataset("electricity_nips")
num_rolling_evals = int(len(dataset.test) / len(dataset.train))
time_features = time_features_from_frequency_str("H")
transformation = create_transforms(
    time_features=time_features, prediction_length=24,
    freq=get_season_length("H"), train_length=len(dataset.train),
)
# Must apply to train first to build stateful transform caches
_ = list(transformation.apply(dataset.train, is_train=True))
test_data = dataset.test
transformed_testdata = transformation.apply(test_data, is_train=False)

test_splitter = create_splitter(
    past_length=max(24 + max(model.lags_seq), model.prior_context_length),
    future_length=24, mode="test",
)

for num_samples in [50]:
    print(f"\nEvaluating TSFlow with {num_samples} samples...")
    model.num_samples = num_samples
    batch_size = 1024 * 64 // num_samples

    predictor = model.get_predictor(test_splitter, batch_size=batch_size, device=device)

    t0 = time.time()
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=transformed_testdata, predictor=predictor, num_samples=num_samples,
    )
    forecasts = list(tqdm(forecast_it, total=len(transformed_testdata), desc="TSFlow"))
    tss = list(ts_it)
    inference_time = time.time() - t0

    metrics, _ = Evaluator(num_workers=0)(tss, forecasts)
    metrics["CRPS"] = metrics["mean_wQuantileLoss"]

    print(f"\nTSFlow electricity_nips ({num_samples} samples, {inference_time:.1f}s):")
    for k in ["CRPS", "ND", "NRMSE", "MSE", "MASE", "sMAPE", "MSIS",
              "Coverage[0.5]", "Coverage[0.9]", "mean_absolute_QuantileLoss"]:
        if k in metrics:
            print(f"  {k:<35}: {metrics[k]}")

    # Save
    result = {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
              for k, v in metrics.items()}
    result["inference_time_s"] = inference_time
    result["num_samples"] = num_samples
    with open("tsflow_electricity_metrics.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("Saved to tsflow_electricity_metrics.json")

# Now print comparison
print("\n" + "="*80)
print("COMPARISON: MeanFlow-TS vs TSFlow on electricity_nips")
print("="*80)

mf = json.load(open("final_all_metrics.json"))["electricity_nips"]["all_metrics"]
ts = result

key_metrics = ["CRPS", "ND", "NRMSE", "sMAPE", "Coverage[0.5]", "Coverage[0.9]", "MSE"]
print(f"{'Metric':<25} | {'MeanFlow (1-step)':>18} | {'TSFlow (32-step)':>18} | {'Ratio':>8}")
print("-" * 80)
for k in key_metrics:
    mv = float(mf.get(k, 0))
    tv = float(ts.get(k, 0))
    ratio = mv / tv if tv != 0 else float('inf')
    print(f"{k:<25} | {mv:>18.6f} | {tv:>18.6f} | {ratio:>7.2f}x")
print(f"{'Inference time (s)':<25} | {34.6:>18.1f} | {inference_time:>18.1f} |")
print(f"{'NFE (network evals)':<25} | {'1':>18} | {'32':>18} |")
