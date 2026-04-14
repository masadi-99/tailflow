"""
MeanFlow-TS v4: Training with lag features (improvement iteration 1).
Usage: python train_forecasting_v4.py <dataset> --epochs 600
"""
import os, sys, time, argparse, logging, tempfile
import numpy as np
import torch
from torch.optim import AdamW
from copy import deepcopy
from tqdm.auto import tqdm

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.loader import TrainDataLoader
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.itertools import Cached
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator, AddTimeFeatures, AsNumpyArray,
    Chain, ExpectedNumInstanceSampler, InstanceSplitter, TestSplitSampler,
)

try:
    import pykeops
    tmp = tempfile.mkdtemp(prefix="pykeops_build_")
    pykeops.set_build_folder(tmp)
    pykeops.clean_pykeops()
except: pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v2 import (
    ConditionalMeanFlowNetV2, conditional_meanflow_loss_v2,
    MeanFlowForecasterV2, extract_lag_features,
)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIGS = {
    "electricity_nips":   {"freq": "H", "ctx": 24, "pred": 24},
    "solar_nips":         {"freq": "H", "ctx": 24, "pred": 24},
    "traffic_nips":       {"freq": "H", "ctx": 24, "pred": 24},
    "exchange_rate_nips": {"freq": "B", "ctx": 30, "pred": 30},
    "m4_hourly":          {"freq": "H", "ctx": 48, "pred": 48},
}
LAG_MAP = {"H": 672, "B": 750}
TSFLOW_CRPS = {
    "electricity_nips": 0.045, "solar_nips": 0.341,
    "traffic_nips": 0.082, "exchange_rate_nips": 0.005, "m4_hourly": 0.029,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--n-lags", type=int, default=7)
    args = parser.parse_args()

    name = args.dataset
    cfg = CONFIGS[name]
    freq, ctx_len, pred_len = cfg["freq"], cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP.get(freq, 672)
    n_lags = args.n_lags
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(6432)
    np.random.seed(6432)

    logger.info(f"=== {name} | v4 (lags={n_lags}) | ctx={ctx_len} pred={pred_len} ===")

    dataset = get_dataset(name)
    transformation = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(
            start_field="start", target_field="target", output_field="time_feat",
            time_features=time_features_from_frequency_str(freq), pred_length=pred_len,
        ),
    ])
    train_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start",
        instance_sampler=ExpectedNumInstanceSampler(num_instances=1, min_future=pred_len),
        past_length=ctx_len + max_lag, future_length=pred_len,
        time_series_fields=["time_feat", "observed_values"],
    )
    transformed_data = transformation.apply(dataset.train, is_train=True)
    train_loader = TrainDataLoader(
        Cached(transformed_data), batch_size=64, stack_fn=batchify,
        transform=train_splitter, num_batches_per_epoch=128, shuffle_buffer_length=10000,
    )

    net = ConditionalMeanFlowNetV2(
        pred_len=pred_len, ctx_len=ctx_len, n_lags=n_lags,
        model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
    ).to(device)
    net_ema = deepcopy(net).eval()
    logger.info(f"Params: {sum(p.numel() for p in net.parameters()):,}")

    optimizer = AdamW(net.parameters(), lr=6e-4)
    best_crps = float('inf')

    for epoch in range(args.epochs):
        net.train()
        epoch_loss, n_b = 0, 0
        t0 = time.time()

        for batch in train_loader:
            past = batch["past_target"].to(device)
            future = batch["future_target"].to(device)
            ctx = past[:, -ctx_len:]
            loc = ctx.abs().mean(dim=1, keepdim=True).clamp(min=0.01)

            # Extract lag features and scale everything
            ctx_with_lags = extract_lag_features(past, ctx_len, freq, n_lags) / loc.unsqueeze(1)
            scaled_future = future / loc

            loss = conditional_meanflow_loss_v2(net, scaled_future, ctx_with_lags)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for p, pe in zip(net.parameters(), net_ema.parameters()):
                    pe.data.lerp_(p.data, 1e-4)
            epoch_loss += loss.item()
            n_b += 1

        avg = epoch_loss / n_b
        elapsed = time.time() - t0
        if (epoch+1) % 20 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:>3}/{args.epochs} | Loss: {avg:.4f} | {elapsed:.1f}s")

        if (epoch+1) % 100 == 0 or (epoch+1) == args.epochs:
            logger.info("Evaluating...")
            net_ema.eval()
            test_transform = transformation.apply(dataset.test, is_train=False)
            test_splitter = InstanceSplitter(
                target_field="target", is_pad_field="is_pad", start_field="start",
                forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
                past_length=ctx_len + max_lag, future_length=pred_len,
                time_series_fields=["time_feat", "observed_values"],
            )
            forecaster = MeanFlowForecasterV2(
                net_ema, ctx_len, pred_len, num_samples=16, freq=freq, n_lags=n_lags,
            ).to(device)
            predictor = PyTorchPredictor(
                prediction_length=pred_len,
                input_names=["past_target", "past_observed_values"],
                prediction_net=forecaster, batch_size=512,
                input_transform=test_splitter, device=device,
            )
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=test_transform, predictor=predictor, num_samples=16,
            )
            forecasts = list(tqdm(forecast_it, total=len(test_transform), desc="Eval"))
            tss = list(ts_it)
            metrics, _ = Evaluator(num_workers=0)(tss, forecasts)
            crps = metrics["mean_wQuantileLoss"]
            nd = metrics["ND"]
            nrmse = metrics["NRMSE"]

            tag = " ***BEST***" if crps < best_crps else ""
            if crps < best_crps:
                best_crps = crps
                torch.save({
                    'net_ema': net_ema.state_dict(), 'epoch': epoch+1, 'crps': crps,
                }, f'best_v4_{name}.pt')

            tsflow = TSFLOW_CRPS.get(name, "?")
            logger.info(f"  CRPS={crps:.6f} | ND={nd:.6f} | NRMSE={nrmse:.4f}{tag}")
            logger.info(f"  TSFlow: {tsflow} | Best: {best_crps:.6f}")

    logger.info(f"DONE. Best CRPS: {best_crps:.6f}")


if __name__ == "__main__":
    main()
