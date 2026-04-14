"""
Run MeanFlow on a single dataset. Called with: python run_single_dataset.py <dataset_name>
"""
import os, sys, time, math, logging, tempfile, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
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
except:
    pass

sys.path.insert(0, os.path.dirname(__file__))
from meanflow_ts.model import ConditionalMeanFlowNet, conditional_meanflow_loss, MeanFlowForecaster

CONFIGS = {
    "solar_nips":          {"freq": "H", "ctx": 24, "pred": 24},
    "traffic_nips":        {"freq": "H", "ctx": 24, "pred": 24},
    "exchange_rate_nips":  {"freq": "B", "ctx": 30, "pred": 30},
    "m4_hourly":           {"freq": "H", "ctx": 48, "pred": 48},
}
TSFLOW = {
    "solar_nips": 0.340, "traffic_nips": 0.079,
    "exchange_rate_nips": 0.008, "m4_hourly": 0.042,
}
LAG_MAP = {"H": 672, "B": 750}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    name = args.dataset
    cfg = CONFIGS[name]
    freq, ctx_len, pred_len = cfg["freq"], cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP.get(freq, 672)
    max_epochs = args.epochs

    logfile = f"{name}.log"
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO,
        handlers=[logging.FileHandler(logfile, mode='w'), logging.StreamHandler()]
    )
    logger = logging.getLogger()

    device = torch.device(f"cuda:{args.cuda}")
    torch.manual_seed(6432)
    np.random.seed(6432)

    logger.info(f"=== {name} (freq={freq}, ctx={ctx_len}, pred={pred_len}) ===")
    dataset = get_dataset(name)
    logger.info(f"Train: {len(dataset.train)}, Test: {len(dataset.test)}")

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
        Cached(transformed_data), batch_size=256, stack_fn=batchify,
        transform=train_splitter, num_batches_per_epoch=32, shuffle_buffer_length=10000,
    )

    net = ConditionalMeanFlowNet(
        pred_len=pred_len, ctx_len=ctx_len,
        model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
    ).to(device)
    net_ema = deepcopy(net).eval()
    logger.info(f"Params: {sum(p.numel() for p in net.parameters()):,}")

    optimizer = AdamW(net.parameters(), lr=6e-4)
    best_crps = float('inf')
    start_epoch = 0

    # Resume from checkpoint if requested
    ckpt_path = f'checkpoint_{name}.pt'
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt['net'])
        net_ema.load_state_dict(ckpt['net_ema'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        best_crps = ckpt.get('best_crps', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}, best CRPS={best_crps:.6f}")

    for epoch in range(start_epoch, max_epochs):
        net.train()
        epoch_loss, n_b = 0, 0
        t0 = time.time()
        for batch in train_loader:
            past = batch["past_target"].to(device)
            future = batch["future_target"].to(device)
            ctx = past[:, -ctx_len:]
            loc = ctx.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
            loss = conditional_meanflow_loss(net, future / loc, ctx / loc)
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
            logger.info(f"Epoch {epoch+1:>3}/{max_epochs} | Loss: {avg:.4f} | {elapsed:.1f}s")

        if (epoch+1) % 100 == 0 or (epoch+1) == max_epochs:
            logger.info("Evaluating...")
            net_ema.eval()
            test_transform = transformation.apply(dataset.test, is_train=False)
            test_splitter = InstanceSplitter(
                target_field="target", is_pad_field="is_pad", start_field="start",
                forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
                past_length=ctx_len + max_lag, future_length=pred_len,
                time_series_fields=["time_feat", "observed_values"],
            )
            forecaster = MeanFlowForecaster(net_ema, ctx_len, pred_len, num_samples=16).to(device)
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
            metrics, _ = Evaluator(num_workers=1)(tss, forecasts)
            crps = metrics["mean_wQuantileLoss"]
            nd = metrics["ND"]
            nrmse = metrics["NRMSE"]
            tag = " ***BEST***" if crps < best_crps else ""
            if crps < best_crps:
                best_crps = crps
                torch.save({'net_ema': net_ema.state_dict(), 'crps': crps, 'epoch': epoch+1},
                           f'best_meanflow_{name}.pt')
            # Always save resumable checkpoint
            torch.save({
                'net': net.state_dict(), 'net_ema': net_ema.state_dict(),
                'optimizer': optimizer.state_dict(), 'epoch': epoch+1,
                'best_crps': best_crps,
            }, f'checkpoint_{name}.pt')
            tsflow_crps = TSFLOW.get(name, "?")
            logger.info(f"CRPS={crps:.6f} | ND={nd:.6f} | NRMSE={nrmse:.4f}{tag}")
            logger.info(f"TSFlow: {tsflow_crps} | Best: {best_crps:.6f}")

    logger.info(f"DONE. Best CRPS: {best_crps:.6f}")


if __name__ == "__main__":
    main()
