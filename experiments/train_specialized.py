"""
Specialized training for Solar and Exchange rate datasets.

Solar: Seasonal decomposition + zero-aware modeling
Exchange: Small model + data augmentation + multi-scale features

Usage:
    python experiments/train_specialized.py solar_nips --epochs 1500
    python experiments/train_specialized.py exchange_rate_nips --epochs 2000
"""
import os, sys, time, argparse, logging, json, tempfile
import numpy as np
import torch
from torch.optim import AdamW
from copy import deepcopy

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
    import pykeops; tmp = tempfile.mkdtemp(prefix="pykeops_"); pykeops.set_build_folder(tmp); pykeops.clean_pykeops()
except: pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v4 import RobustNorm, extract_lags_v4, get_lag_indices_v4

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

LAG_MAP = {"H": 672, "B": 750}
TSFLOW = {"solar_nips": 0.341, "exchange_rate_nips": 0.005}


def train_solar(args):
    from meanflow_ts.model_solar import (
        SolarMeanFlowNet, solar_meanflow_loss, SolarForecaster,
        hour_from_time_feat,
    )

    name = "solar_nips"
    freq, ctx_len, pred_len = "H", 72, 24
    max_lag = LAG_MAP["H"]
    n_lags = len(get_lag_indices_v4(freq))
    n_tf = len(time_features_from_frequency_str(freq))
    device = torch.device("cuda")
    torch.manual_seed(42); np.random.seed(42)

    outdir = os.path.join(os.path.dirname(__file__), '..', 'results_solar', name)
    os.makedirs(outdir, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"Solar Specialized Training")
    logger.info(f"{'='*60}")

    dataset = get_dataset(name)
    transformation = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(start_field="start", target_field="target", output_field="time_feat",
                        time_features=time_features_from_frequency_str(freq), pred_length=pred_len),
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
        Cached(transformed_data), batch_size=args.batch_size, stack_fn=batchify,
        transform=train_splitter, num_batches_per_epoch=100, shuffle_buffer_length=2000,
    )

    norm = RobustNorm()
    net = SolarMeanFlowNet(
        pred_len=pred_len, ctx_len=ctx_len, d_model=192, n_s4d_blocks=6,
        ssm_dim=64, n_lags=n_lags, n_time_features=n_tf,
    ).to(device)
    net_ema = deepcopy(net).eval()
    logger.info(f"Params: {sum(p.numel() for p in net.parameters()):,}")

    optimizer = AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    best_crps = float('inf')

    for epoch in range(args.epochs):
        net.train()
        epoch_loss, n_b = 0, 0
        t0 = time.time()
        for batch in train_loader:
            past = batch["past_target"].to(device)
            future = batch["future_target"].to(device)
            past_tf = batch["past_time_feat"].to(device)
            future_tf = batch["future_time_feat"].to(device)
            ctx = past[:, -ctx_len:]

            # Get hours
            ctx_hours = hour_from_time_feat(past_tf)[:, -ctx_len:]
            fut_hours = hour_from_time_feat(future_tf)[:, :pred_len]

            # Deseasonalize
            ctx_deseas = net.daily.remove_pattern(ctx, ctx_hours)
            fut_deseas = net.daily.remove_pattern(future, fut_hours)

            # Normalize deseasonalized
            ctx_normed, loc, scale = norm(ctx_deseas)
            fut_normed = (fut_deseas - loc) / scale

            # Build context channels
            lags = extract_lags_v4(past, ctx_len, freq).to(device)
            lags_deseas = lags.clone()
            lags_deseas[:, 0:1, :] = ctx_deseas.unsqueeze(1)
            lags_normed = (lags_deseas - loc.unsqueeze(1)) / scale.unsqueeze(1)

            # Time features
            ptf_t = past_tf.transpose(1, 2) if past_tf.shape[2] < past_tf.shape[1] else past_tf
            tf_ctx = ptf_t[:, :, -ctx_len:]
            ctx_channels = torch.cat([lags_normed, tf_ctx], dim=1)

            ftf_t = future_tf.transpose(1, 2) if future_tf.shape[2] < future_tf.shape[1] else future_tf
            ft_feat = ftf_t[:, :, :pred_len]

            loss = solar_meanflow_loss(net, fut_normed, ctx_channels, ft_feat)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                for p, pe in zip(net.parameters(), net_ema.parameters()):
                    pe.data.lerp_(p.data, 1e-4)
            epoch_loss += loss.item(); n_b += 1

        scheduler.step()
        if (epoch+1) % 20 == 0 or epoch == 0:
            logger.info(f"Ep {epoch+1:>4}/{args.epochs} | Loss: {epoch_loss/n_b:.4f} | "
                        f"lr={scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s")

        if (epoch+1) % 50 == 0 or (epoch+1) == args.epochs:
            net_ema.eval()
            test_transform = transformation.apply(dataset.test, is_train=False)
            test_splitter = InstanceSplitter(
                target_field="target", is_pad_field="is_pad", start_field="start",
                forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
                past_length=ctx_len + max_lag, future_length=pred_len,
                time_series_fields=["time_feat", "observed_values"],
            )
            forecaster = SolarForecaster(net_ema, ctx_len, pred_len, num_samples=16).to(device)
            predictor = PyTorchPredictor(
                prediction_length=pred_len,
                input_names=["past_target", "past_observed_values", "past_time_feat", "future_time_feat"],
                prediction_net=forecaster, batch_size=128,
                input_transform=test_splitter, device=device,
            )
            fc_it, ts_it = make_evaluation_predictions(dataset=test_transform, predictor=predictor, num_samples=16)
            forecasts = list(fc_it); tss = list(ts_it)
            metrics, _ = Evaluator(num_workers=0)(tss, forecasts)
            crps = metrics["mean_wQuantileLoss"]
            if crps < best_crps:
                best_crps = crps
                torch.save({'net': net.state_dict(), 'net_ema': net_ema.state_dict(),
                            'epoch': epoch+1, 'crps': crps}, os.path.join(outdir, 'best.pt'))
            torch.save({'net': net.state_dict(), 'net_ema': net_ema.state_dict(),
                        'epoch': epoch+1, 'crps': crps}, os.path.join(outdir, 'latest.pt'))
            beat = " *** BEATS TSFLOW! ***" if crps < 0.341 else ""
            logger.info(f"  CRPS={crps:.6f} | TSFlow=0.341 | Best={best_crps:.6f}{beat}")

    logger.info(f"\nFINAL Solar: Best={best_crps:.6f}, TSFlow=0.341")


def train_exchange(args):
    from meanflow_ts.model_exchange import (
        ExchangeMeanFlowNet, exchange_meanflow_loss, ExchangeForecaster,
        build_exchange_context,
    )

    name = "exchange_rate_nips"
    freq, ctx_len, pred_len = "B", 30, 30
    max_lag = LAG_MAP["B"]
    n_lags = len(get_lag_indices_v4(freq))
    n_tf = len(time_features_from_frequency_str(freq))
    device = torch.device("cuda")
    torch.manual_seed(42); np.random.seed(42)

    outdir = os.path.join(os.path.dirname(__file__), '..', 'results_exchange', name)
    os.makedirs(outdir, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"Exchange Specialized Training (small model + augmentation)")
    logger.info(f"{'='*60}")

    dataset = get_dataset(name)
    transformation = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(start_field="start", target_field="target", output_field="time_feat",
                        time_features=time_features_from_frequency_str(freq), pred_length=pred_len),
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
        Cached(transformed_data), batch_size=args.batch_size, stack_fn=batchify,
        transform=train_splitter, num_batches_per_epoch=100, shuffle_buffer_length=2000,
    )

    norm = RobustNorm()
    net = ExchangeMeanFlowNet(
        pred_len=pred_len, ctx_len=ctx_len, d_model=96, n_s4d_blocks=4,
        ssm_dim=32, dropout=0.2, n_lags=n_lags, n_time_features=n_tf,
        augment_noise=0.01,
    ).to(device)
    net_ema = deepcopy(net).eval()
    logger.info(f"Params: {sum(p.numel() for p in net.parameters()):,}")

    optimizer = AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    best_crps = float('inf')

    for epoch in range(args.epochs):
        net.train()
        epoch_loss, n_b = 0, 0
        t0 = time.time()
        for batch in train_loader:
            past = batch["past_target"].to(device)
            future = batch["future_target"].to(device)
            past_tf = batch["past_time_feat"].to(device)
            future_tf = batch["future_time_feat"].to(device)
            ctx = past[:, -ctx_len:]

            ctx_normed, loc, scale = norm(ctx)
            fut_normed = (future - loc) / scale

            ctx_channels = build_exchange_context(past, ctx_len, freq, past_tf)
            n_lags_ch = 1 + n_lags
            ctx_channels_normed = ctx_channels.clone()
            ctx_channels_normed[:, :n_lags_ch] = (ctx_channels[:, :n_lags_ch] - loc.unsqueeze(1)) / scale.unsqueeze(1)

            ftf = future_tf
            if ftf.shape[2] < ftf.shape[1]:
                ftf = ftf.transpose(1, 2)
            ft_feat = ftf[:, :, :pred_len]

            loss = exchange_meanflow_loss(net, fut_normed, ctx_channels_normed, ft_feat)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                for p, pe in zip(net.parameters(), net_ema.parameters()):
                    pe.data.lerp_(p.data, 1e-4)
            epoch_loss += loss.item(); n_b += 1

        scheduler.step()
        if (epoch+1) % 20 == 0 or epoch == 0:
            logger.info(f"Ep {epoch+1:>4}/{args.epochs} | Loss: {epoch_loss/n_b:.4f} | "
                        f"lr={scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s")

        if (epoch+1) % 50 == 0 or (epoch+1) == args.epochs:
            net_ema.eval()
            test_transform = transformation.apply(dataset.test, is_train=False)
            test_splitter = InstanceSplitter(
                target_field="target", is_pad_field="is_pad", start_field="start",
                forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
                past_length=ctx_len + max_lag, future_length=pred_len,
                time_series_fields=["time_feat", "observed_values"],
            )
            forecaster = ExchangeForecaster(net_ema, ctx_len, pred_len, num_samples=16, freq=freq).to(device)
            predictor = PyTorchPredictor(
                prediction_length=pred_len,
                input_names=["past_target", "past_observed_values", "past_time_feat", "future_time_feat"],
                prediction_net=forecaster, batch_size=128,
                input_transform=test_splitter, device=device,
            )
            fc_it, ts_it = make_evaluation_predictions(dataset=test_transform, predictor=predictor, num_samples=16)
            forecasts = list(fc_it); tss = list(ts_it)
            metrics, _ = Evaluator(num_workers=0)(tss, forecasts)
            crps = metrics["mean_wQuantileLoss"]
            if crps < best_crps:
                best_crps = crps
                torch.save({'net': net.state_dict(), 'net_ema': net_ema.state_dict(),
                            'epoch': epoch+1, 'crps': crps}, os.path.join(outdir, 'best.pt'))
            torch.save({'net': net.state_dict(), 'net_ema': net_ema.state_dict(),
                        'epoch': epoch+1, 'crps': crps}, os.path.join(outdir, 'latest.pt'))
            beat = " *** BEATS TSFLOW! ***" if crps < 0.005 else ""
            logger.info(f"  CRPS={crps:.6f} | TSFlow=0.005 | Best={best_crps:.6f}{beat}")

    logger.info(f"\nFINAL Exchange: Best={best_crps:.6f}, TSFlow=0.005")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=["solar_nips", "exchange_rate_nips"])
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.dataset == "solar_nips":
        train_solar(args)
    else:
        train_exchange(args)


if __name__ == "__main__":
    main()
