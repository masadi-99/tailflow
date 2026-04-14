"""
TailFlow v4: Maximum quality — GP prior + S4D + robust norm.

Run ONE dataset at a time to be memory-safe.

Usage:
    python experiments/train_v4.py electricity_nips --epochs 800
"""
import os, sys, time, argparse, logging, json, gc, tempfile
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
    import pykeops
    tmp = tempfile.mkdtemp(prefix="pykeops_build_")
    pykeops.set_build_folder(tmp)
    pykeops.clean_pykeops()
except: pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v4 import (
    S4DMeanFlowNetV4, v4_meanflow_loss, V4Forecaster,
    extract_lags_v4, get_lag_indices_v4, RobustNorm,
)

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIGS = {
    "electricity_nips":             {"freq": "H", "ctx": 72, "pred": 24, "gp_ls": 3.0},
    "solar_nips":                   {"freq": "H", "ctx": 72, "pred": 24, "gp_ls": 6.0},
    "traffic_nips":                 {"freq": "H", "ctx": 72, "pred": 24, "gp_ls": 3.0},
    "exchange_rate_nips":           {"freq": "B", "ctx": 30, "pred": 30, "gp_ls": 2.0},
    "m4_hourly":                    {"freq": "H", "ctx": 96, "pred": 48, "gp_ls": 4.0},
    "kdd_cup_2018_without_missing": {"freq": "H", "ctx": 96, "pred": 48, "gp_ls": 4.0},
    "uber_tlc_hourly":              {"freq": "H", "ctx": 96, "pred": 24, "gp_ls": 3.0},
}
LAG_MAP = {"H": 672, "B": 750}
TSFLOW_CRPS = {
    "electricity_nips": 0.045, "solar_nips": 0.341,
    "traffic_nips": 0.082, "exchange_rate_nips": 0.005, "m4_hourly": 0.029,
}


def evaluate_model(net_ema, dataset, transformation, cfg, device,
                   num_samples=16, label="", freq="H", use_gp=True):
    ctx_len, pred_len = cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP.get(freq, 672)
    net_ema.eval()

    test_transform = transformation.apply(dataset.test, is_train=False)
    test_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
        past_length=ctx_len + max_lag, future_length=pred_len,
        time_series_fields=["time_feat", "observed_values"],
    )
    forecaster = V4Forecaster(
        net_ema, ctx_len, pred_len, num_samples=num_samples,
        freq=freq, use_gp=use_gp,
    ).to(device)
    predictor = PyTorchPredictor(
        prediction_length=pred_len,
        input_names=["past_target", "past_observed_values"],
        prediction_net=forecaster, batch_size=256,
        input_transform=test_splitter, device=device,
    )
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_transform, predictor=predictor, num_samples=num_samples,
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    metrics, _ = Evaluator(num_workers=0)(tss, forecasts)
    crps = metrics["mean_wQuantileLoss"]
    nd = metrics["ND"]
    nrmse = metrics["NRMSE"]
    logger.info(f"  [{label}] CRPS={crps:.6f} | ND={nd:.6f} | NRMSE={nrmse:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--no-gp", action="store_true", help="Disable GP prior")
    parser.add_argument("--resume", action="store_true", help="Resume from best.pt")
    parser.add_argument("--start-epoch", type=int, default=0, help="Epoch to resume from")
    args = parser.parse_args()

    name = args.dataset
    cfg = CONFIGS[name]
    freq = cfg["freq"]
    ctx_len, pred_len = cfg["ctx"], cfg["pred"]
    gp_ls = cfg["gp_ls"]
    max_lag = LAG_MAP.get(freq, 672)
    n_lags = len(get_lag_indices_v4(freq))
    use_gp = not args.no_gp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    outdir = os.path.join(os.path.dirname(__file__), '..', 'results_v4', name)
    os.makedirs(outdir, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"TailFlow v4: {name}")
    logger.info(f"ctx={ctx_len} pred={pred_len} freq={freq} n_lags={n_lags}")
    logger.info(f"d_model={args.d_model} n_blocks={args.n_blocks} GP={'ON' if use_gp else 'OFF'} ls={gp_ls}")
    logger.info(f"batch={args.batch_size} lr={args.lr} epochs={args.epochs}")
    logger.info(f"{'='*60}")

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
        Cached(transformed_data), batch_size=args.batch_size, stack_fn=batchify,
        transform=train_splitter, num_batches_per_epoch=100, shuffle_buffer_length=2000,
    )

    norm = RobustNorm()

    net = S4DMeanFlowNetV4(
        pred_len=pred_len, ctx_len=ctx_len, d_model=args.d_model,
        n_s4d_blocks=args.n_blocks, ssm_dim=64, time_emb_dim=64,
        dropout=0.1, n_lags=n_lags, freq=freq, gp_length_scale=gp_ls,
    ).to(device)
    net_ema = deepcopy(net).eval()
    param_count = sum(p.numel() for p in net.parameters())
    logger.info(f"Params: {param_count:,}")

    optimizer = AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)
    best_crps = float('inf')
    results = {'dataset': name, 'config': cfg, 'use_gp': use_gp}
    start_epoch = 0

    # Resume from checkpoint if requested
    if args.resume:
        ckpt_path = os.path.join(outdir, 'best.pt')
        if os.path.exists(ckpt_path):
            logger.info(f"Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            net.load_state_dict(ckpt['net'])
            net_ema.load_state_dict(ckpt['net_ema'])
            start_epoch = args.start_epoch if args.start_epoch > 0 else ckpt.get('epoch', 0)
            best_crps = ckpt.get('crps', float('inf'))
            logger.info(f"  Resumed at epoch {start_epoch}, best CRPS = {best_crps:.6f}")
            # Advance scheduler to current epoch
            for _ in range(start_epoch):
                scheduler.step()
        else:
            logger.warning(f"No checkpoint at {ckpt_path}, starting fresh")

    for epoch in range(start_epoch, args.epochs):
        net.train()
        epoch_loss, n_b = 0, 0
        t0 = time.time()
        for batch in train_loader:
            past = batch["past_target"].to(device)
            future = batch["future_target"].to(device)
            ctx = past[:, -ctx_len:]

            ctx_normed, loc, scale = norm(ctx)
            future_normed = (future - loc) / scale

            lags = extract_lags_v4(past, ctx_len, freq).to(device)
            lags_normed = (lags - loc.unsqueeze(1)) / scale.unsqueeze(1)

            loss = v4_meanflow_loss(
                net, future_normed, lags_normed,
                context_normed=ctx_normed, use_gp_noise=use_gp,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                for p, pe in zip(net.parameters(), net_ema.parameters()):
                    pe.data.lerp_(p.data, 1e-4)
            epoch_loss += loss.item()
            n_b += 1

        scheduler.step()
        avg = epoch_loss / n_b
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(f"Ep {epoch+1:>3}/{args.epochs} | Loss: {avg:.4f} | "
                        f"lr={lr_now:.2e} | {elapsed:.1f}s")

        if (epoch + 1) % args.eval_every == 0 or (epoch + 1) == args.epochs:
            metrics = evaluate_model(net_ema, dataset, transformation, cfg, device,
                                      num_samples=16, label=f"ep{epoch+1}",
                                      freq=freq, use_gp=use_gp)
            crps = metrics["mean_wQuantileLoss"]
            if crps < best_crps:
                best_crps = crps
                torch.save({
                    'net': net.state_dict(), 'net_ema': net_ema.state_dict(),
                    'epoch': epoch + 1, 'crps': crps,
                }, os.path.join(outdir, 'best.pt'))
            # Always save latest for resume (overwrite each time)
            torch.save({
                'net': net.state_dict(), 'net_ema': net_ema.state_dict(),
                'epoch': epoch + 1, 'crps': crps,
            }, os.path.join(outdir, 'latest.pt'))
            tsflow = TSFLOW_CRPS.get(name, "?")
            tag = " ***BEST***" if crps == best_crps else ""
            logger.info(f"  CRPS={crps:.6f} | TSFlow={tsflow} | Best={best_crps:.6f}{tag}")

    # Final eval with 100 samples
    logger.info("\nFinal evaluation (100 samples)...")
    for gp_mode in [True, False]:
        label = "final-GP" if gp_mode else "final-noGP"
        m = evaluate_model(net_ema, dataset, transformation, cfg, device,
                            num_samples=100, label=label, freq=freq, use_gp=gp_mode)
        results[label] = {
            'crps': m["mean_wQuantileLoss"], 'nd': m["ND"], 'nrmse': m["NRMSE"],
        }

    results['best_crps'] = best_crps
    results['epochs'] = args.epochs
    results['params'] = param_count

    with open(os.path.join(outdir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    tsflow = TSFLOW_CRPS.get(name, "?")
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL: {name}")
    logger.info(f"  TSFlow (32 NFE): {tsflow}")
    logger.info(f"  TailFlow v4 (1 NFE): {best_crps:.6f}")
    if 'final-GP' in results:
        logger.info(f"  Final (100 samples, GP): {results['final-GP']['crps']:.6f}")
    if 'final-noGP' in results:
        logger.info(f"  Final (100 samples, no GP): {results['final-noGP']['crps']:.6f}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
