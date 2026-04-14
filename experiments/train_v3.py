"""
TailFlow v3: S4D backbone + lag features + RevIN.

Two phases:
  Phase 1: Train S4D-MeanFlow base model (no conditioning)
  Phase 2: Add extremity adapter + fine-tune

Usage:
    python experiments/train_v3.py electricity_nips
    python experiments/train_v3.py exchange_rate_nips --phase1-epochs 600
"""
import os, sys, time, argparse, logging, json, pickle, gc, tempfile
import numpy as np
import torch
import torch.nn as nn
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
from meanflow_ts.model_v3 import (
    S4DMeanFlowNet, s4d_meanflow_loss, S4DMeanFlowForecaster,
    extract_lags, RevIN, get_lag_indices,
    S4DConditionedNet,
)
from meanflow_ts.model_tail import QuantileMapper, compute_raw_extremity

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIGS = {
    "electricity_nips":   {"freq": "H", "ctx": 72, "pred": 24},   # 3 days context
    "solar_nips":         {"freq": "H", "ctx": 72, "pred": 24},
    "traffic_nips":       {"freq": "H", "ctx": 72, "pred": 24},
    "exchange_rate_nips": {"freq": "B", "ctx": 30, "pred": 30},
    "m4_hourly":          {"freq": "H", "ctx": 96, "pred": 48},
}
LAG_MAP = {"H": 672, "B": 750}
TSFLOW_CRPS = {
    "electricity_nips": 0.045, "solar_nips": 0.341,
    "traffic_nips": 0.082, "exchange_rate_nips": 0.005, "m4_hourly": 0.029,
}


def evaluate_model(net_ema, dataset, transformation, cfg, device,
                   num_samples=16, label="", freq="H"):
    """Evaluate using GluonTS pipeline."""
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
    forecaster = S4DMeanFlowForecaster(
        net_ema, ctx_len, pred_len, num_samples=num_samples, freq=freq
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
    parser.add_argument("--phase1-epochs", type=int, default=600)
    parser.add_argument("--phase2-epochs", type=int, default=200)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--ssm-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=50)
    args = parser.parse_args()

    name = args.dataset
    cfg = CONFIGS[name]
    freq, ctx_len, pred_len = cfg["freq"], cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP.get(freq, 672)
    n_lags = len(get_lag_indices(freq))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    outdir = os.path.join(os.path.dirname(__file__), '..', 'results_v3', name)
    os.makedirs(outdir, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"TailFlow v3 (S4D): {name}")
    logger.info(f"ctx={ctx_len} pred={pred_len} freq={freq} n_lags={n_lags}")
    logger.info(f"d_model={args.d_model} n_blocks={args.n_blocks} ssm_dim={args.ssm_dim}")
    logger.info(f"batch={args.batch_size} lr={args.lr}")
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
        transform=train_splitter, num_batches_per_epoch=50, shuffle_buffer_length=2000,
    )

    revin = RevIN()
    all_results = {'dataset': name, 'config': cfg}

    # ============================================================
    # Phase 1: Train S4D base model
    # ============================================================
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 1: Train S4D-MeanFlow base model")
    logger.info(f"{'='*60}")

    net = S4DMeanFlowNet(
        pred_len=pred_len, ctx_len=ctx_len, d_model=args.d_model,
        n_s4d_blocks=args.n_blocks, ssm_dim=args.ssm_dim,
        time_emb_dim=64, dropout=0.1, n_lags=n_lags, freq=freq,
    ).to(device)
    net_ema = deepcopy(net).eval()
    param_count = sum(p.numel() for p in net.parameters())
    logger.info(f"Params: {param_count:,}")

    optimizer = AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.phase1_epochs, eta_min=1e-5)
    best_crps = float('inf')

    for epoch in range(args.phase1_epochs):
        net.train()
        epoch_loss, n_b = 0, 0
        t0 = time.time()
        for batch in train_loader:
            past = batch["past_target"].to(device)
            future = batch["future_target"].to(device)
            ctx = past[:, -ctx_len:]

            # RevIN normalization
            ctx_normed, mean, std = revin(ctx)
            future_normed = (future - mean) / std

            # Extract lag features (normalized)
            lags = extract_lags(past, ctx_len, freq).to(device)
            lags_normed = (lags - mean.unsqueeze(1)) / std.unsqueeze(1)

            loss = s4d_meanflow_loss(net, future_normed, lags_normed)
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
            logger.info(f"P1 Ep {epoch+1:>3}/{args.phase1_epochs} | Loss: {avg:.4f} | "
                        f"lr={lr_now:.2e} | {elapsed:.1f}s")

        if (epoch + 1) % args.eval_every == 0 or (epoch + 1) == args.phase1_epochs:
            metrics = evaluate_model(net_ema, dataset, transformation, cfg, device,
                                      num_samples=16, label=f"P1-ep{epoch+1}", freq=freq)
            crps = metrics["mean_wQuantileLoss"]
            if crps < best_crps:
                best_crps = crps
                torch.save({
                    'net': net.state_dict(), 'net_ema': net_ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1, 'crps': crps,
                }, os.path.join(outdir, 'phase1_best.pt'))
            tsflow = TSFLOW_CRPS.get(name, "?")
            tag = " ***BEST***" if crps == best_crps else ""
            logger.info(f"  CRPS={crps:.6f} | TSFlow={tsflow} | Best={best_crps:.6f}{tag}")

    all_results['phase1'] = {'best_crps': best_crps, 'epochs': args.phase1_epochs}

    # Full eval with 100 samples
    logger.info("\nPhase 1 final evaluation (100 samples)...")
    metrics = evaluate_model(net_ema, dataset, transformation, cfg, device,
                              num_samples=100, label="P1-final", freq=freq)
    all_results['phase1_final'] = {
        'crps': metrics["mean_wQuantileLoss"], 'nd': metrics["ND"],
    }

    # ============================================================
    # Phase 2: Add extremity adapter + fine-tune
    # ============================================================
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 2: Add extremity adapter + fine-tune")
    logger.info(f"{'='*60}")

    # Precompute extremity scores
    logger.info("Precomputing extremity scores...")
    futures_list = []
    for entry in dataset.train:
        ts = np.array(entry["target"], dtype=np.float32)
        stride = max(pred_len, 1)
        for start in range(0, len(ts) - ctx_len - pred_len + 1, stride):
            ctx_np = ts[start:start + ctx_len]
            fut_np = ts[start + ctx_len:start + ctx_len + pred_len]
            m, s = np.abs(ctx_np).mean(), max(ctx_np.std(), 1e-6)
            futures_list.append((fut_np - m) / s)
    futures_arr = np.array(futures_list[:50000])  # cap for memory
    scores = compute_raw_extremity(torch.tensor(futures_arr)).numpy()
    qmap = QuantileMapper(n_bins=10)
    qmap.fit(scores)
    with open(os.path.join(outdir, 'quantile_mapper.pkl'), 'wb') as f:
        pickle.dump(qmap, f)
    logger.info(f"  {len(futures_arr)} windows, extremity: mean={scores.mean():.3f}")
    del futures_arr, futures_list; gc.collect()

    # Create conditioned model
    cond_net = S4DConditionedNet(net, cfg_drop_prob=0.2).to(device)
    cond_ema = deepcopy(cond_net).eval()

    optimizer_p2 = AdamW([
        {'params': cond_net.base.parameters(), 'lr': 1e-4},
        {'params': cond_net.adapter.parameters(), 'lr': 6e-4},
    ], weight_decay=0.01)
    best_crps_p2 = float('inf')

    for epoch in range(args.phase2_epochs):
        cond_net.train()
        epoch_loss, n_b = 0, 0
        t0 = time.time()
        for batch in train_loader:
            past = batch["past_target"].to(device)
            future = batch["future_target"].to(device)
            ctx = past[:, -ctx_len:]
            ctx_normed, mean, std = revin(ctx)
            future_normed = (future - mean) / std
            lags = extract_lags(past, ctx_len, freq).to(device)
            lags_normed = (lags - mean.unsqueeze(1)) / std.unsqueeze(1)

            with torch.no_grad():
                ext_scores = compute_raw_extremity(future_normed)
                ext_q = qmap.to_quantile(ext_scores)

            # Use conditioned loss
            B = future_normed.shape[0]
            e = torch.randn_like(future_normed)
            from meanflow_ts.model_v3 import sample_t_r
            t_s, r_s = sample_t_r(B, device)
            t_bc, r_bc = t_s.unsqueeze(-1), r_s.unsqueeze(-1)
            z = (1 - t_bc) * future_normed + t_bc * e
            v = e - future_normed

            def u_func(z, t_bc, r_bc):
                h_bc = t_bc - r_bc
                return cond_net(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)), lags_normed, ext_q)

            with torch.amp.autocast("cuda", enabled=False):
                u_pred, dudt = torch.func.jvp(
                    u_func, (z, t_bc, r_bc),
                    (v, torch.ones_like(t_bc), torch.zeros_like(r_bc)),
                )
                u_tgt = (v - (t_bc - r_bc) * dudt).detach()
                loss = (u_pred - u_tgt) ** 2
                loss = loss.sum(dim=1)
                adp_wt = (loss.detach() + 1e-3) ** 0.75
                loss = (loss / adp_wt).mean()

            optimizer_p2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cond_net.parameters(), 0.5)
            optimizer_p2.step()
            with torch.no_grad():
                for p, pe in zip(cond_net.parameters(), cond_ema.parameters()):
                    pe.data.lerp_(p.data, 2e-4)
            epoch_loss += loss.item()
            n_b += 1

        avg = epoch_loss / n_b
        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(f"P2 Ep {epoch+1:>3}/{args.phase2_epochs} | Loss: {avg:.4f} | {time.time()-t0:.1f}s")

        if (epoch + 1) % 50 == 0 or (epoch + 1) == args.phase2_epochs:
            # Evaluate with neutral conditioning (use base model path)
            cond_ema.eval()
            # Quick eval via base forecaster (neutral = no adapter effect)
            metrics = evaluate_model(net_ema, dataset, transformation, cfg, device,
                                      num_samples=16, label=f"P2-ep{epoch+1}", freq=freq)
            crps = metrics["mean_wQuantileLoss"]
            if crps < best_crps_p2:
                best_crps_p2 = crps
                torch.save({
                    'cond_net': cond_net.state_dict(), 'cond_ema': cond_ema.state_dict(),
                    'epoch': epoch + 1, 'crps': crps,
                }, os.path.join(outdir, 'phase2_best.pt'))
            logger.info(f"  P2 CRPS={crps:.6f} | Best={best_crps_p2:.6f}")

    all_results['phase2'] = {'best_crps': best_crps_p2}

    # Save
    with open(os.path.join(outdir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    tsflow = TSFLOW_CRPS.get(name, "?")
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL: {name}")
    logger.info(f"  TSFlow (32 NFE): {tsflow}")
    logger.info(f"  Phase 1 (1 NFE): {all_results['phase1']['best_crps']:.6f}")
    logger.info(f"  Phase 2 (1 NFE): {all_results['phase2']['best_crps']:.6f}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
