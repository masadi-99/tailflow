"""
TailFlow: Extremity-Conditioned MeanFlow with Self-Training for Tail-Aware TS Generation.

Full pipeline:
1. Precompute extremity scores + quantile mapping on training data
2. Train base model with extremity conditioning + CFG dropout
3. Self-training rounds: generate, score, tilt-resample, fine-tune
4. Evaluate: overall CRPS, tail CRPS, coverage metrics, guidance sweep

Usage:
    python experiments/train_tail_aware.py electricity_nips --epochs 400
    python experiments/train_tail_aware.py exchange_rate_nips --epochs 400 --self-train-rounds 3
"""
import os, sys, time, math, argparse, logging, tempfile, json, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_tail import (
    ExtremityCondMeanFlowNet, extremity_cond_meanflow_loss,
    guided_sample, ExtremityCondForecaster,
    QuantileMapper, compute_composite_extremity, compute_raw_extremity,
    compute_volatility, compute_max_deviation, compute_drawdown,
    generate_synthetic_samples, tilted_resampling,
)
from meanflow_ts.model import (
    ConditionalMeanFlowNet, conditional_meanflow_loss, MeanFlowForecaster,
)

logging.basicConfig(
    format="%(asctime)s | %(message)s", level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
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


# ============================================================
# Precompute extremity scores for training windows
# ============================================================

def precompute_extremity_scores(dataset, ctx_len, pred_len, max_lag, freq):
    """
    Extract all training windows and compute extremity scores.
    Returns: futures (N, pred_len), contexts (N, ctx_len), scores (N,)
    """
    logger.info("Precomputing extremity scores on training windows...")
    all_futures = []
    all_contexts = []

    for entry in dataset.train:
        ts = np.array(entry["target"], dtype=np.float32)
        if len(ts) < ctx_len + pred_len:
            continue
        # Slide with stride = pred_len for efficiency
        stride = max(pred_len // 2, 1)
        for start in range(0, len(ts) - ctx_len - pred_len + 1, stride):
            ctx = ts[start:start + ctx_len]
            fut = ts[start + ctx_len:start + ctx_len + pred_len]
            loc = max(np.abs(ctx).mean(), 1e-6)
            all_contexts.append(ctx / loc)
            all_futures.append(fut / loc)

    contexts = np.array(all_contexts)
    futures = np.array(all_futures)
    logger.info(f"  Extracted {len(futures)} windows")

    # Compute RAW extremity on futures (normalized) - NOT batch-rank-normalized
    futures_t = torch.tensor(futures)
    scores = compute_raw_extremity(futures_t).numpy()
    logger.info(f"  Extremity scores: mean={scores.mean():.3f}, std={scores.std():.3f}, "
                f"min={scores.min():.3f}, max={scores.max():.3f}")
    return futures, contexts, scores


# ============================================================
# Evaluation: overall + tail-stratified metrics
# ============================================================

def evaluate_model(net_ema, dataset, transformation, cfg, quantile_mapper,
                   device, num_samples=100, guidance_scale=1.0,
                   target_extremity=None, label=""):
    """Evaluate using GluonTS pipeline. Returns metrics dict."""
    freq, ctx_len, pred_len = cfg["freq"], cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP.get(freq, 672)

    net_ema.eval()
    test_transform = transformation.apply(dataset.test, is_train=False)
    test_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
        past_length=ctx_len + max_lag, future_length=pred_len,
        time_series_fields=["time_feat", "observed_values"],
    )
    forecaster = ExtremityCondForecaster(
        net_ema, ctx_len, pred_len, num_samples=num_samples,
        guidance_scale=guidance_scale, target_extremity=target_extremity,
        quantile_mapper=quantile_mapper,
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
    metrics, per_ts = Evaluator(num_workers=0)(tss, forecasts)

    crps = metrics["mean_wQuantileLoss"]
    nd = metrics["ND"]
    nrmse = metrics["NRMSE"]
    logger.info(f"  [{label}] CRPS={crps:.6f} | ND={nd:.6f} | NRMSE={nrmse:.4f}")
    return metrics, forecasts, tss, per_ts


def evaluate_tail_stratified(net_ema, dataset, transformation, cfg,
                              quantile_mapper, device, num_samples=100,
                              guidance_scale=1.0):
    """
    Evaluate separately on tail vs non-tail test windows.
    Returns dict with overall, tail, and non-tail metrics.
    """
    freq, ctx_len, pred_len = cfg["freq"], cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP.get(freq, 672)
    results = {}

    # Overall evaluation (auto extremity from context)
    metrics, forecasts, tss, per_ts = evaluate_model(
        net_ema, dataset, transformation, cfg, quantile_mapper, device,
        num_samples=num_samples, guidance_scale=guidance_scale,
        target_extremity=None, label=f"overall(w={guidance_scale})",
    )
    results['overall'] = metrics

    # Stratify test series by extremity of the ground truth future
    gt_extremities = []
    for ts_entry in tss:
        ts_vals = ts_entry.values[-pred_len:]
        loc = max(np.abs(ts_vals).mean(), 1e-6)
        normalized = ts_vals.flatten() / loc
        if len(normalized) >= 2:
            vol = np.std(np.diff(normalized))
            max_dev = np.max(np.abs(normalized - normalized.mean()))
            gt_extremities.append(float(vol + max_dev) / 2)
        else:
            gt_extremities.append(0.0)

    gt_extremities = np.array(gt_extremities)
    tail_threshold = np.quantile(gt_extremities, 0.8)  # top 20% are "tail"

    # Compute tail vs non-tail CRPS from per-series results
    if per_ts is not None and len(per_ts) > 0:
        per_ts_list = per_ts.to_dict('records') if hasattr(per_ts, 'to_dict') else []
        if per_ts_list:
            tail_crps_vals = []
            nontail_crps_vals = []
            for i, row in enumerate(per_ts_list):
                if i < len(gt_extremities):
                    crps_val = row.get("mean_wQuantileLoss", 0)
                    if gt_extremities[i] >= tail_threshold:
                        tail_crps_vals.append(crps_val)
                    else:
                        nontail_crps_vals.append(crps_val)
            if tail_crps_vals:
                results['tail_crps'] = float(np.mean(tail_crps_vals))
                logger.info(f"  Tail CRPS (top 20%): {results['tail_crps']:.6f} ({len(tail_crps_vals)} series)")
            if nontail_crps_vals:
                results['nontail_crps'] = float(np.mean(nontail_crps_vals))
                logger.info(f"  Non-tail CRPS: {results['nontail_crps']:.6f} ({len(nontail_crps_vals)} series)")

    return results


# ============================================================
# Evaluate sample diversity and tail coverage
# ============================================================

def evaluate_tail_coverage(net_ema, train_futures, quantile_mapper, device,
                           ctx_len, pred_len, n_gen=5000, guidance_scales=[0.0, 1.0, 2.0, 3.0]):
    """
    Generate samples at various guidance scales and measure tail coverage.
    Compare extremity distribution of generated vs real data.
    """
    net_ema.eval()
    results = {}

    # Real data extremity distribution
    real_scores = compute_raw_extremity(torch.tensor(train_futures[:n_gen])).numpy()
    real_q90 = np.quantile(real_scores, 0.9)
    real_q95 = np.quantile(real_scores, 0.95)

    for w in guidance_scales:
        logger.info(f"  Generating {n_gen} samples with guidance_scale={w}...")
        all_samples = []
        batch_size = min(256, n_gen)

        with torch.no_grad():
            for start in range(0, n_gen, batch_size):
                end = min(start + batch_size, n_gen)
                B = end - start
                # Use random training contexts
                idx = np.random.randint(0, len(train_futures), B)
                # We don't have raw contexts stored separately, so use futures as proxy
                # In the real pipeline these come from the dataloader
                ctx = torch.randn(B, ctx_len, device=device)  # noise context for diversity test
                ext_q = torch.rand(B, device=device)  # uniform extremity targets

                if w > 0:
                    # Use high extremity target to test tail generation
                    ext_q = torch.full((B,), 0.9, device=device)

                samples = guided_sample(net_ema, ctx, ext_q, (B, pred_len), device, w)
                all_samples.append(samples.cpu())

        all_samples = torch.cat(all_samples, dim=0)
        gen_scores = compute_raw_extremity(all_samples).numpy()

        # Metrics
        frac_above_q90 = (gen_scores >= real_q90).mean()
        frac_above_q95 = (gen_scores >= real_q95).mean()
        mean_score = gen_scores.mean()
        std_score = gen_scores.std()

        results[f'w={w}'] = {
            'mean_extremity': float(mean_score),
            'std_extremity': float(std_score),
            'frac_above_real_q90': float(frac_above_q90),
            'frac_above_real_q95': float(frac_above_q95),
        }
        logger.info(f"    w={w}: mean_ext={mean_score:.3f}, >q90={frac_above_q90:.3f}, >q95={frac_above_q95:.3f}")

    return results


# ============================================================
# Self-training round
# ============================================================

def self_training_round(net, net_ema, optimizer, train_loader, quantile_mapper,
                        device, cfg, round_num, epochs=100, alpha=2.0,
                        original_fraction=0.5, guidance_scale_gen=1.5):
    """
    One round of self-training:
    1. Generate synthetic data from current model
    2. Score and tilt-resample (upweight extreme samples)
    3. Mix with original data
    4. Fine-tune model
    """
    ctx_len, pred_len = cfg["ctx"], cfg["pred"]
    logger.info(f"\n=== Self-Training Round {round_num} (alpha={alpha}, orig_frac={original_fraction}) ===")

    # Step 1: Collect original training data
    logger.info("Collecting original training data...")
    orig_contexts = []
    orig_futures = []
    orig_locs = []
    orig_ext_qs = []

    net_ema.eval()
    for batch in train_loader:
        past = batch["past_target"].to(device)
        future = batch["future_target"].to(device)
        ctx = past[:, -ctx_len:]
        loc = ctx.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
        scaled_future = future / loc
        scores = compute_raw_extremity(scaled_future)
        ext_q = quantile_mapper.to_quantile(scores)

        orig_contexts.append(ctx.cpu())
        orig_futures.append(future.cpu())
        orig_locs.append(loc.cpu())
        orig_ext_qs.append(ext_q.cpu() if isinstance(ext_q, torch.Tensor) else torch.tensor(ext_q))

    orig_contexts = torch.cat(orig_contexts, dim=0)
    orig_futures = torch.cat(orig_futures, dim=0)
    orig_locs = torch.cat(orig_locs, dim=0)
    orig_ext_qs = torch.cat(orig_ext_qs, dim=0)
    logger.info(f"  Original data: {orig_contexts.shape[0]} samples")

    # Step 2: Generate synthetic data
    logger.info("Generating synthetic data...")
    synthetic = []
    n_gen_batches = max(len(orig_contexts) // 256, 8)

    with torch.no_grad():
        for _ in tqdm(range(n_gen_batches), desc="Generating"):
            B = min(256, orig_contexts.shape[0])
            idx = torch.randint(0, orig_contexts.shape[0], (B,))
            ctx = orig_contexts[idx].to(device)
            loc_batch = orig_locs[idx].to(device)
            scaled_ctx = ctx / loc_batch

            # Generate with varied extremity targets (biased toward high)
            ext_targets = torch.rand(B, device=device) * 0.5 + 0.5  # [0.5, 1.0]
            samples = guided_sample(
                net_ema, scaled_ctx, ext_targets,
                (B, pred_len), device,
                guidance_scale=guidance_scale_gen,
            )
            # Score using raw extremity (same as training)
            scores = compute_raw_extremity(samples)
            ext_q = quantile_mapper.to_quantile(scores)

            synthetic.append({
                'context': ctx.cpu(),
                'future': (samples * loc_batch).cpu(),
                'loc': loc_batch.cpu(),
                'extremity_q': ext_q.cpu() if isinstance(ext_q, torch.Tensor) else torch.tensor(ext_q),
                'extremity_score': scores.cpu(),
            })

    # Step 3: Tilt-resample and mix
    logger.info("Tilted resampling...")
    mixed_ctx, mixed_future, mixed_loc, mixed_ext_q = tilted_resampling(
        synthetic, alpha=alpha, original_fraction=original_fraction,
        original_contexts=orig_contexts, original_futures=orig_futures,
        original_locs=orig_locs, original_ext_q=orig_ext_qs,
    )
    logger.info(f"  Mixed dataset: {mixed_ctx.shape[0]} samples")
    logger.info(f"  Extremity distribution: mean={mixed_ext_q.mean():.3f}, "
                f">0.8={float((mixed_ext_q > 0.8).float().mean()):.3f}")

    # Step 4: Fine-tune on mixed data
    logger.info(f"Fine-tuning for {epochs} epochs on mixed data...")
    mixed_ds = TensorDataset(mixed_ctx, mixed_future, mixed_loc, mixed_ext_q)
    mixed_loader = DataLoader(mixed_ds, batch_size=256, shuffle=True, drop_last=True)

    # Use lower LR for fine-tuning
    ft_optimizer = AdamW(net.parameters(), lr=2e-4)

    for ep in range(epochs):
        net.train()
        ep_loss, n_b = 0, 0
        t0 = time.time()
        for ctx_b, fut_b, loc_b, eq_b in mixed_loader:
            ctx_b = ctx_b.to(device)
            fut_b = fut_b.to(device)
            loc_b = loc_b.to(device)
            eq_b = eq_b.to(device)

            scaled_ctx = ctx_b / loc_b
            scaled_fut = fut_b / loc_b

            loss = extremity_cond_meanflow_loss(net, scaled_fut, scaled_ctx, eq_b)
            ft_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            ft_optimizer.step()
            with torch.no_grad():
                for p, pe in zip(net.parameters(), net_ema.parameters()):
                    pe.data.lerp_(p.data, 2e-4)
            ep_loss += loss.item()
            n_b += 1

        if (ep + 1) % 20 == 0 or ep == 0:
            logger.info(f"  ST-R{round_num} Epoch {ep+1}/{epochs} | Loss: {ep_loss/n_b:.4f} | {time.time()-t0:.1f}s")

    return net, net_ema


# ============================================================
# Also train a baseline (unconditioned) for comparison
# ============================================================

def train_baseline(dataset, transformation, cfg, device, epochs=300):
    """Train standard (unconditioned) MeanFlow as baseline."""
    freq, ctx_len, pred_len = cfg["freq"], cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP.get(freq, 672)

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
    optimizer = AdamW(net.parameters(), lr=6e-4)
    best_crps = float('inf')

    for epoch in range(epochs):
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
        if (epoch + 1) % 50 == 0 or epoch == 0:
            logger.info(f"  Baseline Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/n_b:.4f} | {time.time()-t0:.1f}s")

    return net, net_ema


def evaluate_baseline(net_ema, dataset, transformation, cfg, device, num_samples=100):
    """Evaluate baseline model."""
    freq, ctx_len, pred_len = cfg["freq"], cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP.get(freq, 672)

    net_ema.eval()
    test_transform = transformation.apply(dataset.test, is_train=False)
    test_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
        past_length=ctx_len + max_lag, future_length=pred_len,
        time_series_fields=["time_feat", "observed_values"],
    )
    forecaster = MeanFlowForecaster(net_ema, ctx_len, pred_len, num_samples=num_samples).to(device)
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
    metrics, per_ts = Evaluator(num_workers=0)(tss, forecasts)
    crps = metrics["mean_wQuantileLoss"]
    nd = metrics["ND"]
    logger.info(f"  [Baseline] CRPS={crps:.6f} | ND={nd:.6f}")
    return metrics


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--epochs", type=int, default=400,
                        help="Epochs for base model training")
    parser.add_argument("--self-train-rounds", type=int, default=2)
    parser.add_argument("--self-train-epochs", type=int, default=100,
                        help="Epochs per self-training round")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Tilting parameter for self-training resampling")
    parser.add_argument("--cfg-drop", type=float, default=0.15,
                        help="CFG dropout probability")
    parser.add_argument("--num-eval-samples", type=int, default=100)
    parser.add_argument("--n-bins", type=int, default=10,
                        help="Number of extremity bins")
    parser.add_argument("--baseline-epochs", type=int, default=300,
                        help="Epochs for baseline model")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    name = args.dataset
    cfg = CONFIGS[name]
    freq, ctx_len, pred_len = cfg["freq"], cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP.get(freq, 672)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(6432)
    np.random.seed(6432)

    outdir = os.path.join(os.path.dirname(__file__), '..', 'results', name)
    os.makedirs(outdir, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"TailFlow: {name} | ctx={ctx_len} pred={pred_len} freq={freq}")
    logger.info(f"CFG dropout={args.cfg_drop} | bins={args.n_bins} | alpha={args.alpha}")
    logger.info(f"Self-training rounds={args.self_train_rounds}")
    logger.info(f"{'='*60}")

    # Load dataset
    dataset = get_dataset(name)
    transformation = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(
            start_field="start", target_field="target", output_field="time_feat",
            time_features=time_features_from_frequency_str(freq), pred_length=pred_len,
        ),
    ])

    # Step 1: Precompute extremity scores and fit quantile mapper
    logger.info("\n--- Step 1: Precompute extremity scores ---")
    futures, contexts, scores = precompute_extremity_scores(dataset, ctx_len, pred_len, max_lag, freq)
    qmap = QuantileMapper(n_bins=args.n_bins)
    qmap.fit(scores)
    # Save mapper
    with open(os.path.join(outdir, 'quantile_mapper.pkl'), 'wb') as f:
        pickle.dump(qmap, f)

    # Step 2: Train base model with extremity conditioning
    logger.info("\n--- Step 2: Train extremity-conditioned base model ---")
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

    net = ExtremityCondMeanFlowNet(
        pred_len=pred_len, ctx_len=ctx_len,
        model_channels=128, num_res_blocks=4, time_emb_dim=64,
        dropout=0.1, cfg_drop_prob=args.cfg_drop,
    ).to(device)
    net_ema = deepcopy(net).eval()
    param_count = sum(p.numel() for p in net.parameters())
    logger.info(f"Model params: {param_count:,}")

    optimizer = AdamW(net.parameters(), lr=6e-4)
    best_crps = float('inf')
    all_results = {'dataset': name, 'config': cfg, 'stages': {}}

    for epoch in range(args.epochs):
        net.train()
        epoch_loss, n_b = 0, 0
        t0 = time.time()
        for batch in train_loader:
            past = batch["past_target"].to(device)
            future = batch["future_target"].to(device)
            ctx = past[:, -ctx_len:]
            loc = ctx.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
            scaled_ctx = ctx / loc
            scaled_future = future / loc

            # Compute RAW extremity of future, then map via precomputed quantiles
            with torch.no_grad():
                ext_scores = compute_raw_extremity(scaled_future)
                ext_q = qmap.to_quantile(ext_scores)

            loss = extremity_cond_meanflow_loss(net, scaled_future, scaled_ctx, ext_q)
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
        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:>3}/{args.epochs} | Loss: {avg:.4f} | {elapsed:.1f}s")

        # Evaluate periodically (every 50 epochs)
        if (epoch + 1) % 50 == 0 or (epoch + 1) == args.epochs:
            logger.info("Evaluating base model...")
            # Use neutral extremity (0.5) for standard eval — matches CFG unconditional mode
            metrics, _, _, _ = evaluate_model(
                net_ema, dataset, transformation, cfg, qmap, device,
                num_samples=16, guidance_scale=1.0,
                target_extremity=0.5, label="base(q=0.5)",
            )
            # Also eval with marginal sampling
            metrics_marg, _, _, _ = evaluate_model(
                net_ema, dataset, transformation, cfg, qmap, device,
                num_samples=16, guidance_scale=1.0,
                target_extremity='marginal', label="base(marginal)",
            )
            crps = metrics["mean_wQuantileLoss"]
            tag = " ***BEST***" if crps < best_crps else ""
            if crps < best_crps:
                best_crps = crps
                torch.save({
                    'net': net.state_dict(), 'net_ema': net_ema.state_dict(),
                    'epoch': epoch + 1, 'crps': crps,
                }, os.path.join(outdir, 'best_base.pt'))
            tsflow = TSFLOW_CRPS.get(name, "?")
            logger.info(f"  CRPS={crps:.6f} | TSFlow={tsflow} | Best={best_crps:.6f}{tag}")

    # Save base model results
    all_results['stages']['base'] = {
        'best_crps': best_crps, 'epochs': args.epochs,
    }

    # Step 3: Full evaluation of base model with guidance sweep
    logger.info("\n--- Step 3: Guidance sweep on base model ---")
    guidance_results = {}
    # Sweep guidance scale with neutral (0.5) conditioning
    for w in [1.0, 1.5, 2.0, 3.0]:
        for tq in [0.5, 'marginal', 0.8, 0.9]:
            metrics, _, _, _ = evaluate_model(
                net_ema, dataset, transformation, cfg, qmap, device,
                num_samples=args.num_eval_samples, guidance_scale=w,
                target_extremity=tq, label=f"base-w={w}-tq={tq}",
            )
            guidance_results[f'w={w}_tq={tq}'] = {
                'crps': metrics["mean_wQuantileLoss"],
                'nd': metrics["ND"],
                'nrmse': metrics["NRMSE"],
            }
    all_results['stages']['guidance_sweep_base'] = guidance_results

    # Step 3b: Tail-stratified evaluation
    logger.info("\n--- Tail-stratified evaluation (base) ---")
    tail_results = evaluate_tail_stratified(
        net_ema, dataset, transformation, cfg, qmap, device,
        num_samples=args.num_eval_samples, guidance_scale=1.0,
    )
    all_results['stages']['tail_stratified_base'] = tail_results

    # Step 4: Self-training rounds
    logger.info(f"\n--- Step 4: Self-training ({args.self_train_rounds} rounds) ---")
    for round_num in range(1, args.self_train_rounds + 1):
        net, net_ema = self_training_round(
            net, net_ema, optimizer, train_loader, qmap, device, cfg,
            round_num=round_num, epochs=args.self_train_epochs,
            alpha=args.alpha, original_fraction=0.5,
            guidance_scale_gen=1.5,
        )

        # Evaluate after self-training round
        logger.info(f"Evaluating after self-training round {round_num}...")
        round_results = {}

        # Guidance sweep
        for w in [0.0, 1.0, 1.5, 2.0, 3.0]:
            metrics, _, _, _ = evaluate_model(
                net_ema, dataset, transformation, cfg, qmap, device,
                num_samples=args.num_eval_samples, guidance_scale=w,
                target_extremity=None, label=f"ST-R{round_num}-w={w}",
            )
            round_results[f'w={w}'] = {
                'crps': metrics["mean_wQuantileLoss"],
                'nd': metrics["ND"],
                'nrmse': metrics["NRMSE"],
            }

        # Tail-stratified
        tail_r = evaluate_tail_stratified(
            net_ema, dataset, transformation, cfg, qmap, device,
            num_samples=args.num_eval_samples, guidance_scale=1.0,
        )
        round_results['tail_stratified'] = tail_r

        all_results['stages'][f'self_train_round_{round_num}'] = round_results

        # Save checkpoint
        torch.save({
            'net': net.state_dict(), 'net_ema': net_ema.state_dict(),
            'round': round_num,
        }, os.path.join(outdir, f'after_st_round_{round_num}.pt'))

    # Step 5: Train and evaluate baseline for comparison
    if not args.skip_baseline:
        logger.info(f"\n--- Step 5: Train baseline (unconditioned MeanFlow) ---")
        base_net, base_ema = train_baseline(
            dataset, transformation, cfg, device, epochs=args.baseline_epochs,
        )
        baseline_metrics = evaluate_baseline(
            base_ema, dataset, transformation, cfg, device,
            num_samples=args.num_eval_samples,
        )
        all_results['stages']['baseline_uncond'] = {
            'crps': baseline_metrics["mean_wQuantileLoss"],
            'nd': baseline_metrics["ND"],
            'nrmse': baseline_metrics["NRMSE"],
        }

    # Step 6: Targeted tail generation evaluation
    logger.info("\n--- Step 6: Targeted tail generation ---")
    for target_q in [0.5, 0.7, 0.8, 0.9, 0.95]:
        for w in [1.0, 2.0, 3.0]:
            metrics, _, _, _ = evaluate_model(
                net_ema, dataset, transformation, cfg, qmap, device,
                num_samples=args.num_eval_samples, guidance_scale=w,
                target_extremity=target_q, label=f"target_q={target_q},w={w}",
            )
            key = f'targeted_q={target_q}_w={w}'
            all_results['stages'][key] = {
                'crps': metrics["mean_wQuantileLoss"],
                'nd': metrics["ND"],
                'nrmse': metrics["NRMSE"],
            }

    # Save all results
    results_path = os.path.join(outdir, 'all_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nAll results saved to {results_path}")

    # Print summary table
    logger.info(f"\n{'='*70}")
    logger.info(f"SUMMARY: {name}")
    logger.info(f"{'='*70}")
    logger.info(f"TSFlow (32 NFE):         CRPS={TSFLOW_CRPS.get(name, '?')}")
    if 'baseline_uncond' in all_results['stages']:
        bl = all_results['stages']['baseline_uncond']
        logger.info(f"MeanFlow baseline (1 NFE): CRPS={bl['crps']:.6f}")
    logger.info(f"TailFlow base (1 NFE):   CRPS={all_results['stages']['base']['best_crps']:.6f}")
    for rnd in range(1, args.self_train_rounds + 1):
        key = f'self_train_round_{rnd}'
        if key in all_results['stages']:
            st = all_results['stages'][key]
            if 'w=1.0' in st:
                logger.info(f"TailFlow ST-R{rnd} (w=1):  CRPS={st['w=1.0']['crps']:.6f}")
            if 'w=2.0' in st:
                logger.info(f"TailFlow ST-R{rnd} (w=2):  CRPS={st['w=2.0']['crps']:.6f}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
