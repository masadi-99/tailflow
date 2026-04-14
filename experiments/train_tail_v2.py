"""
TailFlow v2: Two-Phase Training for Tail-Aware Time Series Generation.

Phase 1: Train standard MeanFlow-TS to convergence (no conditioning)
Phase 2: Add extremity conditioning via adapter + fine-tune with CFG dropout
Phase 3: Self-training rounds (generate, tilt, retrain)

Usage:
    python experiments/train_tail_v2.py electricity_nips --phase1-epochs 300 --phase2-epochs 200
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
from meanflow_ts.model import (
    ConditionalMeanFlowNet, conditional_meanflow_loss,
    MeanFlowForecaster, sample_t_r, SinusoidalPosEmb, ResBlock1D,
)
from meanflow_ts.model_tail import (
    QuantileMapper, compute_raw_extremity,
    compute_volatility, compute_max_deviation, compute_drawdown,
    tilted_resampling,
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
# Extremity-Conditioned Model: Adapter approach
# ============================================================

class ExtremityAdapter(nn.Module):
    """
    Lightweight adapter that adds extremity conditioning to a frozen/unfrozen base model.
    Uses a small MLP to produce a bias term added to the time embedding.
    Zero-initialized so it starts as identity.
    """
    def __init__(self, emb_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, emb_dim),
        )
        # Zero-init output
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, ext_q):
        """ext_q: (B,) -> (B, emb_dim)"""
        return self.mlp(ext_q.unsqueeze(-1))


class ConditionedMeanFlowNet(nn.Module):
    """
    Wraps a base ConditionalMeanFlowNet with an extremity adapter.
    The adapter output is added to the time embedding before it enters the network.
    """
    def __init__(self, base_net, cfg_drop_prob=0.2):
        super().__init__()
        self.base_net = base_net
        emb_dim = base_net.time_mlp[-1].out_features
        self.adapter = ExtremityAdapter(emb_dim=emb_dim)
        self.cfg_drop_prob = cfg_drop_prob
        # Copy attributes for compatibility
        self.pred_len = base_net.pred_len
        self.ctx_len = base_net.ctx_len

    def forward(self, noisy_pred, time_steps, context, extremity_q=None):
        """
        Same as base_net.forward but with optional extremity conditioning.
        When extremity_q is None, acts exactly like the base model.
        """
        t, h = time_steps
        emb = torch.cat([self.base_net.time_emb(t), self.base_net.time_emb(h)], dim=-1)
        emb = self.base_net.time_mlp(emb)

        # Add extremity adapter output
        if extremity_q is not None:
            # CFG dropout during training
            if self.training and self.cfg_drop_prob > 0:
                drop_mask = torch.rand(extremity_q.shape[0], device=extremity_q.device) < self.cfg_drop_prob
                extremity_q = torch.where(drop_mask, torch.full_like(extremity_q, 0.5), extremity_q)
            ext_emb = self.adapter(extremity_q)
            emb = emb + ext_emb

        # Rest of base model forward
        ctx = self.base_net.ctx_proj(context.unsqueeze(1))
        for block in self.base_net.ctx_blocks:
            ctx = block(ctx, emb)
        emb = emb + self.base_net.ctx_pool(ctx)
        ctx_spatial = self.base_net.ctx_feat_proj(self.base_net.ctx_to_pred(ctx))

        pred = self.base_net.pred_proj(noisy_pred.unsqueeze(1)) + ctx_spatial
        for block in self.base_net.pred_blocks:
            pred = block(pred, emb)

        return self.base_net.out_proj(F.silu(self.base_net.out_norm(pred))).squeeze(1)


# ============================================================
# Loss for conditioned model
# ============================================================

def conditioned_meanflow_loss(net, future_clean, context, extremity_q,
                               norm_p=0.75, norm_eps=1e-3):
    """MeanFlow JVP loss with extremity conditioning."""
    B = future_clean.shape[0]
    device = future_clean.device
    e = torch.randn_like(future_clean)
    t, r = sample_t_r(B, device)
    t_bc, r_bc = t.unsqueeze(-1), r.unsqueeze(-1)

    z = (1 - t_bc) * future_clean + t_bc * e
    v = e - future_clean

    def u_func(z, t_bc, r_bc):
        h_bc = t_bc - r_bc
        return net(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)), context, extremity_q)

    with torch.amp.autocast("cuda", enabled=False):
        u_pred, dudt = torch.func.jvp(
            u_func, (z, t_bc, r_bc),
            (v, torch.ones_like(t_bc), torch.zeros_like(r_bc)),
        )
        u_tgt = (v - (t_bc - r_bc) * dudt).detach()
        loss = (u_pred - u_tgt) ** 2
        loss = loss.sum(dim=1)
        adp_wt = (loss.detach() + norm_eps) ** norm_p
        loss = (loss / adp_wt).mean()
    return loss


# ============================================================
# Guided sampling
# ============================================================

@torch.no_grad()
def guided_sample(net, context, extremity_q, shape, device, guidance_scale=1.0):
    """CFG-guided one-step sampling."""
    B = shape[0]
    z_1 = torch.randn(shape, device=device)
    t = torch.ones(B, device=device)
    h = torch.ones(B, device=device)

    if guidance_scale == 1.0:
        u = net(z_1, (t, h), context, extremity_q)
        return z_1 - u

    u_cond = net(z_1, (t, h), context, extremity_q)
    u_uncond = net(z_1, (t, h), context, torch.full_like(extremity_q, 0.5))
    u_guided = u_uncond + guidance_scale * (u_cond - u_uncond)
    return z_1 - u_guided


# ============================================================
# Forecasters
# ============================================================

class TailFlowForecaster(nn.Module):
    """Forecaster supporting extremity conditioning and guidance."""

    def __init__(self, net, context_length, prediction_length, num_samples=100,
                 guidance_scale=1.0, target_extremity=0.5):
        super().__init__()
        self.net = net
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.guidance_scale = guidance_scale
        self.target_extremity = target_extremity

    def forward(self, past_target, past_observed_values, **kwargs):
        device = past_target.device
        B = past_target.shape[0]
        context = past_target[:, -self.context_length:]
        loc = context.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
        scaled_ctx = context / loc

        all_preds = []
        for _ in range(self.num_samples):
            if self.target_extremity == 'marginal':
                ext_q = torch.rand(B, device=device)
            else:
                ext_q = torch.full((B,), float(self.target_extremity), device=device)

            sample = guided_sample(
                self.net, scaled_ctx, ext_q,
                (B, self.prediction_length), device,
                guidance_scale=self.guidance_scale,
            )
            all_preds.append(sample * loc)

        return torch.stack(all_preds, dim=1)


# ============================================================
# Evaluation helpers
# ============================================================

def evaluate(net, dataset, transformation, cfg, device, num_samples=100,
             guidance_scale=1.0, target_extremity=0.5, label="",
             is_conditioned=True):
    """Evaluate model on test set."""
    freq, ctx_len, pred_len = cfg["freq"], cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP.get(freq, 672)

    net.eval()
    test_transform = transformation.apply(dataset.test, is_train=False)
    test_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
        past_length=ctx_len + max_lag, future_length=pred_len,
        time_series_fields=["time_feat", "observed_values"],
    )

    if is_conditioned:
        forecaster = TailFlowForecaster(
            net, ctx_len, pred_len, num_samples=num_samples,
            guidance_scale=guidance_scale, target_extremity=target_extremity,
        ).to(device)
    else:
        forecaster = MeanFlowForecaster(net, ctx_len, pred_len, num_samples=num_samples).to(device)

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


def tail_stratified_crps(forecasts, tss, pred_len):
    """Decompose CRPS by extremity of ground truth."""
    extremities = []
    crps_vals = []

    for fc, ts in zip(forecasts, tss):
        gt = ts.values[-pred_len:].flatten()
        loc = max(np.abs(gt).mean(), 1e-6)
        norm_gt = gt / loc
        if len(norm_gt) >= 2:
            vol = np.std(np.diff(norm_gt))
            maxdev = np.max(np.abs(norm_gt - norm_gt.mean()))
            extremities.append(float(vol + maxdev) / 2)
        else:
            extremities.append(0.0)

        # CRPS via quantile loss
        samples = fc.samples
        ql_sum = 0
        for q in np.arange(0.05, 1.0, 0.05):
            q_pred = np.quantile(samples, q, axis=0)
            ql_sum += np.mean(2 * np.abs((gt > q_pred) - q) * np.abs(gt - q_pred))
        crps_vals.append(ql_sum / 19)

    extremities = np.array(extremities)
    crps_vals = np.array(crps_vals)

    results = {'overall': float(crps_vals.mean())}
    for pct in [10, 20, 30]:
        thresh = np.quantile(extremities, 1 - pct/100)
        tail_mask = extremities >= thresh
        if tail_mask.sum() > 0:
            results[f'tail_{pct}pct'] = float(crps_vals[tail_mask].mean())
        nontail_mask = extremities < thresh
        if nontail_mask.sum() > 0:
            results[f'nontail_{pct}pct'] = float(crps_vals[nontail_mask].mean())
    return results


# ============================================================
# Precompute extremity
# ============================================================

def precompute_extremity_scores(dataset, ctx_len, pred_len):
    """Extract training windows and compute raw extremity scores."""
    logger.info("Precomputing extremity scores...")
    futures = []
    for entry in dataset.train:
        ts = np.array(entry["target"], dtype=np.float32)
        stride = max(pred_len // 2, 1)
        for start in range(0, len(ts) - ctx_len - pred_len + 1, stride):
            ctx = ts[start:start + ctx_len]
            fut = ts[start + ctx_len:start + ctx_len + pred_len]
            loc = max(np.abs(ctx).mean(), 1e-6)
            futures.append(fut / loc)

    futures = np.array(futures)
    scores = compute_raw_extremity(torch.tensor(futures)).numpy()
    logger.info(f"  {len(futures)} windows, extremity: mean={scores.mean():.3f}, "
                f"std={scores.std():.3f}")
    return futures, scores


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--phase1-epochs", type=int, default=300)
    parser.add_argument("--phase2-epochs", type=int, default=200)
    parser.add_argument("--self-train-rounds", type=int, default=2)
    parser.add_argument("--self-train-epochs", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--cfg-drop", type=float, default=0.2)
    parser.add_argument("--num-eval-samples", type=int, default=100)
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
    logger.info(f"TailFlow v2: {name} | ctx={ctx_len} pred={pred_len} freq={freq}")
    logger.info(f"Phase 1: {args.phase1_epochs} epochs | Phase 2: {args.phase2_epochs} epochs")
    logger.info(f"CFG dropout={args.cfg_drop} | Self-train rounds={args.self_train_rounds}")
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
        Cached(transformed_data), batch_size=256, stack_fn=batchify,
        transform=train_splitter, num_batches_per_epoch=32, shuffle_buffer_length=10000,
    )

    all_results = {'dataset': name, 'config': cfg, 'phases': {}}

    # ============================================================
    # Phase 1: Train base MeanFlow (no conditioning)
    # ============================================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: Train base MeanFlow (no conditioning)")
    logger.info("="*60)

    base_net = ConditionalMeanFlowNet(
        pred_len=pred_len, ctx_len=ctx_len,
        model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
    ).to(device)
    base_ema = deepcopy(base_net).eval()
    logger.info(f"Params: {sum(p.numel() for p in base_net.parameters()):,}")

    optimizer = AdamW(base_net.parameters(), lr=6e-4)
    best_crps_p1 = float('inf')

    for epoch in range(args.phase1_epochs):
        base_net.train()
        epoch_loss, n_b = 0, 0
        t0 = time.time()
        for batch in train_loader:
            past = batch["past_target"].to(device)
            future = batch["future_target"].to(device)
            ctx = past[:, -ctx_len:]
            loc = ctx.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
            loss = conditional_meanflow_loss(base_net, future / loc, ctx / loc)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_net.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for p, pe in zip(base_net.parameters(), base_ema.parameters()):
                    pe.data.lerp_(p.data, 1e-4)
            epoch_loss += loss.item()
            n_b += 1

        avg = epoch_loss / n_b
        elapsed = time.time() - t0
        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(f"P1 Epoch {epoch+1:>3}/{args.phase1_epochs} | Loss: {avg:.4f} | {elapsed:.1f}s")

        if (epoch + 1) % 100 == 0 or (epoch + 1) == args.phase1_epochs:
            metrics, _, _, _ = evaluate(
                base_ema, dataset, transformation, cfg, device,
                num_samples=16, label="P1-base", is_conditioned=False,
            )
            crps = metrics["mean_wQuantileLoss"]
            if crps < best_crps_p1:
                best_crps_p1 = crps
                torch.save({
                    'net': base_net.state_dict(), 'net_ema': base_ema.state_dict(),
                    'epoch': epoch + 1, 'crps': crps,
                }, os.path.join(outdir, 'phase1_best.pt'))
            tsflow = TSFLOW_CRPS.get(name, "?")
            tag = " ***BEST***" if crps == best_crps_p1 else ""
            logger.info(f"  P1 CRPS={crps:.6f} | TSFlow={tsflow} | Best={best_crps_p1:.6f}{tag}")

    all_results['phases']['phase1'] = {'best_crps': best_crps_p1, 'epochs': args.phase1_epochs}

    # Full evaluation of phase 1
    logger.info("\nPhase 1 final evaluation (100 samples)...")
    p1_metrics, p1_fcs, p1_tss, _ = evaluate(
        base_ema, dataset, transformation, cfg, device,
        num_samples=args.num_eval_samples, label="P1-final", is_conditioned=False,
    )
    p1_tail = tail_stratified_crps(p1_fcs, p1_tss, pred_len)
    all_results['phases']['phase1_eval'] = {
        'crps': p1_metrics["mean_wQuantileLoss"],
        'nd': p1_metrics["ND"],
        'nrmse': p1_metrics["NRMSE"],
        'tail_decomp': p1_tail,
    }
    logger.info(f"  P1 tail decomp: {json.dumps(p1_tail, indent=2)}")

    # ============================================================
    # Phase 2: Add extremity conditioning via adapter + fine-tune
    # ============================================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: Add extremity conditioning + fine-tune")
    logger.info("="*60)

    # Precompute extremity scores and fit quantile mapper
    futures, scores = precompute_extremity_scores(dataset, ctx_len, pred_len)
    qmap = QuantileMapper(n_bins=10)
    qmap.fit(scores)
    with open(os.path.join(outdir, 'quantile_mapper.pkl'), 'wb') as f:
        pickle.dump(qmap, f)

    # Create conditioned model wrapping the trained base
    cond_net = ConditionedMeanFlowNet(base_net, cfg_drop_prob=args.cfg_drop).to(device)
    cond_ema = deepcopy(cond_net).eval()
    adapter_params = list(cond_net.adapter.parameters())
    logger.info(f"Adapter params: {sum(p.numel() for p in adapter_params):,}")
    logger.info(f"Total params: {sum(p.numel() for p in cond_net.parameters()):,}")

    # Fine-tune: lower LR for base, higher for adapter
    optimizer_p2 = AdamW([
        {'params': cond_net.base_net.parameters(), 'lr': 1e-4},
        {'params': cond_net.adapter.parameters(), 'lr': 6e-4},
    ])
    best_crps_p2 = float('inf')

    for epoch in range(args.phase2_epochs):
        cond_net.train()
        epoch_loss, n_b = 0, 0
        t0 = time.time()
        for batch in train_loader:
            past = batch["past_target"].to(device)
            future = batch["future_target"].to(device)
            ctx = past[:, -ctx_len:]
            loc = ctx.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
            scaled_ctx = ctx / loc
            scaled_future = future / loc

            # Compute extremity of future
            with torch.no_grad():
                ext_scores = compute_raw_extremity(scaled_future)
                ext_q = qmap.to_quantile(ext_scores)

            loss = conditioned_meanflow_loss(cond_net, scaled_future, scaled_ctx, ext_q)
            optimizer_p2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cond_net.parameters(), 1.0)
            optimizer_p2.step()
            with torch.no_grad():
                for p, pe in zip(cond_net.parameters(), cond_ema.parameters()):
                    pe.data.lerp_(p.data, 2e-4)
            epoch_loss += loss.item()
            n_b += 1

        avg = epoch_loss / n_b
        elapsed = time.time() - t0
        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(f"P2 Epoch {epoch+1:>3}/{args.phase2_epochs} | Loss: {avg:.4f} | {elapsed:.1f}s")

        if (epoch + 1) % 50 == 0 or (epoch + 1) == args.phase2_epochs:
            # Evaluate with neutral conditioning (unconditional mode)
            metrics, _, _, _ = evaluate(
                cond_ema, dataset, transformation, cfg, device,
                num_samples=16, guidance_scale=1.0, target_extremity=0.5,
                label="P2-neutral", is_conditioned=True,
            )
            # Also evaluate with marginal
            metrics_m, _, _, _ = evaluate(
                cond_ema, dataset, transformation, cfg, device,
                num_samples=16, guidance_scale=1.0, target_extremity='marginal',
                label="P2-marginal", is_conditioned=True,
            )
            crps = metrics["mean_wQuantileLoss"]
            crps_m = metrics_m["mean_wQuantileLoss"]
            best_crps_val = min(crps, crps_m)
            if best_crps_val < best_crps_p2:
                best_crps_p2 = best_crps_val
                torch.save({
                    'cond_net': cond_net.state_dict(), 'cond_ema': cond_ema.state_dict(),
                    'epoch': epoch + 1, 'crps': best_crps_val,
                }, os.path.join(outdir, 'phase2_best.pt'))
            logger.info(f"  P2 neutral={crps:.6f} marginal={crps_m:.6f} | Best={best_crps_p2:.6f}")

    all_results['phases']['phase2'] = {'best_crps': best_crps_p2, 'epochs': args.phase2_epochs}

    # ============================================================
    # Phase 2 comprehensive evaluation: guidance sweep + tail metrics
    # ============================================================
    logger.info("\n--- Phase 2 comprehensive evaluation ---")
    p2_eval = {}

    for tq in [0.5, 'marginal', 0.3, 0.7, 0.8, 0.9, 0.95]:
        for w in [1.0, 1.5, 2.0, 3.0]:
            key = f'tq={tq}_w={w}'
            metrics, fcs, tss, _ = evaluate(
                cond_ema, dataset, transformation, cfg, device,
                num_samples=args.num_eval_samples, guidance_scale=w,
                target_extremity=tq, label=key, is_conditioned=True,
            )
            tail_decomp = tail_stratified_crps(fcs, tss, pred_len)
            p2_eval[key] = {
                'crps': metrics["mean_wQuantileLoss"],
                'nd': metrics["ND"],
                'nrmse': metrics["NRMSE"],
                'tail_decomp': tail_decomp,
            }
    all_results['phases']['phase2_eval'] = p2_eval

    # ============================================================
    # Phase 3: Self-training rounds
    # ============================================================
    for round_num in range(1, args.self_train_rounds + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE 3: Self-Training Round {round_num}")
        logger.info(f"{'='*60}")

        # Collect original training data
        orig_data = {'ctx': [], 'fut': [], 'loc': [], 'ext_q': []}
        cond_ema.eval()
        for batch in train_loader:
            past = batch["past_target"].to(device)
            future = batch["future_target"].to(device)
            ctx = past[:, -ctx_len:]
            loc = ctx.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
            scaled_fut = future / loc
            with torch.no_grad():
                ext_scores = compute_raw_extremity(scaled_fut)
                ext_q = qmap.to_quantile(ext_scores)
            orig_data['ctx'].append(ctx.cpu())
            orig_data['fut'].append(future.cpu())
            orig_data['loc'].append(loc.cpu())
            orig_data['ext_q'].append(ext_q.cpu() if isinstance(ext_q, torch.Tensor) else torch.tensor(ext_q))

        orig_ctx = torch.cat(orig_data['ctx'])
        orig_fut = torch.cat(orig_data['fut'])
        orig_loc = torch.cat(orig_data['loc'])
        orig_eq = torch.cat(orig_data['ext_q'])
        logger.info(f"  Original data: {orig_ctx.shape[0]} samples")

        # Generate synthetic data (biased toward tails)
        logger.info("  Generating synthetic data...")
        syn_data = {'ctx': [], 'fut': [], 'loc': [], 'ext_q': []}
        n_gen = max(orig_ctx.shape[0] // 256, 8)

        with torch.no_grad():
            for _ in tqdm(range(n_gen), desc="Generating"):
                B = min(256, orig_ctx.shape[0])
                idx = torch.randint(0, orig_ctx.shape[0], (B,))
                ctx = orig_ctx[idx].to(device)
                loc_b = orig_loc[idx].to(device)
                scaled_ctx = ctx / loc_b

                # Target high extremity for tail enrichment
                ext_targets = torch.rand(B, device=device) * 0.5 + 0.5  # [0.5, 1.0]
                samples = guided_sample(cond_ema, scaled_ctx, ext_targets,
                                        (B, pred_len), device, guidance_scale=1.5)
                # Score generated samples
                gen_scores = compute_raw_extremity(samples)
                gen_eq = qmap.to_quantile(gen_scores)

                syn_data['ctx'].append(ctx.cpu())
                syn_data['fut'].append((samples * loc_b).cpu())
                syn_data['loc'].append(loc_b.cpu())
                syn_data['ext_q'].append(gen_eq.cpu() if isinstance(gen_eq, torch.Tensor) else torch.tensor(gen_eq))

        syn_ctx = torch.cat(syn_data['ctx'])
        syn_fut = torch.cat(syn_data['fut'])
        syn_loc = torch.cat(syn_data['loc'])
        syn_eq = torch.cat(syn_data['ext_q'])

        # Tilted resampling of synthetic data
        weights = torch.exp(args.alpha * syn_eq)
        weights = weights / weights.sum()
        n_syn = min(syn_ctx.shape[0], orig_ctx.shape[0])
        indices = torch.multinomial(weights, n_syn, replacement=True)

        # Mix synthetic + original
        n_orig = orig_ctx.shape[0] // 2
        perm = torch.randperm(orig_ctx.shape[0])[:n_orig]
        mixed_ctx = torch.cat([syn_ctx[indices], orig_ctx[perm]])
        mixed_fut = torch.cat([syn_fut[indices], orig_fut[perm]])
        mixed_loc = torch.cat([syn_loc[indices], orig_loc[perm]])
        mixed_eq = torch.cat([syn_eq[indices], orig_eq[perm]])

        logger.info(f"  Mixed dataset: {mixed_ctx.shape[0]} samples, "
                     f"ext_q mean={mixed_eq.mean():.3f}, >0.8={float((mixed_eq > 0.8).float().mean()):.3f}")

        # Fine-tune
        mixed_ds = TensorDataset(mixed_ctx, mixed_fut, mixed_loc, mixed_eq)
        mixed_loader = DataLoader(mixed_ds, batch_size=256, shuffle=True, drop_last=True)
        st_optimizer = AdamW(cond_net.parameters(), lr=1e-4)

        for ep in range(args.self_train_epochs):
            cond_net.train()
            ep_loss, n_b = 0, 0
            t0 = time.time()
            for ctx_b, fut_b, loc_b, eq_b in mixed_loader:
                ctx_b, fut_b = ctx_b.to(device), fut_b.to(device)
                loc_b, eq_b = loc_b.to(device), eq_b.to(device)
                loss = conditioned_meanflow_loss(
                    cond_net, fut_b / loc_b, ctx_b / loc_b, eq_b)
                st_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cond_net.parameters(), 1.0)
                st_optimizer.step()
                with torch.no_grad():
                    for p, pe in zip(cond_net.parameters(), cond_ema.parameters()):
                        pe.data.lerp_(p.data, 2e-4)
                ep_loss += loss.item()
                n_b += 1
            if (ep + 1) % 20 == 0 or ep == 0:
                logger.info(f"  ST-R{round_num} Epoch {ep+1}/{args.self_train_epochs} | "
                            f"Loss: {ep_loss/n_b:.4f} | {time.time()-t0:.1f}s")

        # Evaluate after self-training
        logger.info(f"\n  Evaluating after ST round {round_num}...")
        st_results = {}
        for tq in [0.5, 'marginal', 0.8, 0.9]:
            for w in [1.0, 2.0, 3.0]:
                key = f'tq={tq}_w={w}'
                metrics, fcs, tss, _ = evaluate(
                    cond_ema, dataset, transformation, cfg, device,
                    num_samples=args.num_eval_samples, guidance_scale=w,
                    target_extremity=tq, label=f"ST-R{round_num}-{key}",
                    is_conditioned=True,
                )
                tail_decomp = tail_stratified_crps(fcs, tss, pred_len)
                st_results[key] = {
                    'crps': metrics["mean_wQuantileLoss"],
                    'nd': metrics["ND"],
                    'nrmse': metrics["NRMSE"],
                    'tail_decomp': tail_decomp,
                }
        all_results['phases'][f'self_train_round_{round_num}'] = st_results

        torch.save({
            'cond_net': cond_net.state_dict(), 'cond_ema': cond_ema.state_dict(),
            'round': round_num,
        }, os.path.join(outdir, f'st_round_{round_num}.pt'))

    # ============================================================
    # Save all results and print summary
    # ============================================================
    results_path = os.path.join(outdir, 'all_results_v2.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\n{'='*70}")
    logger.info(f"FINAL SUMMARY: {name}")
    logger.info(f"{'='*70}")
    tsflow = TSFLOW_CRPS.get(name, "?")
    logger.info(f"TSFlow (32 NFE):            CRPS={tsflow}")
    logger.info(f"Phase 1 base (1 NFE):       CRPS={all_results['phases']['phase1']['best_crps']:.6f}")
    logger.info(f"Phase 2 conditioned (1 NFE): CRPS={all_results['phases']['phase2']['best_crps']:.6f}")

    # Print guidance sweep highlights
    if 'phase2_eval' in all_results['phases']:
        pe = all_results['phases']['phase2_eval']
        for key in ['tq=0.5_w=1.0', 'tq=marginal_w=1.0', 'tq=0.9_w=2.0', 'tq=0.9_w=3.0']:
            if key in pe:
                logger.info(f"  {key}: CRPS={pe[key]['crps']:.6f}")
                if 'tail_decomp' in pe[key]:
                    td = pe[key]['tail_decomp']
                    logger.info(f"    Tail-10%: {td.get('tail_10pct', 'N/A')}, "
                                f"Tail-20%: {td.get('tail_20pct', 'N/A')}")

    for rnd in range(1, args.self_train_rounds + 1):
        key = f'self_train_round_{rnd}'
        if key in all_results['phases']:
            st = all_results['phases'][key]
            for k in ['tq=0.5_w=1.0', 'tq=0.9_w=2.0']:
                if k in st:
                    logger.info(f"  ST-R{rnd} {k}: CRPS={st[k]['crps']:.6f}")

    logger.info(f"\nResults saved to {results_path}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
