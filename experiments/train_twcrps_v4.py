"""
Wessel-style tail-weighted fine-tune of a pretrained v4 base.

Optimize a sample-estimator for the upper-tail twCRPS of Gneiting & Ranjan
(2011) / Allen et al. (JASA 2025):

    twCRPS(F, y; t) = E|v(X) - v(y)| - (1/2) * E|v(X) - v(X')|   with v(x) = max(x, t)

Differentiable in X (samples) through max(.) (subgradient is well-defined).
Per-window threshold is `t_i = QuantileMapper.inverse(tau)` where the
mapper is fit on training futures' absolute values at the *per-window*
scale. To avoid mixing units we compute twCRPS in the RobustNorm space of
the ground-truth future: both `pred` and `target` are divided by the
per-window mean-abs before the twCRPS is computed, and the threshold t is
a fraction of that same scale.

We regularize with a small amount of the base MeanFlow JVP loss to keep
calibration from collapsing (Wessel et al. observe that pure tail-weighted
training degrades body CRPS — this is the `body_weight` knob).

We hold τ (the threshold quantile) fixed within a run. To sweep τ, launch
multiple runs with different `--tau` values. The final Pareto plot in
§2.5 of RESULTS.md is built from the set of runs.

Usage:
  python experiments/train_twcrps_v4.py solar_nips --tau 0.9 --epochs 200
"""
from __future__ import annotations
import os, sys, time, argparse, logging
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from copy import deepcopy

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.loader import TrainDataLoader
from gluonts.itertools import Cached
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify
from gluonts.transform import (
    AddObservedValuesIndicator, AddTimeFeatures, AsNumpyArray,
    Chain, ExpectedNumInstanceSampler, InstanceSplitter,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v4 import (
    S4DMeanFlowNetV4, get_lag_indices_v4, extract_lags_v4,
    RobustNorm, sample_t_r,
)
from experiments.train_v4 import CONFIGS, LAG_MAP

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def energy_tw_crps(pred_samples, target, threshold, side="upper"):
    """
    Sample-estimator twCRPS with chaining function v(x) = max(x, t) (upper).
    Assumes `pred_samples` and `target` are already in a common scale (e.g.
    RobustNorm-normalized), and `threshold` is given in that same scale.

    pred_samples: (B, N, T)
    target:       (B, T)
    threshold:    scalar or (B,) or (B, T)
    Returns:      scalar loss.
    """
    if torch.is_tensor(threshold) and threshold.ndim == 1:
        t_samples = threshold.view(-1, 1, 1)
        t_target = threshold.view(-1, 1)
    elif torch.is_tensor(threshold) and threshold.ndim == 2:
        t_samples = threshold.unsqueeze(1)
        t_target = threshold
    else:
        scalar = torch.as_tensor(threshold, dtype=pred_samples.dtype,
                                  device=pred_samples.device)
        t_samples = scalar
        t_target = scalar

    if side == "upper":
        vx = torch.maximum(pred_samples, t_samples)          # (B, N, T)
        vy = torch.maximum(target, t_target)                 # (B, T)
    elif side == "lower":
        vx = torch.minimum(pred_samples, t_samples)
        vy = torch.minimum(target, t_target)
    else:
        raise ValueError(side)

    # Term 1: E|v(X) - v(y)|
    term1 = (vx - vy.unsqueeze(1)).abs().mean(dim=1)         # (B, T)
    # Term 2: (1/2) E|v(X) - v(X')|
    s1 = vx.unsqueeze(2)
    s2 = vx.unsqueeze(1)
    term2 = 0.5 * (s1 - s2).abs().mean(dim=(1, 2))           # (B, T)
    # Normalize by |target| magnitude so the loss scale is comparable across
    # datasets (same convention as GluonTS mean_wQuantileLoss).
    denom = target.abs().mean(dim=-1, keepdim=True).clamp(min=1e-6)
    return ((term1 - term2) / denom).mean()


def twcrps_finetune_loss(net, future_clean, context_channels,
                          threshold, n_samples=16, tail_weight=0.7,
                          body_weight=0.3, noise_scale=1.0):
    """Weighted sum of twCRPS surrogate + MeanFlow JVP self-consistency."""
    B = future_clean.shape[0]
    device = future_clean.device

    # Differentiable one-step sampling
    samples = []
    for _ in range(n_samples):
        z = torch.randn_like(future_clean) * noise_scale
        t = torch.ones(B, device=device); h = t.clone()
        u = net(z, (t, h), context_channels)
        samples.append((z - u).unsqueeze(1))
    samples = torch.cat(samples, dim=1)  # (B, N, T)
    tail = energy_tw_crps(samples, future_clean, threshold, side="upper")

    # MeanFlow JVP (stability regularizer)
    e = torch.randn_like(future_clean)
    t_scalar, r = sample_t_r(B, device)
    t_bc, r_bc = t_scalar.unsqueeze(-1), r.unsqueeze(-1)
    z = (1 - t_bc) * future_clean + t_bc * e
    v = e - future_clean

    def u_func(z, t_bc, r_bc):
        h_bc = t_bc - r_bc
        return net(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)), context_channels)

    with torch.amp.autocast("cuda", enabled=False):
        u_pred, dudt = torch.func.jvp(
            u_func, (z, t_bc, r_bc),
            (v, torch.ones_like(t_bc), torch.zeros_like(r_bc)),
        )
        u_tgt = (v - (t_bc - r_bc) * dudt).detach()
        mf_l = ((u_pred - u_tgt) ** 2).sum(dim=1)
        mf_l = (mf_l / ((mf_l.detach() + 1e-3) ** 0.75)).mean()

    return tail_weight * tail + body_weight * mf_l, tail.item(), mf_l.item()


def compute_threshold_from_tau(dataset_train, ctx_len, pred_len, tau):
    """
    Return the τ-quantile of `future.max() / |future|.mean()` over training
    windows — same scale as RobustNorm-normalized futures used inside the
    training loop. This is a scalar threshold applied to normalized preds
    & targets.
    """
    vals = []
    for entry in dataset_train:
        ts = np.array(entry["target"], dtype=np.float32)
        stride = max(pred_len // 2, 1)
        for start in range(0, len(ts) - ctx_len - pred_len + 1, stride):
            ctx = ts[start:start + ctx_len]
            fut = ts[start + ctx_len:start + ctx_len + pred_len]
            loc = max(np.abs(ctx).mean(), 1e-6)
            vals.append(float(fut.max() / loc))
    vals = np.asarray(vals, dtype=np.float32)
    thr = float(np.quantile(vals, tau))
    logger.info(
        f"Threshold τ={tau}: {thr:.3f} in RobustNorm space "
        f"(mean={vals.mean():.3f} q90={np.quantile(vals,0.9):.3f} "
        f"q95={np.quantile(vals,0.95):.3f} q99={np.quantile(vals,0.99):.3f})"
    )
    return thr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--tau", type=float, default=0.9,
                   help="Threshold quantile: t = Q_tau of future/loc in training")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-5,
                   help="LR for fine-tuning (small by default)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--tail-weight", type=float, default=0.7)
    p.add_argument("--body-weight", type=float, default=0.3)
    p.add_argument("--num-batches", type=int, default=64)
    p.add_argument("--noise-scale", type=float, default=1.0)
    args = p.parse_args()

    name = args.dataset
    cfg = CONFIGS[name]
    freq, ctx_len, pred_len = cfg["freq"], cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP[freq]
    n_lags = len(get_lag_indices_v4(freq))
    device = torch.device("cuda")
    torch.manual_seed(42); np.random.seed(42)

    outdir = os.path.join(os.path.dirname(__file__), "..", "results_twcrps", name)
    os.makedirs(outdir, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"twCRPS fine-tune: {name}  tau={args.tau}  epochs={args.epochs}")
    logger.info(f"  tail_weight={args.tail_weight} body_weight={args.body_weight}")
    logger.info(f"  lr={args.lr}  n_samples={args.n_samples}  batch_size={args.batch_size}")
    logger.info(f"{'='*60}")

    dataset = get_dataset(name)
    threshold = compute_threshold_from_tau(dataset.train, ctx_len, pred_len, args.tau)

    transformation = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(start_field="start", target_field="target", output_field="time_feat",
                        time_features=time_features_from_frequency_str(freq), pred_length=pred_len),
    ])
    splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start",
        instance_sampler=ExpectedNumInstanceSampler(num_instances=1, min_future=pred_len),
        past_length=ctx_len + max_lag, future_length=pred_len,
        time_series_fields=["time_feat", "observed_values"],
    )
    transformed = transformation.apply(dataset.train, is_train=True)
    loader = TrainDataLoader(
        Cached(transformed), batch_size=args.batch_size, stack_fn=batchify,
        transform=splitter, num_batches_per_epoch=args.num_batches,
        shuffle_buffer_length=2000,
    )

    net = S4DMeanFlowNetV4(
        pred_len=pred_len, ctx_len=ctx_len, d_model=192, n_s4d_blocks=6,
        ssm_dim=64, n_lags=n_lags, freq=freq,
    ).to(device)
    v4p = os.path.join(os.path.dirname(__file__), "..", "results_v4", name, "best.pt")
    ck = torch.load(v4p, map_location=device, weights_only=False)
    net.load_state_dict(ck["net_ema"])
    net_ema = deepcopy(net).eval()
    logger.info(f"Loaded v4 base (CRPS={ck.get('crps','?')})")

    opt = AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    norm = RobustNorm()

    best = float("inf")
    for epoch in range(args.epochs):
        net.train()
        sum_loss = sum_tail = sum_mf = 0.0
        nb = 0
        t0 = time.time()
        for batch in loader:
            past = batch["past_target"].to(device).float()
            fut = batch["future_target"].to(device).float()
            ctx = past[:, -ctx_len:]
            ctx_n, loc, scale = norm(ctx)
            fut_n = (fut - loc) / scale
            lags = extract_lags_v4(past, ctx_len, freq)
            lags_n = (lags - loc.unsqueeze(1)) / scale.unsqueeze(1)

            loss, tail_item, mf_item = twcrps_finetune_loss(
                net, fut_n, lags_n, threshold,
                n_samples=args.n_samples,
                tail_weight=args.tail_weight, body_weight=args.body_weight,
                noise_scale=args.noise_scale,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            with torch.no_grad():
                for p_, pe in zip(net.parameters(), net_ema.parameters()):
                    pe.data.lerp_(p_.data, 1e-4)
            sum_loss += loss.item(); sum_tail += tail_item; sum_mf += mf_item
            nb += 1
        sched.step()

        avg = sum_loss / max(nb, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Ep {epoch+1}/{args.epochs}  total={avg:.4f}  "
                f"tail={sum_tail/max(nb,1):.4f}  mf={sum_mf/max(nb,1):.4f}  "
                f"lr={opt.param_groups[0]['lr']:.2e}  {time.time()-t0:.1f}s"
            )
        if avg < best:
            best = avg
            torch.save({
                "net": net.state_dict(),
                "net_ema": net_ema.state_dict(),
                "epoch": epoch + 1,
                "tau": args.tau,
                "threshold": threshold,
                "tail_weight": args.tail_weight,
                "body_weight": args.body_weight,
                "loss": avg,
            }, os.path.join(outdir, f"twcrps_tau{args.tau}_best.pt"))


if __name__ == "__main__":
    main()
