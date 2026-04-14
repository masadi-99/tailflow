"""
Extremity-Conditioned MeanFlow for Tail-Aware Time Series Generation.

Key additions over base MeanFlow-TS:
1. Extremity functionals: volatility, max deviation, drawdown
2. Quantile mapping to [0,1] conditioning variable
3. Classifier-free guidance (CFG) style dropout on extremity condition
4. Guided sampling at inference for controllable tail generation
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .model import SinusoidalPosEmb, ResBlock1D, sample_t_r


# ============================================================
# Extremity Functionals
# ============================================================

def compute_volatility(x):
    """Rolling volatility (std of first differences). x: (B, T) or (N, T)."""
    diffs = x[:, 1:] - x[:, :-1]
    return diffs.std(dim=1)


def compute_max_deviation(x):
    """Max absolute deviation from mean. x: (B, T)."""
    mean = x.mean(dim=1, keepdim=True)
    return (x - mean).abs().max(dim=1).values


def compute_drawdown(x):
    """Maximum drawdown (max peak-to-trough). x: (B, T)."""
    cummax = x.cummax(dim=1).values
    drawdown = cummax - x
    return drawdown.max(dim=1).values


def compute_range(x):
    """Range (max - min). x: (B, T)."""
    return x.max(dim=1).values - x.min(dim=1).values


def compute_tail_ratio(x):
    """Ratio of values beyond 1.5 IQR from median. x: (B, T)."""
    median = x.median(dim=1, keepdim=True).values
    q25 = x.quantile(0.25, dim=1, keepdim=True)
    q75 = x.quantile(0.75, dim=1, keepdim=True)
    iqr = (q75 - q25).clamp(min=1e-6)
    outlier_mask = ((x - median).abs() > 1.5 * iqr).float()
    return outlier_mask.mean(dim=1)


def compute_raw_extremity(x):
    """
    Legacy composite extremity score (NOT rank-normalized).
    Averages normalized versions of volatility, max_deviation, drawdown, range.
    x: (B, T) — should be scale-normalized already.
    Returns: (B,) raw scores suitable for quantile mapping.

    NOTE: superseded by `compute_peak_exceedance` (below), which grounds the
    conditioning variable directly in the tail-calibration framework of
    Allen, Ziegel & Ginsbourger (JASA 2025). Kept for backward compatibility.
    """
    vol = compute_volatility(x)
    maxdev = compute_max_deviation(x)
    dd = compute_drawdown(x)
    rng = compute_range(x)
    return (vol + maxdev + dd + rng) / 4.0


def compute_peak_exceedance(x):
    """
    Scale-invariant peak magnitude score for threshold-exceedance conditioning.

    s(x) = max(|x|) / mean(|x|)

    Interpretation. After fitting a `QuantileMapper` on `s(x)` over training
    windows, the mapped value `q ∈ [0, 1]` is the empirical CDF of the
    window's peak-to-mean ratio under the training distribution. Thus

        q(x) = 0.95  ⇔  window's peak is at the 95th percentile
                         of training windows' peaks (in mean-abs units).

    This connects directly to Allen, Ziegel & Ginsbourger (JASA 2025)'s
    tail-calibration framework, which scores probabilistic forecasts via
    excess-over-threshold distributions. We condition the generative model
    on the *target quantile* of the peak, so that guidance at `tq = 0.95`
    literally asks the model to generate windows whose peak would be at
    the 95th percentile of training peaks — and higher guidance w pushes
    past even that threshold.

    x: (B, T) — should be in raw (non-normalized) units so that the ratio
    max|x| / mean|x| is dimensionless.
    Returns: (B,) scores.
    """
    peak = x.abs().max(dim=1).values
    mean_abs = x.abs().mean(dim=1).clamp(min=1e-6)
    return peak / mean_abs


def compute_composite_extremity(x):
    """
    Composite extremity score combining multiple functionals.
    Each is rank-normalized to [0,1] then averaged.
    x: (B, T) — should be normalized (zero-mean, unit-scale preferred).
    Returns: (B,) scores.
    NOTE: This uses within-batch ranking, only suitable for large batches or evaluation.
    For training, use compute_raw_extremity + QuantileMapper instead.
    """
    scores = []
    for fn in [compute_volatility, compute_max_deviation, compute_drawdown, compute_range]:
        s = fn(x)
        # Rank-normalize within batch
        ranks = s.argsort().argsort().float()
        ranks = ranks / max(ranks.shape[0] - 1, 1)
        scores.append(ranks)
    return torch.stack(scores, dim=0).mean(dim=0)


# ============================================================
# Quantile Mapper — fits on training data, maps scores to [0,1]
# ============================================================

class QuantileMapper:
    """Maps raw extremity scores to empirical quantiles in [0,1]."""

    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.thresholds = None  # (n_bins-1,) quantile boundaries
        self.scores_sorted = None

    def fit(self, scores):
        """scores: 1D numpy array of extremity scores from training set."""
        scores = np.sort(scores)
        self.scores_sorted = scores
        # Compute bin thresholds
        quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.thresholds = np.quantile(scores, quantiles)

    def to_quantile(self, scores):
        """Map scores to continuous quantile in [0,1]. scores: numpy or torch."""
        is_torch = isinstance(scores, torch.Tensor)
        if is_torch:
            device = scores.device
            scores_np = scores.detach().cpu().numpy()
        else:
            scores_np = scores
        # Empirical CDF
        q = np.searchsorted(self.scores_sorted, scores_np) / len(self.scores_sorted)
        q = np.clip(q, 0.0, 1.0)
        if is_torch:
            return torch.tensor(q, dtype=torch.float32, device=device)
        return q.astype(np.float32)

    def to_bin(self, scores):
        """Map scores to ordinal bin in {0, ..., n_bins-1}. scores: numpy or torch."""
        is_torch = isinstance(scores, torch.Tensor)
        if is_torch:
            device = scores.device
            scores_np = scores.detach().cpu().numpy()
        else:
            scores_np = scores
        bins = np.digitize(scores_np, self.thresholds)
        if is_torch:
            return torch.tensor(bins, dtype=torch.long, device=device)
        return bins


# ============================================================
# Extremity-Conditioned MeanFlow Network
# ============================================================

class ExtremityCondMeanFlowNet(nn.Module):
    """
    MeanFlow network conditioned on both context and extremity level.

    The extremity condition c in [0,1] (quantile) is embedded and added
    to the time embedding. During training, c is dropped (set to 0.5 = "neutral")
    with probability cfg_drop_prob to enable classifier-free guidance.
    """

    def __init__(self, pred_len=24, ctx_len=24, model_channels=128,
                 num_res_blocks=4, time_emb_dim=64, dropout=0.1,
                 cfg_drop_prob=0.15):
        super().__init__()
        self.pred_len = pred_len
        self.ctx_len = ctx_len
        self.cfg_drop_prob = cfg_drop_prob

        # Dual time embedding
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        emb_dim = time_emb_dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim * 2, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim),
        )

        # Extremity conditioning embedding
        # Continuous quantile -> embedding via small MLP
        # Zero-initialize output layer so conditioning starts as no-op
        ext_out = nn.Linear(emb_dim, emb_dim)
        nn.init.zeros_(ext_out.weight)
        nn.init.zeros_(ext_out.bias)
        self.ext_mlp = nn.Sequential(
            nn.Linear(1, emb_dim), nn.SiLU(), ext_out,
        )

        # Context encoder
        self.ctx_proj = nn.Conv1d(1, model_channels, 1)
        self.ctx_blocks = nn.ModuleList([
            ResBlock1D(model_channels, emb_dim, dropout) for _ in range(2)
        ])
        self.ctx_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(model_channels, emb_dim),
        )
        self.ctx_to_pred = nn.Linear(ctx_len, pred_len)
        self.ctx_feat_proj = nn.Conv1d(model_channels, model_channels, 1)

        # Prediction pathway
        self.pred_proj = nn.Conv1d(1, model_channels, 1)
        self.pred_blocks = nn.ModuleList([
            ResBlock1D(model_channels, emb_dim, dropout) for _ in range(num_res_blocks)
        ])

        # Output (zero-init)
        self.out_norm = nn.GroupNorm(min(8, model_channels), model_channels)
        self.out_proj = nn.Conv1d(model_channels, 1, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, noisy_pred, time_steps, context, extremity_q):
        """
        noisy_pred: (B, pred_len)
        time_steps: (t, h) each (B,)
        context: (B, ctx_len)
        extremity_q: (B,) in [0,1] — extremity quantile
        """
        t, h = time_steps
        emb = torch.cat([self.time_emb(t), self.time_emb(h)], dim=-1)
        emb = self.time_mlp(emb)

        # Add extremity embedding
        ext_emb = self.ext_mlp(extremity_q.unsqueeze(-1))
        emb = emb + ext_emb

        # Encode context
        ctx = self.ctx_proj(context.unsqueeze(1))
        for block in self.ctx_blocks:
            ctx = block(ctx, emb)
        emb = emb + self.ctx_pool(ctx)
        ctx_spatial = self.ctx_feat_proj(self.ctx_to_pred(ctx))

        # Process prediction
        pred = self.pred_proj(noisy_pred.unsqueeze(1)) + ctx_spatial
        for block in self.pred_blocks:
            pred = block(pred, emb)

        return self.out_proj(F.silu(self.out_norm(pred))).squeeze(1)


# ============================================================
# Loss with CFG dropout
# ============================================================

def extremity_cond_meanflow_loss(net, future_clean, context, extremity_q,
                                  norm_p=0.75, norm_eps=1e-3):
    """
    MeanFlow JVP loss with extremity conditioning and CFG dropout.

    During training, with probability cfg_drop_prob, the extremity condition
    is replaced with 0.5 (the unconditional/neutral value).
    """
    B = future_clean.shape[0]
    device = future_clean.device

    # CFG dropout: replace extremity_q with 0.5 (neutral)
    if net.training and net.cfg_drop_prob > 0:
        drop_mask = torch.rand(B, device=device) < net.cfg_drop_prob
        extremity_q = torch.where(drop_mask, torch.full_like(extremity_q, 0.5), extremity_q)

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
# Guided Sampling
# ============================================================

@torch.no_grad()
def guided_sample(net, context, extremity_q, shape, device, guidance_scale=1.0):
    """
    One-step MeanFlow generation with classifier-free guidance.

    u_guided = u_uncond + w * (u_cond - u_uncond)

    where w = guidance_scale. w=1 is standard conditional, w>1 amplifies
    the effect of extremity conditioning (pushes toward more extreme samples),
    w=0 is unconditional.

    Args:
        net: ExtremityCondMeanFlowNet
        context: (B, ctx_len) — scaled context
        extremity_q: (B,) in [0,1] — target extremity quantile
        shape: (B, pred_len)
        device: torch device
        guidance_scale: float, guidance weight (1.0 = standard, >1 = amplified)
    Returns:
        samples: (B, pred_len)
    """
    B = shape[0]
    z_1 = torch.randn(shape, device=device)
    t = torch.ones(B, device=device)
    h = torch.ones(B, device=device)

    if guidance_scale == 1.0:
        u = net(z_1, (t, h), context, extremity_q)
        return z_1 - u

    # Conditional prediction
    u_cond = net(z_1, (t, h), context, extremity_q)
    # Unconditional prediction (neutral extremity = 0.5)
    u_uncond = net(z_1, (t, h), context, torch.full_like(extremity_q, 0.5))
    # Guided output
    u_guided = u_uncond + guidance_scale * (u_cond - u_uncond)
    return z_1 - u_guided


# ============================================================
# Forecaster for evaluation
# ============================================================

class ExtremityCondForecaster(nn.Module):
    """Wraps ExtremityCondMeanFlowNet for GluonTS evaluation pipeline.

    Modes:
    - target_extremity=float: use fixed extremity for all samples
    - target_extremity='marginal': each sample draws random ext_q ~ Uniform(0,1)
    - target_extremity=None: use 0.5 (neutral/unconditional)
    """

    def __init__(self, net, context_length, prediction_length, num_samples=100,
                 guidance_scale=1.0, target_extremity=None, quantile_mapper=None,
                 extremity_fn=None):
        super().__init__()
        self.net = net
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.guidance_scale = guidance_scale
        self.target_extremity = target_extremity
        self.quantile_mapper = quantile_mapper
        self.extremity_fn = extremity_fn or compute_raw_extremity

    def forward(self, past_target, past_observed_values, **kwargs):
        device = past_target.device
        B = past_target.shape[0]
        context = past_target[:, -self.context_length:]
        loc = context.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
        scaled_ctx = context / loc

        all_preds = []
        for _ in range(self.num_samples):
            # Determine extremity conditioning per sample
            if self.target_extremity == 'marginal':
                # Marginal: each sample gets a random extremity
                ext_q = torch.rand(B, device=device)
            elif isinstance(self.target_extremity, (int, float)):
                ext_q = torch.full((B,), float(self.target_extremity), device=device)
            else:
                # Default: neutral (unconditional mode)
                ext_q = torch.full((B,), 0.5, device=device)

            sample = guided_sample(
                self.net, scaled_ctx, ext_q,
                (B, self.prediction_length), device,
                guidance_scale=self.guidance_scale,
            )
            all_preds.append(sample * loc)

        return torch.stack(all_preds, dim=1)


# ============================================================
# Self-Training Utilities
# ============================================================

def generate_synthetic_samples(net, dataloader, quantile_mapper, device,
                               n_samples_per_batch=8, guidance_scale=1.0,
                               extremity_fn=None):
    """
    Generate synthetic samples from the model for self-training.

    Returns list of dicts with: context, future, extremity_score, extremity_q
    """
    extremity_fn = extremity_fn or compute_raw_extremity
    net.eval()
    synthetic = []

    with torch.no_grad():
        for batch in dataloader:
            past = batch["past_target"].to(device)
            B = past.shape[0]
            ctx_len = net.ctx_len if hasattr(net, 'ctx_len') else net.module.ctx_len
            pred_len = net.pred_len if hasattr(net, 'pred_len') else net.module.pred_len
            ctx = past[:, -ctx_len:]
            loc = ctx.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
            scaled_ctx = ctx / loc

            for _ in range(n_samples_per_batch):
                # Sample with varied extremity targets
                ext_targets = torch.rand(B, device=device)  # uniform over quantiles
                samples = guided_sample(
                    net, scaled_ctx, ext_targets,
                    (B, pred_len), device,
                    guidance_scale=guidance_scale,
                )
                # Score the generated samples
                scores = extremity_fn(samples)
                ext_q = quantile_mapper.to_quantile(scores)

                synthetic.append({
                    'context': ctx.cpu(),
                    'future': (samples * loc).cpu(),
                    'loc': loc.cpu(),
                    'extremity_q': ext_q.cpu() if isinstance(ext_q, torch.Tensor) else torch.tensor(ext_q),
                    'extremity_score': scores.cpu(),
                })

    return synthetic


def tilted_resampling(synthetic_data, alpha=2.0, original_fraction=0.5,
                      original_contexts=None, original_futures=None,
                      original_locs=None, original_ext_q=None):
    """
    Resample synthetic data with tilted distribution proportional to exp(alpha * q).

    Then mix with original_fraction of real data.

    Returns: (contexts, futures, locs, ext_qs) tensors.
    """
    # Concatenate all synthetic
    all_ctx = torch.cat([s['context'] for s in synthetic_data], dim=0)
    all_future = torch.cat([s['future'] for s in synthetic_data], dim=0)
    all_loc = torch.cat([s['loc'] for s in synthetic_data], dim=0)
    all_ext_q = torch.cat([s['extremity_q'] for s in synthetic_data], dim=0)

    N = all_ctx.shape[0]

    # Compute tilted weights
    weights = torch.exp(alpha * all_ext_q)
    weights = weights / weights.sum()

    # Resample according to tilted distribution
    n_synthetic = int(N * (1 - original_fraction))
    indices = torch.multinomial(weights, n_synthetic, replacement=True)

    resampled_ctx = all_ctx[indices]
    resampled_future = all_future[indices]
    resampled_loc = all_loc[indices]
    resampled_ext_q = all_ext_q[indices]

    # Mix with original data
    if original_contexts is not None:
        n_original = min(int(N * original_fraction), original_contexts.shape[0])
        perm = torch.randperm(original_contexts.shape[0])[:n_original]

        mixed_ctx = torch.cat([resampled_ctx, original_contexts[perm]], dim=0)
        mixed_future = torch.cat([resampled_future, original_futures[perm]], dim=0)
        mixed_loc = torch.cat([resampled_loc, original_locs[perm]], dim=0)
        mixed_ext_q = torch.cat([resampled_ext_q, original_ext_q[perm]], dim=0)
    else:
        mixed_ctx = resampled_ctx
        mixed_future = resampled_future
        mixed_loc = resampled_loc
        mixed_ext_q = resampled_ext_q

    return mixed_ctx, mixed_future, mixed_loc, mixed_ext_q
