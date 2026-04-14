"""
MeanFlow-TS v4: Maximum quality — matching TSFlow design choices.

Changes from v3:
1. GP prior (OU kernel) for noise initialization instead of N(0,I)
2. Robust normalization (handles zeros in solar data)
3. Bigger model (d_model=192, 6 S4D blocks)
4. More lag features (up to 28 days for hourly)
5. Conditional GP posterior for inference initialization
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .model_v3 import (
    S4DKernel, S4DBlock, SinusoidalPosEmb,
    sample_t_r, logit_normal_sample,
)


# ============================================================
# GP Prior: Ornstein-Uhlenbeck (OU) kernel
# ============================================================

class OUKernel:
    """
    Ornstein-Uhlenbeck kernel for GP prior.
    K(t, t') = sigma^2 * exp(-|t-t'| / length_scale)

    This provides correlated noise that respects temporal structure,
    giving the model a much better starting point than i.i.d. Gaussian.
    """
    def __init__(self, length_scale=1.0, sigma=1.0):
        self.length_scale = length_scale
        self.sigma = sigma

    def covariance_matrix(self, n, device):
        """Compute (n, n) covariance matrix."""
        t = torch.arange(n, device=device, dtype=torch.float32)
        diff = (t.unsqueeze(0) - t.unsqueeze(1)).abs()
        K = self.sigma**2 * torch.exp(-diff / self.length_scale)
        return K

    def sample(self, shape, device):
        """
        Sample from GP with OU kernel.
        shape: (B, T)
        Returns: (B, T) samples
        """
        B, T = shape
        K = self.covariance_matrix(T, device)
        # Add jitter for numerical stability
        K = K + 1e-5 * torch.eye(T, device=device)
        L = torch.linalg.cholesky(K)
        z = torch.randn(B, T, device=device)
        return z @ L.T  # (B, T)

    def conditional_sample(self, context, pred_len, device):
        """
        Sample from GP posterior conditioned on observed context.

        Given observed values y_obs at times t_obs, sample future values
        at times t_pred from p(y_pred | y_obs).

        context: (B, ctx_len) — observed values (normalized)
        Returns: (B, pred_len) — samples from conditional GP
        """
        B, ctx_len = context.shape
        total_len = ctx_len + pred_len

        K = self.covariance_matrix(total_len, device)
        K = K + 1e-5 * torch.eye(total_len, device=device)

        # Partition: K = [[K_oo, K_op], [K_po, K_pp]]
        K_oo = K[:ctx_len, :ctx_len]
        K_op = K[:ctx_len, ctx_len:]
        K_po = K[ctx_len:, :ctx_len]
        K_pp = K[ctx_len:, ctx_len:]

        # Conditional distribution: p(y_pred | y_obs) = N(mu_cond, K_cond)
        # mu_cond = K_po @ K_oo^{-1} @ y_obs
        # K_cond = K_pp - K_po @ K_oo^{-1} @ K_op
        K_oo_inv = torch.linalg.solve(K_oo, torch.eye(ctx_len, device=device))
        mu_cond = context @ (K_oo_inv @ K_op)  # (B, pred_len)
        K_cond = K_pp - K_po @ K_oo_inv @ K_op
        K_cond = K_cond + 1e-5 * torch.eye(pred_len, device=device)

        # Sample
        L_cond = torch.linalg.cholesky(K_cond)
        z = torch.randn(B, pred_len, device=device)
        return mu_cond + z @ L_cond.T


# ============================================================
# Robust Normalization (handles zeros)
# ============================================================

class RobustNorm(nn.Module):
    """
    Robust normalization that handles solar-style data with many zeros.
    Uses mean absolute value for scale (more robust than std for zero-heavy data).
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """
        x: (B, T).
        Returns: normalized x, loc (mean), scale.
        """
        loc = x.mean(dim=1, keepdim=True)
        scale = x.abs().mean(dim=1, keepdim=True).clamp(min=self.eps)
        return (x - loc) / scale, loc, scale

    def inverse(self, x, loc, scale):
        return x * scale + loc


# ============================================================
# Extended Lag Features (TSFlow-style)
# ============================================================

def get_lag_indices_v4(freq):
    """Extended lag offsets matching TSFlow."""
    if freq == "H":
        # Hourly: 1d, 2d, 3d, 4d, 5d, 6d, 1w, 2w, 3w, 4w
        return [24, 48, 72, 96, 120, 144, 168, 336, 504, 672]
    elif freq == "B":
        # Business daily: 1w, 2w, 3w, 4w, 5w, 6w
        return [5, 10, 15, 20, 25, 30]
    else:
        return [24, 48, 72, 168, 336]


def extract_lags_v4(past_target, ctx_len, freq="H"):
    """Extract lag features."""
    B = past_target.shape[0]
    context = past_target[:, -ctx_len:]
    lag_offsets = get_lag_indices_v4(freq)

    channels = [context.unsqueeze(1)]
    for offset in lag_offsets:
        start = past_target.shape[1] - ctx_len - offset
        end = past_target.shape[1] - offset
        if start >= 0:
            lag = past_target[:, start:end]
        else:
            lag = torch.zeros(B, ctx_len, device=past_target.device)
            if end > 0:
                available = past_target[:, :end]
                lag[:, -available.shape[1]:] = available
        channels.append(lag.unsqueeze(1))

    return torch.cat(channels, dim=1)


# ============================================================
# S4D-MeanFlow v4: Maximum Quality
# ============================================================

class S4DMeanFlowNetV4(nn.Module):
    """
    S4D MeanFlow with GP prior, robust norm, extended lags, bigger model.
    """

    def __init__(self, pred_len=24, ctx_len=72, d_model=192, n_s4d_blocks=6,
                 ssm_dim=64, time_emb_dim=64, dropout=0.1, n_lags=10, freq="H",
                 gp_length_scale=3.0):
        super().__init__()
        self.pred_len = pred_len
        self.ctx_len = ctx_len
        self.d_model = d_model
        self.freq = freq
        self.n_lags = n_lags
        n_input_channels = 1 + n_lags

        # GP prior
        self.gp = OUKernel(length_scale=gp_length_scale, sigma=1.0)

        # Time embedding
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        emb_dim = time_emb_dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim * 2, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim),
        )

        # Context encoder (3 blocks for deeper encoding)
        self.ctx_proj = nn.Linear(n_input_channels, d_model)
        self.ctx_blocks = nn.ModuleList([
            S4DBlock(d_model, N=ssm_dim, dropout=dropout, emb_dim=emb_dim)
            for _ in range(3)
        ])
        self.ctx_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(d_model, emb_dim),
        )

        # Context-to-prediction projection
        self.ctx_to_pred = nn.Linear(ctx_len, pred_len)
        self.ctx_feat_proj = nn.Linear(d_model, d_model)

        # Prediction decoder
        self.pred_proj = nn.Linear(1, d_model)
        self.pred_blocks = nn.ModuleList([
            S4DBlock(d_model, N=ssm_dim, dropout=dropout, emb_dim=emb_dim)
            for _ in range(n_s4d_blocks)
        ])

        # Output (zero-init)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, noisy_pred, time_steps, context_with_lags):
        t, h = time_steps
        emb = torch.cat([self.time_emb(t), self.time_emb(h)], dim=-1)
        emb = self.time_mlp(emb)

        ctx = self.ctx_proj(context_with_lags.transpose(1, 2))
        for block in self.ctx_blocks:
            ctx = block(ctx, emb)

        emb = emb + self.ctx_pool(ctx.transpose(1, 2))

        ctx_spatial = self.ctx_feat_proj(
            self.ctx_to_pred(ctx.transpose(1, 2)).transpose(1, 2)
        )

        pred = self.pred_proj(noisy_pred.unsqueeze(-1)) + ctx_spatial
        for block in self.pred_blocks:
            pred = block(pred, emb)

        return self.out_proj(F.silu(self.out_norm(pred))).squeeze(-1)


# ============================================================
# Loss with GP prior noise
# ============================================================

def v4_meanflow_loss(net, future_clean, context_with_lags, context_normed=None,
                     use_gp_noise=True, norm_p=0.75, norm_eps=1e-3):
    """
    MeanFlow JVP loss with optional GP prior noise.

    Instead of e ~ N(0, I), we use e ~ GP(0, K_OU) which gives
    temporally correlated noise. When context is provided, we use
    the conditional GP posterior for even better initialization.
    """
    B = future_clean.shape[0]
    device = future_clean.device

    if use_gp_noise and context_normed is not None:
        # Conditional GP noise: sample from p(e_pred | context)
        e = net.gp.conditional_sample(context_normed, future_clean.shape[1], device)
    elif use_gp_noise:
        # Unconditional GP noise
        e = net.gp.sample(future_clean.shape, device)
    else:
        e = torch.randn_like(future_clean)

    t, r = sample_t_r(B, device)
    t_bc, r_bc = t.unsqueeze(-1), r.unsqueeze(-1)

    z = (1 - t_bc) * future_clean + t_bc * e
    v = e - future_clean

    def u_func(z, t_bc, r_bc):
        h_bc = t_bc - r_bc
        return net(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)), context_with_lags)

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
# Forecaster with GP prior sampling
# ============================================================

class V4Forecaster(nn.Module):
    """GluonTS-compatible forecaster with GP prior."""

    def __init__(self, net, context_length, prediction_length, num_samples=100,
                 freq="H", use_gp=True):
        super().__init__()
        self.net = net
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.freq = freq
        self.use_gp = use_gp
        self.norm = RobustNorm()

    def forward(self, past_target, past_observed_values, **kwargs):
        device = past_target.device
        B = past_target.shape[0]
        context = past_target[:, -self.context_length:]

        # Robust normalization
        ctx_normed, loc, scale = self.norm(context)

        # Extract and normalize lags
        lags = extract_lags_v4(past_target, self.context_length, self.freq)
        lags_normed = (lags - loc.unsqueeze(1)) / scale.unsqueeze(1)

        all_preds = []
        for _ in range(self.num_samples):
            if self.use_gp:
                z_1 = self.net.gp.conditional_sample(ctx_normed, self.prediction_length, device)
            else:
                z_1 = torch.randn(B, self.prediction_length, device=device)

            t = torch.ones(B, device=device)
            h = torch.ones(B, device=device)
            u = self.net(z_1, (t, h), lags_normed)
            pred_normed = z_1 - u
            pred = self.norm.inverse(pred_normed, loc, scale)
            all_preds.append(pred)

        return torch.stack(all_preds, dim=1)
