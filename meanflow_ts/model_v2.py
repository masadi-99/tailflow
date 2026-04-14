"""
MeanFlow-TS v2: Add lag features to improve forecasting quality.

Improvement over v1:
- Context now includes lag features (same-hour values from 1-7 days ago)
- This provides seasonality information similar to TSFlow's lag features
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import (
    SinusoidalPosEmb, ResBlock1D, MeanFlowForecaster,
    conditional_meanflow_loss, sample_t_r, logit_normal_sample,
    unconditional_meanflow_loss, meanflow_sample,
    UnconditionalMeanFlowNet,
)


class ConditionalMeanFlowNetV2(nn.Module):
    """
    MeanFlow net with lag feature support.
    Context includes both the immediate window and lag features.
    """

    def __init__(self, pred_len=24, ctx_len=24, n_lags=7,
                 model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.ctx_len = ctx_len
        self.n_lags = n_lags  # number of daily lags

        # Dual time embedding
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        emb_dim = time_emb_dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim * 2, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim),
        )

        # Context encoder: now takes (1 + n_lags) channels
        ctx_channels = 1 + n_lags
        self.ctx_proj = nn.Conv1d(ctx_channels, model_channels, 1)
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

        # Output
        self.out_norm = nn.GroupNorm(min(8, model_channels), model_channels)
        self.out_proj = nn.Conv1d(model_channels, 1, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, noisy_pred, time_steps, context_with_lags):
        """
        noisy_pred: (B, pred_len)
        time_steps: (t, h) each (B,)
        context_with_lags: (B, 1+n_lags, ctx_len) — channel 0 is immediate context,
                          channels 1..n_lags are lag features
        """
        t, h = time_steps
        emb = torch.cat([self.time_emb(t), self.time_emb(h)], dim=-1)
        emb = self.time_mlp(emb)

        # Encode context with lags
        ctx = self.ctx_proj(context_with_lags)  # (B, C, ctx_len)
        for block in self.ctx_blocks:
            ctx = block(ctx, emb)
        emb = emb + self.ctx_pool(ctx)
        ctx_spatial = self.ctx_feat_proj(self.ctx_to_pred(ctx))

        # Process prediction
        pred = self.pred_proj(noisy_pred.unsqueeze(1)) + ctx_spatial
        for block in self.pred_blocks:
            pred = block(pred, emb)

        return self.out_proj(F.silu(self.out_norm(pred))).squeeze(1)


def extract_lag_features(past_target, ctx_len, freq="H", n_lags=7):
    """
    Extract lag features from past_target.
    For hourly data: lags at 24h, 48h, ..., 24*n_lags hours ago.
    For business daily: lags at 5, 10, ..., 5*n_lags days ago.

    past_target: (B, past_len)
    Returns: (B, 1+n_lags, ctx_len)
    """
    B = past_target.shape[0]
    context = past_target[:, -ctx_len:]  # (B, ctx_len)

    if freq == "H":
        lag_offsets = [24 * (i + 1) for i in range(n_lags)]  # 24, 48, 72, ...
    elif freq == "B":
        lag_offsets = [5 * (i + 1) for i in range(n_lags)]   # 5, 10, 15, ...
    else:
        lag_offsets = [24 * (i + 1) for i in range(n_lags)]   # default hourly

    channels = [context.unsqueeze(1)]  # (B, 1, ctx_len)
    for offset in lag_offsets:
        # Get lagged values: past_target at position -(ctx_len + offset) : -offset
        start = past_target.shape[1] - ctx_len - offset
        end = past_target.shape[1] - offset
        if start >= 0:
            lag = past_target[:, start:end]  # (B, ctx_len)
        else:
            # Not enough history — pad with zeros
            lag = torch.zeros(B, ctx_len, device=past_target.device)
            if end > 0:
                available = past_target[:, :end]
                lag[:, -available.shape[1]:] = available
        channels.append(lag.unsqueeze(1))

    return torch.cat(channels, dim=1)  # (B, 1+n_lags, ctx_len)


class MeanFlowForecasterV2(nn.Module):
    """Forecaster with lag features."""

    def __init__(self, net, context_length, prediction_length, num_samples=16,
                 freq="H", n_lags=7):
        super().__init__()
        self.net = net
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.freq = freq
        self.n_lags = n_lags

    def forward(self, past_target, past_observed_values, **kwargs):
        device = past_target.device
        B = past_target.shape[0]
        context = past_target[:, -self.context_length:]
        loc = context.abs().mean(dim=1, keepdim=True).clamp(min=0.01)

        # Extract lag features and scale
        ctx_with_lags = extract_lag_features(
            past_target, self.context_length, self.freq, self.n_lags
        ) / loc.unsqueeze(1)  # scale all channels

        all_preds = []
        for _ in range(self.num_samples):
            z_1 = torch.randn(B, self.prediction_length, device=device)
            t = torch.ones(B, device=device)
            h = torch.ones(B, device=device)
            u = self.net(z_1, (t, h), ctx_with_lags)
            all_preds.append((z_1 - u) * loc)

        return torch.stack(all_preds, dim=1)


def conditional_meanflow_loss_v2(net, future_clean, context_with_lags,
                                  norm_p=0.75, norm_eps=1e-3):
    """MeanFlow JVP loss with lag-enhanced context."""
    B = future_clean.shape[0]
    device = future_clean.device
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
