"""
Solar-specialized MeanFlow: Seasonal decomposition + zero-aware modeling.

Key insight: Solar data is 50% zeros (nighttime). The model should:
1. Decompose into daily pattern + residual
2. Model the residual with MeanFlow (much easier distribution)
3. Use hour-of-day to gate predictions (hard zero at night)

This is a form of "deseasonalization" that simplifies what the flow model needs to learn.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .model_v3 import S4DBlock, SinusoidalPosEmb, sample_t_r
from .model_v4 import RobustNorm, get_lag_indices_v4, extract_lags_v4


class LearnableDailyPattern(nn.Module):
    """
    Learns a per-hour-of-day scaling pattern.
    Solar output has a strong daily cycle: zero at night, peak at noon.
    This module learns this pattern and factors it out.
    """
    def __init__(self, period=24):
        super().__init__()
        self.period = period
        # Learnable daily pattern (initialized to 1 = no effect)
        self.pattern = nn.Parameter(torch.ones(period))
        # Learnable zero gate: probability of being zero per hour
        self.zero_logits = nn.Parameter(torch.zeros(period))

    def get_pattern(self, hour_indices):
        """
        hour_indices: (B, T) integer hour-of-day [0-23]
        Returns: scale (B, T), zero_prob (B, T)
        """
        scale = F.softplus(self.pattern[hour_indices])  # always positive
        zero_prob = torch.sigmoid(self.zero_logits[hour_indices])
        return scale, zero_prob

    def remove_pattern(self, x, hour_indices):
        """Deseasonalize: divide by daily pattern."""
        scale, _ = self.get_pattern(hour_indices)
        return x / scale.clamp(min=1e-6)

    def add_pattern(self, x, hour_indices):
        """Re-seasonalize: multiply by daily pattern."""
        scale, zero_prob = self.get_pattern(hour_indices)
        # Gate: at nighttime hours (high zero_prob), push output toward zero
        return x * scale * (1 - zero_prob)


class SolarMeanFlowNet(nn.Module):
    """
    Solar-specialized MeanFlow with seasonal decomposition.

    Architecture:
    1. Extract hour-of-day from time features
    2. Deseasonalize context using learned daily pattern
    3. Run S4D MeanFlow on deseasonalized signal
    4. Re-seasonalize output
    5. Apply learned nighttime gating
    """

    def __init__(self, pred_len=24, ctx_len=72, d_model=192, n_s4d_blocks=6,
                 ssm_dim=64, time_emb_dim=64, dropout=0.1, n_lags=10,
                 n_time_features=4):
        super().__init__()
        self.pred_len = pred_len
        self.ctx_len = ctx_len
        self.d_model = d_model

        # Daily pattern module
        self.daily = LearnableDailyPattern(period=24)

        # Input: 1 (deseasonalized context) + n_lags + n_time_features
        n_input = 1 + n_lags + n_time_features
        self.n_lags = n_lags
        self.n_time_features = n_time_features

        # Time embedding (MeanFlow t, h)
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        emb_dim = time_emb_dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim * 2, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim),
        )

        # Context encoder
        self.ctx_proj = nn.Linear(n_input, d_model)
        self.ctx_blocks = nn.ModuleList([
            S4DBlock(d_model, N=ssm_dim, dropout=dropout, emb_dim=emb_dim)
            for _ in range(3)
        ])
        self.ctx_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(d_model, emb_dim),
        )
        self.ctx_to_pred = nn.Linear(ctx_len, pred_len)
        self.ctx_feat_proj = nn.Linear(d_model, d_model)

        # Future time feature projection
        self.future_time_proj = nn.Linear(n_time_features, d_model)

        # Prediction decoder
        self.pred_proj = nn.Linear(1, d_model)
        self.pred_blocks = nn.ModuleList([
            S4DBlock(d_model, N=ssm_dim, dropout=dropout, emb_dim=emb_dim)
            for _ in range(n_s4d_blocks)
        ])

        # Output
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, noisy_pred, time_steps, context_channels, future_time_feat=None):
        """
        noisy_pred: (B, pred_len) — in deseasonalized, normalized space
        time_steps: (t, h)
        context_channels: (B, n_input, ctx_len) — deseasonalized + lags + time feats
        future_time_feat: (B, n_tf, pred_len)
        """
        t, h = time_steps
        emb = torch.cat([self.time_emb(t), self.time_emb(h)], dim=-1)
        emb = self.time_mlp(emb)

        ctx = self.ctx_proj(context_channels.transpose(1, 2))
        for block in self.ctx_blocks:
            ctx = block(ctx, emb)
        emb = emb + self.ctx_pool(ctx.transpose(1, 2))

        ctx_spatial = self.ctx_feat_proj(
            self.ctx_to_pred(ctx.transpose(1, 2)).transpose(1, 2)
        )

        pred = self.pred_proj(noisy_pred.unsqueeze(-1)) + ctx_spatial

        if future_time_feat is not None:
            ft = self.future_time_proj(future_time_feat.transpose(1, 2))
            pred = pred + ft

        for block in self.pred_blocks:
            pred = block(pred, emb)

        return self.out_proj(F.silu(self.out_norm(pred))).squeeze(-1)


def solar_meanflow_loss(net, future_deseas_normed, context_channels,
                        future_time_feat=None, norm_p=0.75, norm_eps=1e-3):
    """MeanFlow JVP loss operating in deseasonalized space."""
    B = future_deseas_normed.shape[0]
    device = future_deseas_normed.device
    e = torch.randn_like(future_deseas_normed)
    t, r = sample_t_r(B, device)
    t_bc, r_bc = t.unsqueeze(-1), r.unsqueeze(-1)

    z = (1 - t_bc) * future_deseas_normed + t_bc * e
    v = e - future_deseas_normed

    def u_func(z, t_bc, r_bc):
        h_bc = t_bc - r_bc
        return net(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)),
                   context_channels, future_time_feat)

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


def hour_from_time_feat(time_feat):
    """
    Extract integer hour-of-day from GluonTS time features.
    GluonTS hour_of_day feature is in [-0.5, 0.5], maps to [0, 23].
    time_feat: (B, T, n_tf) or (B, n_tf, T)
    Returns: (B, T) integer hours
    """
    # First feature is hour_of_day, normalized to [-0.5, 0.5]
    if time_feat.shape[-1] < time_feat.shape[1]:
        # (B, T, n_tf) format
        hour_feat = time_feat[:, :, 0]
    else:
        # (B, n_tf, T) format
        hour_feat = time_feat[:, 0, :]
    # Map from [-0.5, 0.5] to [0, 23]
    hours = ((hour_feat + 0.5) * 23).round().long().clamp(0, 23)
    return hours


class SolarForecaster(nn.Module):
    """Forecaster with seasonal decomposition for solar data."""

    def __init__(self, net, context_length, prediction_length, num_samples=100):
        super().__init__()
        self.net = net
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.norm = RobustNorm()

    def forward(self, past_target, past_observed_values, past_time_feat=None,
                future_time_feat=None, **kwargs):
        device = past_target.device
        B = past_target.shape[0]
        context = past_target[:, -self.context_length:]

        # Get hours for context and future
        if past_time_feat is not None:
            ctx_hours = hour_from_time_feat(past_time_feat)[:, -self.context_length:]
            fut_hours = hour_from_time_feat(future_time_feat)[:, :self.prediction_length]
        else:
            # Fallback: assume sequential hours
            ctx_hours = torch.arange(self.context_length, device=device).unsqueeze(0) % 24
            ctx_hours = ctx_hours.expand(B, -1)
            fut_hours = torch.arange(self.prediction_length, device=device).unsqueeze(0) % 24
            fut_hours = fut_hours.expand(B, -1)

        # Deseasonalize context
        ctx_deseas = self.net.daily.remove_pattern(context, ctx_hours)

        # Normalize deseasonalized context
        ctx_normed, loc, scale = self.norm(ctx_deseas)

        # Build context channels with lags (on original data, then deseasonalize)
        lags = extract_lags_v4(past_target, self.context_length, 'H')
        # Deseasonalize lags using context hours (approximate)
        lags_deseas = lags.clone()
        lags_deseas[:, 0:1, :] = ctx_deseas.unsqueeze(1)  # first channel is context
        lags_normed = (lags_deseas - loc.unsqueeze(1)) / scale.unsqueeze(1)

        # Add time features
        if past_time_feat is not None:
            ptf = past_time_feat
            if ptf.shape[2] < ptf.shape[1]:
                ptf = ptf.transpose(1, 2)
            tf_ctx = ptf[:, :, -self.context_length:]
            ctx_channels = torch.cat([lags_normed, tf_ctx], dim=1)
        else:
            ctx_channels = lags_normed

        # Future time features
        ft_feat = None
        if future_time_feat is not None:
            ftf = future_time_feat
            if ftf.shape[2] < ftf.shape[1]:
                ftf = ftf.transpose(1, 2)
            ft_feat = ftf[:, :, :self.prediction_length]

        all_preds = []
        for _ in range(self.num_samples):
            z_1 = torch.randn(B, self.prediction_length, device=device)
            t = torch.ones(B, device=device)
            h = torch.ones(B, device=device)
            u = self.net(z_1, (t, h), ctx_channels, ft_feat)
            pred_deseas_normed = z_1 - u

            # De-normalize
            pred_deseas = self.norm.inverse(pred_deseas_normed, loc, scale)
            # Re-seasonalize
            pred = self.net.daily.add_pattern(pred_deseas, fut_hours)
            # Clamp to non-negative (solar can't be negative)
            pred = pred.clamp(min=0)
            all_preds.append(pred)

        return torch.stack(all_preds, dim=1)
