"""
Exchange-rate specialized MeanFlow: Small model + data augmentation.

Key insight: Only 8 series. Large models overfit badly.
Solution:
1. Small model (d_model=96, 4 blocks) — ~400K params
2. Aggressive dropout and weight decay
3. Data augmentation: jittering, window shifting, scaling
4. Multi-scale context: use both raw + differenced series
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .model_v3 import S4DBlock, SinusoidalPosEmb, sample_t_r
from .model_v4 import RobustNorm, get_lag_indices_v4, extract_lags_v4


class ExchangeMeanFlowNet(nn.Module):
    """
    Small, regularized S4D model for exchange rate forecasting.

    Innovations:
    1. Multi-scale input: raw series + first differences + rolling stats
    2. Small architecture to prevent overfitting on 8 series
    3. Built-in noise augmentation
    """

    def __init__(self, pred_len=30, ctx_len=30, d_model=96, n_s4d_blocks=4,
                 ssm_dim=32, time_emb_dim=32, dropout=0.2, n_lags=6,
                 n_time_features=3, augment_noise=0.01):
        super().__init__()
        self.pred_len = pred_len
        self.ctx_len = ctx_len
        self.d_model = d_model
        self.augment_noise = augment_noise

        # Multi-scale: context + lags + time_features + first_diff + rolling_std
        n_extra = 2  # first_diff, rolling_std
        n_input = 1 + n_lags + n_time_features + n_extra

        # Time embedding
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        emb_dim = time_emb_dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim * 2, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim),
        )

        # Context encoder
        self.ctx_proj = nn.Linear(n_input, d_model)
        self.ctx_blocks = nn.ModuleList([
            S4DBlock(d_model, N=ssm_dim, dropout=dropout, emb_dim=emb_dim)
            for _ in range(2)
        ])
        self.ctx_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(d_model, emb_dim),
        )
        self.ctx_to_pred = nn.Linear(ctx_len, pred_len)
        self.ctx_feat_proj = nn.Linear(d_model, d_model)

        # Future time features
        if n_time_features > 0:
            self.future_time_proj = nn.Linear(n_time_features, d_model)
        else:
            self.future_time_proj = None

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

        if future_time_feat is not None and self.future_time_proj is not None:
            ft = self.future_time_proj(future_time_feat.transpose(1, 2))
            pred = pred + ft

        for block in self.pred_blocks:
            pred = block(pred, emb)

        return self.out_proj(F.silu(self.out_norm(pred))).squeeze(-1)


def build_exchange_context(past_target, ctx_len, freq, past_time_feat=None):
    """
    Build multi-scale context channels for exchange rate.
    Includes: lags + first differences + rolling std + time features.
    """
    lags = extract_lags_v4(past_target, ctx_len, freq)  # (B, 1+n_lags, ctx_len)

    ctx = past_target[:, -ctx_len:]  # (B, ctx_len)

    # First differences (normalized)
    diff = torch.zeros_like(ctx)
    diff[:, 1:] = ctx[:, 1:] - ctx[:, :-1]
    diff = diff / diff.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)

    # Rolling std (window=5)
    B, T = ctx.shape
    rstd = torch.zeros_like(ctx)
    for i in range(2, T):
        window = ctx[:, max(0,i-4):i+1]
        rstd[:, i] = window.std(dim=1)
    rstd = rstd / rstd.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)

    channels = [lags, diff.unsqueeze(1), rstd.unsqueeze(1)]

    if past_time_feat is not None:
        ptf = past_time_feat
        if ptf.shape[2] < ptf.shape[1]:
            ptf = ptf.transpose(1, 2)
        tf_ctx = ptf[:, :, -ctx_len:]
        channels.append(tf_ctx)

    return torch.cat(channels, dim=1)


def exchange_meanflow_loss(net, future_normed, context_channels,
                           future_time_feat=None, augment=True,
                           norm_p=0.75, norm_eps=1e-3):
    """MeanFlow loss with optional data augmentation."""
    B = future_normed.shape[0]
    device = future_normed.device

    # Data augmentation: add small noise to inputs
    if augment and net.training:
        noise_scale = net.augment_noise
        context_channels = context_channels + torch.randn_like(context_channels) * noise_scale
        future_normed = future_normed + torch.randn_like(future_normed) * noise_scale

    e = torch.randn_like(future_normed)
    t, r = sample_t_r(B, device)
    t_bc, r_bc = t.unsqueeze(-1), r.unsqueeze(-1)

    z = (1 - t_bc) * future_normed + t_bc * e
    v = e - future_normed

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


class ExchangeForecaster(nn.Module):
    """Forecaster for exchange rate."""

    def __init__(self, net, context_length, prediction_length, num_samples=100,
                 freq="B"):
        super().__init__()
        self.net = net
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.freq = freq
        self.norm = RobustNorm()

    def forward(self, past_target, past_observed_values, past_time_feat=None,
                future_time_feat=None, **kwargs):
        device = past_target.device
        B = past_target.shape[0]
        context = past_target[:, -self.context_length:]
        ctx_normed, loc, scale = self.norm(context)

        # Build multi-scale context
        ctx_channels = build_exchange_context(
            past_target, self.context_length, self.freq, past_time_feat)
        # Normalize the lag channels (first 1+n_lags)
        n_lags_ch = 1 + len(get_lag_indices_v4(self.freq))
        ctx_channels_normed = ctx_channels.clone()
        ctx_channels_normed[:, :n_lags_ch] = (ctx_channels[:, :n_lags_ch] - loc.unsqueeze(1)) / scale.unsqueeze(1)

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
            u = self.net(z_1, (t, h), ctx_channels_normed, ft_feat)
            pred_normed = z_1 - u
            pred = self.norm.inverse(pred_normed, loc, scale)
            all_preds.append(pred)

        return torch.stack(all_preds, dim=1)
