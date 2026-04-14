"""
MeanFlow-TS v6: v4 S4D + time features + extended training.

Key addition: time-of-day/day-of-week features passed to the model.
This is critical for solar (zeros at night) and traffic (rush hours).
TSFlow uses these but our v3/v4 models missed them.

Also: longer context for exchange rate to give more data to the small model.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .model_v3 import S4DKernel, S4DBlock, SinusoidalPosEmb, sample_t_r
from .model_v4 import RobustNorm, get_lag_indices_v4, extract_lags_v4


class S4DMeanFlowNetV6(nn.Module):
    """
    S4D MeanFlow v6: adds time features as input channels.

    Time features (hour-of-day, day-of-week, etc.) from GluonTS are
    passed as additional channels alongside lag features.
    """

    def __init__(self, pred_len=24, ctx_len=72, d_model=192, n_s4d_blocks=6,
                 ssm_dim=64, time_emb_dim=64, dropout=0.1, n_lags=10, freq="H",
                 n_time_features=0):
        super().__init__()
        self.pred_len = pred_len
        self.ctx_len = ctx_len
        self.d_model = d_model
        self.freq = freq
        self.n_lags = n_lags
        self.n_time_features = n_time_features

        # Input channels: 1 (context) + n_lags + n_time_features
        n_input_channels = 1 + n_lags + n_time_features

        # Time embedding (MeanFlow t, h)
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        emb_dim = time_emb_dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim * 2, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim),
        )

        # Context encoder
        self.ctx_proj = nn.Linear(n_input_channels, d_model)
        self.ctx_blocks = nn.ModuleList([
            S4DBlock(d_model, N=ssm_dim, dropout=dropout, emb_dim=emb_dim)
            for _ in range(3)
        ])
        self.ctx_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(d_model, emb_dim),
        )
        self.ctx_to_pred = nn.Linear(ctx_len, pred_len)
        self.ctx_feat_proj = nn.Linear(d_model, d_model)

        # Also project future time features to prediction
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

        # Output (zero-init)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, noisy_pred, time_steps, context_with_lags,
                future_time_feat=None):
        """
        noisy_pred: (B, pred_len)
        time_steps: (t, h) each (B,)
        context_with_lags: (B, n_channels, ctx_len) — includes time feats
        future_time_feat: optional (B, n_time_features, pred_len)
        """
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

        # Add future time features if available
        if future_time_feat is not None and self.future_time_proj is not None:
            # (B, n_tf, pred_len) -> (B, pred_len, n_tf) -> (B, pred_len, d_model)
            ft = self.future_time_proj(future_time_feat.transpose(1, 2))
            pred = pred + ft

        for block in self.pred_blocks:
            pred = block(pred, emb)

        return self.out_proj(F.silu(self.out_norm(pred))).squeeze(-1)


def extract_context_channels(past_target, past_time_feat, ctx_len, freq="H"):
    """
    Build context input channels: [context, lags, time_features].

    past_target: (B, past_len)
    past_time_feat: (B, n_tf, past_len) — GluonTS time features
    Returns: (B, 1 + n_lags + n_tf, ctx_len)
    """
    # Lag features
    lags = extract_lags_v4(past_target, ctx_len, freq)  # (B, 1+n_lags, ctx_len)

    # Time features for context window
    if past_time_feat is not None and past_time_feat.shape[-1] >= ctx_len:
        tf_ctx = past_time_feat[:, :, -ctx_len:]  # (B, n_tf, ctx_len)
        return torch.cat([lags, tf_ctx], dim=1)
    else:
        return lags


def extract_future_time_feat(past_time_feat, future_time_feat, pred_len):
    """
    Get time features for the prediction window.
    future_time_feat: (B, n_tf, fut_len) from GluonTS
    """
    if future_time_feat is not None:
        return future_time_feat[:, :, :pred_len]
    return None


def v6_meanflow_loss(net, future_clean, context_channels, future_time_feat=None,
                     norm_p=0.75, norm_eps=1e-3):
    """MeanFlow JVP loss for v6 with time features."""
    B = future_clean.shape[0]
    device = future_clean.device
    e = torch.randn_like(future_clean)
    t, r = sample_t_r(B, device)
    t_bc, r_bc = t.unsqueeze(-1), r.unsqueeze(-1)

    z = (1 - t_bc) * future_clean + t_bc * e
    v = e - future_clean

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


class V6Forecaster(nn.Module):
    """GluonTS forecaster with time features."""

    def __init__(self, net, context_length, prediction_length, num_samples=100,
                 freq="H"):
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

        # Build context channels (lags + time features)
        lags = extract_lags_v4(past_target, self.context_length, self.freq)
        lags_normed = (lags - loc.unsqueeze(1)) / scale.unsqueeze(1)

        if past_time_feat is not None:
            # GluonTS gives (B, T, n_tf), we need (B, n_tf, T)
            # If shape[2] < shape[1], it's (B, T, n_tf) — transpose
            ptf = past_time_feat
            if ptf.shape[2] < ptf.shape[1]:
                ptf = ptf.transpose(1, 2)  # -> (B, n_tf, T)
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
            pred_normed = z_1 - u
            pred = self.norm.inverse(pred_normed, loc, scale)
            all_preds.append(pred)

        return torch.stack(all_preds, dim=1)
