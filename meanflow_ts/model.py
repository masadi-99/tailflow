"""
MeanFlow-TS: Core model and loss.

Conditional MeanFlow for time series forecasting.
MeanFlow operates on the prediction window only, with context as conditioning.
Uses JVP-based self-consistency loss from Geng et al. (2025).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock1D(nn.Module):
    """Residual block with FiLM conditioning."""

    def __init__(self, channels, emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, channels * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.emb_proj(F.silu(emb)).chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return x + h


class ConditionalMeanFlowNet(nn.Module):
    """
    Predicts average velocity for the prediction region only.
    Context is injected as conditioning.

    Input: noisy_pred (B, pred_len) — noisy future
    Conditioning: context (B, ctx_len) — observed past
    Time: (t, h) — MeanFlow dual time
    Output: velocity (B, pred_len)
    """

    def __init__(self, pred_len=24, ctx_len=24, model_channels=128,
                 num_res_blocks=4, time_emb_dim=64, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.ctx_len = ctx_len

        # Dual time embedding (same encoder for t and h, then concat — as in MeanFlow paper)
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        emb_dim = time_emb_dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim * 2, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
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

        # Output (zero-init for stable training start)
        self.out_norm = nn.GroupNorm(min(8, model_channels), model_channels)
        self.out_proj = nn.Conv1d(model_channels, 1, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, noisy_pred, time_steps, context):
        t, h = time_steps
        emb = torch.cat([self.time_emb(t), self.time_emb(h)], dim=-1)
        emb = self.time_mlp(emb)

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
# MeanFlow loss
# ============================================================

def logit_normal_sample(P_mean, P_std, n, device):
    rnd = torch.randn(n, device=device)
    return torch.sigmoid(rnd * P_std + P_mean).clip(1e-5, 1 - 1e-5)


def sample_t_r(n, device, ratio=0.75):
    t = logit_normal_sample(-0.6, 1.6, n, device)
    r = logit_normal_sample(-4.0, 1.6, n, device)
    t, r = torch.maximum(t, r), torch.minimum(t, r)
    mask = torch.rand(n, device=device) < (1 - ratio)
    r = torch.where(mask, t, r)
    return t, r


def conditional_meanflow_loss(net, future_clean, context, norm_p=0.75, norm_eps=1e-3):
    """
    MeanFlow self-consistency loss via JVP (Geng et al., 2025).

    The network predicts average velocity u(z_t, r, t). The JVP computes
    du/dt along the flow, and the self-consistency identity
    u = v - (t-r) * du/dt provides the training target.

    Args:
        net: ConditionalMeanFlowNet
        future_clean: (B, pred_len) — scaled future target
        context: (B, ctx_len) — scaled observed context
    """
    B = future_clean.shape[0]
    device = future_clean.device
    e = torch.randn_like(future_clean)
    t, r = sample_t_r(B, device)
    t_bc = t.unsqueeze(-1)
    r_bc = r.unsqueeze(-1)

    z = (1 - t_bc) * future_clean + t_bc * e
    v = e - future_clean

    def u_func(z, t_bc, r_bc):
        h_bc = t_bc - r_bc
        return net(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)), context)

    dtdt = torch.ones_like(t_bc)
    drdt = torch.zeros_like(r_bc)

    with torch.amp.autocast("cuda", enabled=False):
        u_pred, dudt = torch.func.jvp(u_func, (z, t_bc, r_bc), (v, dtdt, drdt))
        u_tgt = (v - (t_bc - r_bc) * dudt).detach()
        loss = (u_pred - u_tgt) ** 2
        loss = loss.sum(dim=1)
        adp_wt = (loss.detach() + norm_eps) ** norm_p
        loss = (loss / adp_wt).mean()
    return loss


# ============================================================
# Unconditional MeanFlow (for Table 1 & 2: generation quality)
# ============================================================

class UnconditionalMeanFlowNet(nn.Module):
    """
    Generates full time series unconditionally.
    Input: noisy_ts (B, seq_len), time: (t, h)
    Output: velocity (B, seq_len)
    """

    def __init__(self, seq_len=48, model_channels=128, num_res_blocks=4,
                 time_emb_dim=64, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len

        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        emb_dim = time_emb_dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim * 2, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim),
        )

        self.input_proj = nn.Conv1d(1, model_channels, 1)
        self.blocks = nn.ModuleList([
            ResBlock1D(model_channels, emb_dim, dropout) for _ in range(num_res_blocks)
        ])
        self.out_norm = nn.GroupNorm(min(8, model_channels), model_channels)
        self.out_proj = nn.Conv1d(model_channels, 1, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, time_steps):
        t, h = time_steps
        emb = torch.cat([self.time_emb(t), self.time_emb(h)], dim=-1)
        emb = self.time_mlp(emb)

        x = self.input_proj(x.unsqueeze(1))
        for block in self.blocks:
            x = block(x, emb)
        return self.out_proj(F.silu(self.out_norm(x))).squeeze(1)


def unconditional_meanflow_loss(net, x_clean, norm_p=0.75, norm_eps=1e-3):
    """MeanFlow JVP loss for unconditional generation."""
    B = x_clean.shape[0]
    device = x_clean.device
    e = torch.randn_like(x_clean)
    t, r = sample_t_r(B, device)
    t_bc, r_bc = t.unsqueeze(-1), r.unsqueeze(-1)

    z = (1 - t_bc) * x_clean + t_bc * e
    v = e - x_clean

    def u_func(z, t_bc, r_bc):
        h_bc = t_bc - r_bc
        return net(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)))

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


@torch.no_grad()
def meanflow_sample(net, shape, device):
    """One-step generation: z_0 = z_1 - u(z_1, t=1, h=1)."""
    z_1 = torch.randn(shape, device=device)
    t = torch.ones(shape[0], device=device)
    h = torch.ones(shape[0], device=device)
    if hasattr(net, 'ctx_len'):
        raise ValueError("Use MeanFlowForecaster for conditional models")
    u = net(z_1, (t, h))
    return z_1 - u


# ============================================================
# GluonTS-compatible forecaster
# ============================================================

class MeanFlowForecaster(nn.Module):
    """Wraps ConditionalMeanFlowNet for GluonTS evaluation."""

    def __init__(self, net, context_length, prediction_length, num_samples=100):
        super().__init__()
        self.net = net
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples

    def forward(self, past_target, past_observed_values, **kwargs):
        device = past_target.device
        B = past_target.shape[0]
        context = past_target[:, -self.context_length:]
        loc = context.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
        scaled_ctx = context / loc

        all_preds = []
        for _ in range(self.num_samples):
            z_1 = torch.randn(B, self.prediction_length, device=device)
            t = torch.ones(B, device=device)
            h = torch.ones(B, device=device)
            u = self.net(z_1, (t, h), scaled_ctx)
            all_preds.append((z_1 - u) * loc)

        return torch.stack(all_preds, dim=1)
