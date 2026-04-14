"""
MeanFlow-TS v5: Cross-attention + bidirectional S4D + improved solar handling.

Pushing beyond TSFlow with:
1. Cross-attention from prediction tokens to context tokens (replaces linear projection)
2. Bidirectional S4D blocks (forward + backward state space)
3. Observed-value mask channel (helps with solar zeros)
4. Stochastic depth for regularization
5. SwiGLU MLP in S4D blocks (modern transformer-style)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .model_v3 import S4DKernel, SinusoidalPosEmb, sample_t_r
from .model_v4 import RobustNorm, get_lag_indices_v4, extract_lags_v4


# ============================================================
# Bidirectional S4D Block
# ============================================================

class BidirectionalS4DBlock(nn.Module):
    """
    Bidirectional S4D: forward kernel + backward kernel.
    Improves modeling of sequences with future context.
    """
    def __init__(self, d_model, N=64, dropout=0.1, emb_dim=None,
                 stochastic_depth=0.0):
        super().__init__()
        self.d_model = d_model
        self.kernel_fwd = S4DKernel(d_model, N=N)
        self.kernel_bwd = S4DKernel(d_model, N=N)
        self.D = nn.Parameter(torch.randn(d_model) * 0.01)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.stochastic_depth = stochastic_depth

        # SwiGLU output
        self.gate_proj = nn.Linear(d_model, d_model * 2)
        self.up_proj = nn.Linear(d_model, d_model * 2)
        self.down_proj = nn.Linear(d_model * 2, d_model)

        # Project bidirectional outputs to one
        self.bidir_proj = nn.Linear(d_model * 2, d_model)

        if emb_dim is not None:
            self.film = nn.Linear(emb_dim, d_model * 2)
        else:
            self.film = None

    def conv_fft(self, x_t, K):
        """Causal FFT convolution. x_t: (B, d, L), K: (d, L)."""
        L = x_t.shape[-1]
        K_f = torch.fft.rfft(K, n=2*L)
        x_f = torch.fft.rfft(x_t, n=2*L)
        y_f = K_f.unsqueeze(0) * x_f
        y = torch.fft.irfft(y_f, n=2*L)[..., :L]
        return y

    def forward(self, x, emb=None):
        # Stochastic depth (during training)
        if self.training and self.stochastic_depth > 0:
            if torch.rand(1).item() < self.stochastic_depth:
                return x

        residual = x
        x = self.norm(x)
        L = x.shape[1]

        # Forward direction
        K_fwd = self.kernel_fwd(L)
        x_t = x.transpose(1, 2)  # (B, d, L)
        y_fwd = self.conv_fft(x_t, K_fwd)

        # Backward direction
        K_bwd = self.kernel_bwd(L)
        x_t_rev = torch.flip(x_t, dims=[-1])
        y_bwd_rev = self.conv_fft(x_t_rev, K_bwd)
        y_bwd = torch.flip(y_bwd_rev, dims=[-1])

        # Combine bidirectional
        y_combined = torch.cat([y_fwd, y_bwd], dim=1)  # (B, 2d, L)
        y = y_combined.transpose(1, 2)  # (B, L, 2d)
        y = self.bidir_proj(y)  # (B, L, d)

        # Skip connection
        y = y + x_t.transpose(1, 2) * self.D.unsqueeze(0).unsqueeze(0)

        # FiLM conditioning
        if self.film is not None and emb is not None:
            scale, shift = self.film(F.silu(emb)).chunk(2, dim=-1)
            y = y * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # SwiGLU MLP
        gate = F.silu(self.gate_proj(y))
        up = self.up_proj(y)
        y = self.down_proj(gate * up)
        y = self.dropout(y)

        return residual + y


# ============================================================
# Cross-attention block
# ============================================================

class CrossAttention(nn.Module):
    """Cross-attention from query (prediction) to context."""
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        """
        query: (B, L_q, d) — prediction tokens
        context: (B, L_c, d) — context tokens
        Returns: (B, L_q, d)
        """
        B, L_q, d = query.shape
        L_c = context.shape[1]
        residual = query

        q = self.norm_q(query)
        kv = self.norm_kv(context)

        Q = self.q_proj(q).view(B, L_q, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(kv).view(B, L_c, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(kv).view(B, L_c, self.n_heads, self.d_head).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ V  # (B, h, L_q, d_head)
        out = out.transpose(1, 2).contiguous().view(B, L_q, d)
        out = self.out_proj(out)
        return residual + out


# ============================================================
# V5 Network
# ============================================================

class S4DMeanFlowNetV5(nn.Module):
    """
    V5: Bidirectional S4D + cross-attention + observation mask.
    """

    def __init__(self, pred_len=24, ctx_len=72, d_model=192, n_s4d_blocks=6,
                 ssm_dim=64, time_emb_dim=64, dropout=0.1, n_lags=10, freq="H",
                 n_attn_heads=4, stochastic_depth=0.05, use_obs_mask=True):
        super().__init__()
        self.pred_len = pred_len
        self.ctx_len = ctx_len
        self.d_model = d_model
        self.freq = freq
        self.n_lags = n_lags
        self.use_obs_mask = use_obs_mask
        n_input_channels = 1 + n_lags + (1 if use_obs_mask else 0)

        # Time embedding
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        emb_dim = time_emb_dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim * 2, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim),
        )

        # Context encoder: bidirectional S4D
        self.ctx_proj = nn.Linear(n_input_channels, d_model)
        self.ctx_blocks = nn.ModuleList([
            BidirectionalS4DBlock(d_model, N=ssm_dim, dropout=dropout,
                                   emb_dim=emb_dim, stochastic_depth=stochastic_depth)
            for _ in range(3)
        ])
        self.ctx_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(d_model, emb_dim),
        )

        # Prediction decoder: bidirectional S4D + cross-attention
        self.pred_proj = nn.Linear(1, d_model)
        self.pred_blocks = nn.ModuleList([
            BidirectionalS4DBlock(d_model, N=ssm_dim, dropout=dropout,
                                   emb_dim=emb_dim, stochastic_depth=stochastic_depth)
            for _ in range(n_s4d_blocks)
        ])
        # Cross-attention layers (every 2 blocks)
        self.cross_attns = nn.ModuleList([
            CrossAttention(d_model, n_heads=n_attn_heads, dropout=dropout)
            for _ in range(n_s4d_blocks // 2)
        ])

        # Output (zero-init)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, noisy_pred, time_steps, context_with_lags, obs_mask=None):
        """
        noisy_pred: (B, pred_len)
        time_steps: (t, h)
        context_with_lags: (B, n_channels, ctx_len)
        obs_mask: optional (B, ctx_len) — 1=observed, 0=zero/missing
        """
        t, h = time_steps
        emb = torch.cat([self.time_emb(t), self.time_emb(h)], dim=-1)
        emb = self.time_mlp(emb)

        # Add observation mask channel
        if self.use_obs_mask:
            if obs_mask is None:
                obs_mask = (context_with_lags[:, 0:1, :].abs() > 1e-6).float()
            ctx_input = torch.cat([context_with_lags, obs_mask], dim=1)
        else:
            ctx_input = context_with_lags

        # Encode context
        ctx = self.ctx_proj(ctx_input.transpose(1, 2))  # (B, ctx_len, d_model)
        for block in self.ctx_blocks:
            ctx = block(ctx, emb)

        emb = emb + self.ctx_pool(ctx.transpose(1, 2))

        # Decode prediction with cross-attention
        pred = self.pred_proj(noisy_pred.unsqueeze(-1))  # (B, pred_len, d_model)

        attn_idx = 0
        for i, block in enumerate(self.pred_blocks):
            pred = block(pred, emb)
            # Apply cross-attention every 2 blocks
            if i % 2 == 1 and attn_idx < len(self.cross_attns):
                pred = self.cross_attns[attn_idx](pred, ctx)
                attn_idx += 1

        return self.out_proj(F.silu(self.out_norm(pred))).squeeze(-1)


# ============================================================
# Loss
# ============================================================

def v5_meanflow_loss(net, future_clean, context_with_lags, obs_mask=None,
                     norm_p=0.75, norm_eps=1e-3):
    """MeanFlow JVP loss for v5 (Gaussian noise — GP found to hurt)."""
    B = future_clean.shape[0]
    device = future_clean.device
    e = torch.randn_like(future_clean)
    t, r = sample_t_r(B, device)
    t_bc, r_bc = t.unsqueeze(-1), r.unsqueeze(-1)

    z = (1 - t_bc) * future_clean + t_bc * e
    v = e - future_clean

    def u_func(z, t_bc, r_bc):
        h_bc = t_bc - r_bc
        return net(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)), context_with_lags, obs_mask)

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
# Forecaster
# ============================================================

class V5Forecaster(nn.Module):
    """GluonTS-compatible forecaster for v5."""

    def __init__(self, net, context_length, prediction_length, num_samples=100, freq="H"):
        super().__init__()
        self.net = net
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.freq = freq
        self.norm = RobustNorm()

    def forward(self, past_target, past_observed_values, **kwargs):
        device = past_target.device
        B = past_target.shape[0]
        context = past_target[:, -self.context_length:]

        ctx_normed, loc, scale = self.norm(context)
        lags = extract_lags_v4(past_target, self.context_length, self.freq)
        lags_normed = (lags - loc.unsqueeze(1)) / scale.unsqueeze(1)

        # Compute observation mask from raw context (before normalization)
        obs_mask = (context.abs() > 1e-6).float().unsqueeze(1)  # (B, 1, ctx_len)

        all_preds = []
        for _ in range(self.num_samples):
            z_1 = torch.randn(B, self.prediction_length, device=device)
            t = torch.ones(B, device=device)
            h = torch.ones(B, device=device)
            u = self.net(z_1, (t, h), lags_normed, obs_mask)
            pred_normed = z_1 - u
            pred = self.norm.inverse(pred_normed, loc, scale)
            all_preds.append(pred)

        return torch.stack(all_preds, dim=1)
