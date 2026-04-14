"""
MeanFlow-TS v3: S4D backbone + lag features + RevIN normalization.

Key upgrades over v1:
1. S4D (diagonal state space) blocks replacing 1D convolutions
2. Lag features (daily/weekly lags for hourly, weekly for business daily)
3. RevIN (Reversible Instance Normalization)
4. Deeper context encoder with cross-attention to prediction
5. Extremity conditioning adapter (zero-init, added in Phase 2)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# S4D: Diagonal State Space Model (pure PyTorch)
# ============================================================

class S4DKernel(nn.Module):
    """
    Diagonal State Space kernel (S4D from Gu et al., 2022).
    Computes the SSM convolution kernel in O(N*L) time.
    """
    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Diagonal A matrix: initialized with HiPPO
        # A = -1/2 + ni where n = 0..N-1 (S4D-Lin initialization)
        A_real = -0.5 * torch.ones(d_model, N)
        A_imag = math.pi * torch.arange(N).float().unsqueeze(0).expand(d_model, -1)
        self.A_real = nn.Parameter(A_real)
        self.A_imag = nn.Parameter(A_imag)

        # C: output matrix
        C = torch.randn(d_model, N, 2) * 0.5  # real and imag parts
        self.C = nn.Parameter(C)

        # dt: discretization step
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # B: input matrix (fixed to 1 for S4D)
        self.register_buffer('B', torch.ones(d_model, N))

    def forward(self, L):
        """Compute the SSM convolution kernel of length L."""
        dt = self.log_dt.exp()  # (d_model,)
        # Discretize: A_bar = exp(A * dt)
        A = torch.complex(self.A_real, self.A_imag)  # (d_model, N)
        dtA = A * dt.unsqueeze(-1)  # (d_model, N)

        C = torch.complex(self.C[..., 0], self.C[..., 1])  # (d_model, N)

        # Vandermonde-style computation: K = sum_n C_n * A_bar_n^k for k=0..L-1
        # K[k] = real(C @ diag(A_bar^k) @ B)
        A_bar = torch.exp(dtA)  # (d_model, N)
        # Powers: A_bar^0, A_bar^1, ..., A_bar^(L-1)
        # Shape: (d_model, N, L)
        powers = A_bar.unsqueeze(-1) ** torch.arange(L, device=A.device).float()

        # K = real(sum over N of C * B * powers)
        B = self.B.unsqueeze(-1)  # (d_model, N, 1)
        K = torch.einsum('dn,dnl->dl', C * self.B.to(C.dtype), powers)
        return K.real * dt.unsqueeze(-1)  # (d_model, L)


class S4DBlock(nn.Module):
    """
    Single S4D block: SSM convolution + gated MLP + residual + norm.
    """
    def __init__(self, d_model, N=64, dropout=0.1, emb_dim=None):
        super().__init__()
        self.d_model = d_model
        self.kernel = S4DKernel(d_model, N=N)
        self.D = nn.Parameter(torch.randn(d_model) * 0.01)  # skip connection
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Gated linear unit for mixing
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GLU(dim=-1),
        )

        # Optional FiLM conditioning from time embedding
        if emb_dim is not None:
            self.film = nn.Linear(emb_dim, d_model * 2)
        else:
            self.film = None

    def forward(self, x, emb=None):
        """
        x: (B, L, d_model)
        emb: (B, emb_dim) optional conditioning
        Returns: (B, L, d_model)
        """
        residual = x
        x = self.norm(x)

        # SSM convolution via FFT
        L = x.shape[1]
        K = self.kernel(L)  # (d_model, L)

        # Causal convolution: x * K
        x_t = x.transpose(1, 2)  # (B, d_model, L)
        # FFT-based convolution
        K_f = torch.fft.rfft(K, n=2*L)  # (d_model, L+1)
        x_f = torch.fft.rfft(x_t, n=2*L)  # (B, d_model, L+1)
        y_f = K_f.unsqueeze(0) * x_f  # (B, d_model, L+1)
        y = torch.fft.irfft(y_f, n=2*L)[..., :L]  # (B, d_model, L)

        # Add skip connection
        y = y + x_t * self.D.unsqueeze(0).unsqueeze(-1)
        y = y.transpose(1, 2)  # (B, L, d_model)

        # FiLM conditioning
        if self.film is not None and emb is not None:
            scale, shift = self.film(F.silu(emb)).chunk(2, dim=-1)
            y = y * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        y = self.dropout(self.out_proj(y))
        return residual + y


# ============================================================
# RevIN: Reversible Instance Normalization
# ============================================================

class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., 2022)."""
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """Normalize. x: (B, T). Returns normalized x, mean, std."""
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=self.eps)
        return (x - mean) / std, mean, std

    def inverse(self, x, mean, std):
        """Denormalize."""
        return x * std + mean


# ============================================================
# Lag Feature Extraction
# ============================================================

def get_lag_indices(freq):
    """Return lag offsets for a given frequency."""
    if freq == "H":
        # Hourly: 24h, 48h, 72h, 1w, 2w (like TSFlow)
        return [24, 48, 72, 168, 336]
    elif freq == "B":
        # Business daily: 5d, 10d, 15d, 20d, 25d
        return [5, 10, 15, 20, 25]
    else:
        return [24, 48, 168]


def extract_lags(past_target, ctx_len, freq="H"):
    """
    Extract lag features from past_target.
    past_target: (B, past_len)
    Returns: (B, n_channels, ctx_len) where n_channels = 1 + n_lags
    """
    B = past_target.shape[0]
    context = past_target[:, -ctx_len:]  # (B, ctx_len)
    lag_offsets = get_lag_indices(freq)

    channels = [context.unsqueeze(1)]  # (B, 1, ctx_len)
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

    return torch.cat(channels, dim=1)  # (B, 1+n_lags, ctx_len)


# ============================================================
# Time Embedding
# ============================================================

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


# ============================================================
# Main Model: S4D-MeanFlow
# ============================================================

class S4DMeanFlowNet(nn.Module):
    """
    MeanFlow with S4D backbone for time series forecasting.

    Architecture:
    - Context encoder: Linear projection + S4D blocks
    - Cross-conditioning: context summary injected via FiLM
    - Prediction decoder: Linear projection + S4D blocks + output head
    - Dual time embedding (t, h) as in MeanFlow
    - Lag features as additional input channels
    """

    def __init__(self, pred_len=24, ctx_len=24, d_model=128, n_s4d_blocks=4,
                 ssm_dim=64, time_emb_dim=64, dropout=0.1, n_lags=5, freq="H"):
        super().__init__()
        self.pred_len = pred_len
        self.ctx_len = ctx_len
        self.n_lags = n_lags
        self.freq = freq
        self.d_model = d_model

        n_input_channels = 1 + n_lags

        # Time embedding
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        emb_dim = time_emb_dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim * 2, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # Context encoder
        self.ctx_proj = nn.Linear(n_input_channels, d_model)
        self.ctx_blocks = nn.ModuleList([
            S4DBlock(d_model, N=ssm_dim, dropout=dropout, emb_dim=emb_dim)
            for _ in range(2)
        ])
        self.ctx_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(d_model, emb_dim),
        )

        # Context-to-prediction cross-projection
        self.ctx_to_pred = nn.Linear(ctx_len, pred_len)
        self.ctx_feat_proj = nn.Linear(d_model, d_model)

        # Prediction decoder
        self.pred_proj = nn.Linear(1, d_model)
        self.pred_blocks = nn.ModuleList([
            S4DBlock(d_model, N=ssm_dim, dropout=dropout, emb_dim=emb_dim)
            for _ in range(n_s4d_blocks)
        ])

        # Output head (zero-init for stable start)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, noisy_pred, time_steps, context_with_lags):
        """
        noisy_pred: (B, pred_len)
        time_steps: (t, h) each (B,)
        context_with_lags: (B, n_channels, ctx_len)
        """
        t, h = time_steps
        emb = torch.cat([self.time_emb(t), self.time_emb(h)], dim=-1)
        emb = self.time_mlp(emb)  # (B, emb_dim)

        # Encode context: (B, ctx_len, n_channels) -> (B, ctx_len, d_model)
        ctx = self.ctx_proj(context_with_lags.transpose(1, 2))
        for block in self.ctx_blocks:
            ctx = block(ctx, emb)

        # Context summary -> add to embedding
        ctx_pooled = self.ctx_pool(ctx.transpose(1, 2))  # (B, emb_dim)
        emb = emb + ctx_pooled

        # Cross-project context features to prediction length
        # ctx: (B, ctx_len, d_model) -> (B, pred_len, d_model)
        ctx_spatial = self.ctx_feat_proj(
            self.ctx_to_pred(ctx.transpose(1, 2)).transpose(1, 2)
        )

        # Prediction decoder
        pred = self.pred_proj(noisy_pred.unsqueeze(-1))  # (B, pred_len, d_model)
        pred = pred + ctx_spatial
        for block in self.pred_blocks:
            pred = block(pred, emb)

        out = self.out_proj(F.silu(self.out_norm(pred))).squeeze(-1)  # (B, pred_len)
        return out


# ============================================================
# Loss function (same MeanFlow JVP loss)
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


def s4d_meanflow_loss(net, future_clean, context_with_lags, norm_p=0.75, norm_eps=1e-3):
    """MeanFlow JVP self-consistency loss for S4D model."""
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


# ============================================================
# Forecaster for GluonTS evaluation
# ============================================================

class S4DMeanFlowForecaster(nn.Module):
    """GluonTS-compatible forecaster with lag features and RevIN."""

    def __init__(self, net, context_length, prediction_length, num_samples=100,
                 freq="H"):
        super().__init__()
        self.net = net
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.freq = freq
        self.revin = RevIN()

    def forward(self, past_target, past_observed_values, **kwargs):
        device = past_target.device
        B = past_target.shape[0]
        context = past_target[:, -self.context_length:]

        # RevIN normalization
        ctx_normed, mean, std = self.revin(context)

        # Extract lag features (all normalized by same stats)
        lags = extract_lags(past_target, self.context_length, self.freq)
        # Normalize lags with context stats
        lags_normed = (lags - mean.unsqueeze(1)) / std.unsqueeze(1)

        all_preds = []
        for _ in range(self.num_samples):
            z_1 = torch.randn(B, self.prediction_length, device=device)
            t = torch.ones(B, device=device)
            h = torch.ones(B, device=device)
            u = self.net(z_1, (t, h), lags_normed)
            pred_normed = z_1 - u
            # RevIN inverse
            pred = self.revin.inverse(pred_normed, mean, std)
            all_preds.append(pred)

        return torch.stack(all_preds, dim=1)


# ============================================================
# Conditioned version with extremity adapter
# ============================================================

class ExtremityAdapterV3(nn.Module):
    """Zero-initialized adapter for extremity conditioning."""
    def __init__(self, emb_dim):
        super().__init__()
        out = nn.Linear(64, emb_dim)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        self.mlp = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), out)

    def forward(self, ext_q):
        return self.mlp(ext_q.unsqueeze(-1))


class S4DConditionedNet(nn.Module):
    """S4D model with extremity conditioning adapter."""

    def __init__(self, base_net, cfg_drop_prob=0.2):
        super().__init__()
        self.base = base_net
        emb_dim = base_net.time_mlp[-1].out_features
        self.adapter = ExtremityAdapterV3(emb_dim)
        self.cfg_drop_prob = cfg_drop_prob
        self.pred_len = base_net.pred_len
        self.ctx_len = base_net.ctx_len

    def forward(self, noisy_pred, time_steps, context_with_lags, extremity_q=None):
        t, h = time_steps
        emb = torch.cat([self.base.time_emb(t), self.base.time_emb(h)], dim=-1)
        emb = self.base.time_mlp(emb)

        if extremity_q is not None:
            if self.training and self.cfg_drop_prob > 0:
                drop = torch.rand(extremity_q.shape[0], device=extremity_q.device) < self.cfg_drop_prob
                extremity_q = torch.where(drop, torch.full_like(extremity_q, 0.5), extremity_q)
            emb = emb + self.adapter(extremity_q)

        ctx = self.base.ctx_proj(context_with_lags.transpose(1, 2))
        for block in self.base.ctx_blocks:
            ctx = block(ctx, emb)
        emb = emb + self.base.ctx_pool(ctx.transpose(1, 2))
        ctx_spatial = self.base.ctx_feat_proj(
            self.base.ctx_to_pred(ctx.transpose(1, 2)).transpose(1, 2)
        )

        pred = self.base.pred_proj(noisy_pred.unsqueeze(-1)) + ctx_spatial
        for block in self.base.pred_blocks:
            pred = block(pred, emb)

        return self.base.out_proj(F.silu(self.base.out_norm(pred))).squeeze(-1)
