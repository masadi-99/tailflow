"""
Methodological innovations for pushing beyond TSFlow.

1. Spectral features + frequency-domain auxiliary loss
2. Multi-step self-refinement (iterative correction)
3. Self-conditioning (feed model's own prediction back)

These are modular — can be composed with any base model.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Spectral Features + Frequency-Domain Loss
# ============================================================

class SpectralFeatureExtractor(nn.Module):
    """
    Extract spectral features from context using FFT.
    Captures periodicity (daily, weekly) that temporal models may miss.
    """
    def __init__(self, ctx_len, d_model, n_freq_bins=32):
        super().__init__()
        self.ctx_len = ctx_len
        self.n_freq_bins = n_freq_bins
        # Project spectral features to model dimension
        # FFT gives ctx_len//2+1 complex values -> 2*(ctx_len//2+1) real values
        # We take top n_freq_bins magnitudes + phases
        self.proj = nn.Linear(n_freq_bins * 2, d_model)
        # Zero-init so it starts as no-op when fine-tuning
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, context_normed):
        """
        context_normed: (B, ctx_len) — normalized context
        Returns: (B, d_model) — spectral feature vector
        """
        # FFT
        spec = torch.fft.rfft(context_normed, dim=-1)  # (B, ctx_len//2+1)
        mag = spec.abs()  # (B, ctx_len//2+1)
        phase = spec.angle()  # (B, ctx_len//2+1)

        # Take top-n_freq_bins by magnitude (per sample)
        n_freqs = mag.shape[-1]
        n_keep = min(self.n_freq_bins, n_freqs)

        # Just use the first n_keep frequency bins (low frequencies are most informative)
        mag_feat = mag[:, :n_keep]
        phase_feat = phase[:, :n_keep]

        # Pad if needed
        if n_keep < self.n_freq_bins:
            pad = self.n_freq_bins - n_keep
            mag_feat = F.pad(mag_feat, (0, pad))
            phase_feat = F.pad(phase_feat, (0, pad))

        # Concatenate and project
        feat = torch.cat([mag_feat, phase_feat], dim=-1)  # (B, n_freq_bins*2)
        return self.proj(feat)  # (B, d_model)


class SpectralLoss(nn.Module):
    """
    Frequency-domain auxiliary loss.
    Penalizes mismatch between predicted and target spectra.
    This helps the model capture periodic patterns (solar, traffic).
    """
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

    def forward(self, pred, target):
        """
        pred: (B, T) — predicted time series (normalized)
        target: (B, T) — ground truth (normalized)
        Returns: scalar loss
        """
        spec_pred = torch.fft.rfft(pred, dim=-1)
        spec_target = torch.fft.rfft(target, dim=-1)

        # Magnitude loss (L1 on log-magnitudes for scale invariance)
        mag_pred = spec_pred.abs().clamp(min=1e-8).log()
        mag_target = spec_target.abs().clamp(min=1e-8).log()
        mag_loss = F.l1_loss(mag_pred, mag_target)

        # Phase loss (cosine similarity of complex coefficients)
        # Normalized dot product in complex space
        dot = (spec_pred * spec_target.conj()).real
        norm_pred = spec_pred.abs().clamp(min=1e-8)
        norm_target = spec_target.abs().clamp(min=1e-8)
        phase_loss = 1 - (dot / (norm_pred * norm_target)).mean()

        return self.weight * (mag_loss + 0.5 * phase_loss)


# ============================================================
# 2. Multi-Step Self-Refinement
# ============================================================

class SelfRefinementModule(nn.Module):
    """
    After 1-step MeanFlow prediction, apply K correction steps.

    Each step: x_{k+1} = x_k - alpha_k * correction_net(x_k, context)

    The correction_net is a lightweight network that learns to fix
    errors in the initial prediction. This is conceptually similar to
    Langevin dynamics correction in score-based models.

    Key insight: MeanFlow's 1-step output is good but not perfect.
    A small learned corrector can fix systematic errors without
    requiring full ODE integration.
    """
    def __init__(self, pred_len, d_model=64, n_steps=2):
        super().__init__()
        self.n_steps = n_steps
        self.pred_len = pred_len

        # Lightweight correction network (shared across steps)
        self.correction = nn.Sequential(
            nn.Linear(pred_len, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, pred_len),
        )
        # Learnable step sizes (initialized to zero — starts as identity)
        self.alphas = nn.Parameter(torch.zeros(n_steps))

        # Context-dependent gating
        self.gate = nn.Sequential(
            nn.Linear(pred_len, n_steps),
            nn.Sigmoid(),
        )

    def forward(self, x0, context_normed):
        """
        x0: (B, pred_len) — initial MeanFlow prediction (normalized)
        context_normed: (B, ctx_len) — normalized context
        Returns: refined prediction (B, pred_len)
        """
        # Context-dependent gate weights
        gate = self.gate(context_normed[:, -self.pred_len:] if context_normed.shape[1] >= self.pred_len
                         else F.pad(context_normed, (0, self.pred_len - context_normed.shape[1])))

        x = x0
        for k in range(self.n_steps):
            correction = self.correction(x)
            alpha = self.alphas[k] * gate[:, k:k+1]  # (B, 1)
            x = x - alpha * correction

        return x


class SelfConditioningWrapper:
    """
    Self-conditioning: during training, sometimes feed the model's own
    (stopped-gradient) prediction as an additional input.

    This teaches the model to refine its own outputs, making the
    single-step prediction more accurate.

    Usage during training:
        1. 50% of the time: pass zeros as self-cond input (normal training)
        2. 50% of the time: run model once to get prediction, detach,
           and pass as self-cond input for the actual training step

    At inference: always run two forward passes (one to get self-cond,
    one for the final prediction).
    """

    @staticmethod
    def get_self_cond(net, z, time_steps, context, prob=0.5):
        """
        Get self-conditioning signal with probability `prob`.
        Returns: (B, pred_len) — self-conditioning signal or zeros
        """
        B, T = z.shape
        if torch.rand(1).item() < prob and net.training:
            with torch.no_grad():
                u_pred = net(z, time_steps, context)
                self_cond = (z - u_pred).detach()
            return self_cond
        else:
            return torch.zeros_like(z)


# ============================================================
# 3. Integrated Model: S4D + Spectral + Refinement
# ============================================================

class S4DSpectralRefinedNet(nn.Module):
    """
    Wraps any base S4D model with spectral features and self-refinement.
    """
    def __init__(self, base_net, ctx_len, pred_len, n_refine_steps=2,
                 spectral_freq_bins=32):
        super().__init__()
        self.base = base_net
        self.pred_len = pred_len
        self.ctx_len = ctx_len

        d_model = base_net.d_model
        emb_dim = base_net.time_mlp[-1].out_features

        # Spectral feature extractor
        self.spectral = SpectralFeatureExtractor(ctx_len, emb_dim, spectral_freq_bins)

        # Self-refinement
        self.refiner = SelfRefinementModule(pred_len, d_model=128, n_steps=n_refine_steps)

    def forward(self, noisy_pred, time_steps, context_with_lags, context_normed=None):
        """
        Same interface as base model, plus optional context_normed for spectral features.
        """
        # Get base prediction (velocity)
        t, h = time_steps
        emb = torch.cat([self.base.time_emb(t), self.base.time_emb(h)], dim=-1)
        emb = self.base.time_mlp(emb)

        # Add spectral features to embedding
        if context_normed is not None:
            spec_feat = self.spectral(context_normed)
            emb = emb + spec_feat

        # Context encoder
        ctx = self.base.ctx_proj(context_with_lags.transpose(1, 2))
        for block in self.base.ctx_blocks:
            ctx = block(ctx, emb)
        emb = emb + self.base.ctx_pool(ctx.transpose(1, 2))
        ctx_spatial = self.base.ctx_feat_proj(
            self.base.ctx_to_pred(ctx.transpose(1, 2)).transpose(1, 2)
        )

        # Prediction decoder
        pred = self.base.pred_proj(noisy_pred.unsqueeze(-1)) + ctx_spatial
        for block in self.base.pred_blocks:
            pred = block(pred, emb)

        velocity = self.base.out_proj(F.silu(self.base.out_norm(pred))).squeeze(-1)
        return velocity

    def sample_with_refinement(self, context_normed, context_with_lags, device):
        """
        Full inference: 1-step MeanFlow + multi-step refinement.
        Returns: refined prediction (normalized)
        """
        B = context_normed.shape[0]
        z_1 = torch.randn(B, self.pred_len, device=device)
        t = torch.ones(B, device=device)
        h = torch.ones(B, device=device)

        # 1-step MeanFlow
        u = self.forward(z_1, (t, h), context_with_lags, context_normed)
        x0 = z_1 - u

        # Refinement steps
        x_refined = self.refiner(x0, context_normed)
        return x_refined
