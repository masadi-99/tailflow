"""
Extremity-conditioned wrapper on top of S4DMeanFlowNetV4.

This is a drop-in Phase-2 adapter: it loads a frozen (or fine-tuneable)
S4DMeanFlowNetV4 base and adds a zero-initialized MLP that maps an
extremity quantile q ∈ [0, 1] to an additive bias on the time embedding
before it's used by the context + prediction paths.

During training, extremity_q is replaced by the marginal value 0.5 with
probability `cfg_drop_prob`, implementing classifier-free-guidance dropout.
At inference, a guidance scale `w` blends u_cond and u_uncond via

    u_guided = u_uncond + w * (u_cond - u_uncond).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_v4 import S4DMeanFlowNetV4, sample_t_r


class ExtremityAdapterV4(nn.Module):
    """
    Zero-init MLP from q ∈ [0, 1] to a time-embedding bias of dim emb_dim.
    Identity at initialization so Phase-2 starts exactly at the Phase-1
    behavior.
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, emb_dim),
        )
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, ext_q: torch.Tensor) -> torch.Tensor:
        # ext_q: (B,) in [0, 1] -> (B, emb_dim)
        return self.mlp(ext_q.unsqueeze(-1))


class ConditionedS4DMeanFlowNetV4(nn.Module):
    """
    Wraps a S4DMeanFlowNetV4 with an extremity adapter. All attributes of
    the base net remain accessible (forward, gp, time_mlp, ctx_proj, …).

    Forward signature mirrors the base net but takes an optional
    `extremity_q` argument; when it is None, the model behaves exactly like
    the base net.
    """

    def __init__(self, base_net: S4DMeanFlowNetV4, cfg_drop_prob: float = 0.2):
        super().__init__()
        self.base = base_net
        emb_dim = base_net.time_mlp[-1].out_features
        self.adapter = ExtremityAdapterV4(emb_dim)
        self.cfg_drop_prob = cfg_drop_prob
        self.pred_len = base_net.pred_len
        self.ctx_len = base_net.ctx_len
        self.freq = base_net.freq
        self.n_lags = base_net.n_lags
        self.gp = base_net.gp

    def forward(self, noisy_pred, time_steps, context_with_lags,
                extremity_q=None):
        base = self.base
        t, h = time_steps
        emb = torch.cat([base.time_emb(t), base.time_emb(h)], dim=-1)
        emb = base.time_mlp(emb)

        if extremity_q is not None:
            if self.training and self.cfg_drop_prob > 0:
                drop = torch.rand(extremity_q.shape[0], device=extremity_q.device) < self.cfg_drop_prob
                extremity_q = torch.where(drop, torch.full_like(extremity_q, 0.5), extremity_q)
            emb = emb + self.adapter(extremity_q)

        ctx = base.ctx_proj(context_with_lags.transpose(1, 2))
        for block in base.ctx_blocks:
            ctx = block(ctx, emb)

        emb = emb + base.ctx_pool(ctx.transpose(1, 2))

        ctx_spatial = base.ctx_feat_proj(
            base.ctx_to_pred(ctx.transpose(1, 2)).transpose(1, 2)
        )

        pred = base.pred_proj(noisy_pred.unsqueeze(-1)) + ctx_spatial
        for block in base.pred_blocks:
            pred = block(pred, emb)

        return base.out_proj(F.silu(base.out_norm(pred))).squeeze(-1)


def conditioned_v4_meanflow_loss(net, future_clean, context_with_lags,
                                  extremity_q, norm_p=0.75, norm_eps=1e-3):
    """MeanFlow JVP loss with extremity conditioning. Same shape as v4_meanflow_loss."""
    B = future_clean.shape[0]
    device = future_clean.device
    e = torch.randn_like(future_clean)
    t, r = sample_t_r(B, device)
    t_bc, r_bc = t.unsqueeze(-1), r.unsqueeze(-1)

    z = (1 - t_bc) * future_clean + t_bc * e
    v = e - future_clean

    def u_func(z, t_bc, r_bc):
        h_bc = t_bc - r_bc
        return net(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)), context_with_lags, extremity_q)

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
def guided_sample_v4(net, context_with_lags, extremity_q, shape, device,
                     guidance_scale=1.0, noise_scale=1.0):
    """CFG-guided one-step sampling for the v4 backbone."""
    B = shape[0]
    z_1 = torch.randn(shape, device=device) * noise_scale
    t = torch.ones(B, device=device)
    h = torch.ones(B, device=device)

    if guidance_scale == 1.0:
        u = net(z_1, (t, h), context_with_lags, extremity_q)
        return z_1 - u

    u_cond = net(z_1, (t, h), context_with_lags, extremity_q)
    u_uncond = net(z_1, (t, h), context_with_lags,
                   torch.full_like(extremity_q, 0.5))
    u_guided = u_uncond + guidance_scale * (u_cond - u_uncond)
    return z_1 - u_guided
