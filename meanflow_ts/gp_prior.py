"""
GP Prior methods compatible with MeanFlow.

The core problem: MeanFlow trains with the interpolation z = (1-t)*x_0 + t*e
where e ~ N(0,I). If we change e to GP noise, the JVP loss becomes unstable
because the correlated noise creates a harder velocity field.

Three approaches to make GP priors work with MeanFlow:

1. **Whitened GP**: Transform the GP prior into a whitened space where the noise
   IS N(0,I), but the data is transformed. Train MeanFlow in whitened space,
   then un-whiten at inference. This is mathematically equivalent to using a
   GP prior but compatible with i.i.d. noise training.

2. **Conditional GP inference only**: Train with N(0,I) noise normally, but at
   inference time start from GP posterior conditioned on context. This is a
   free lunch — the better starting point means the 1-step approximation is
   more accurate even though training used i.i.d. noise.

3. **GP-guided noise schedule**: Use a mixture of N(0,I) and GP noise during
   training, gradually increasing GP fraction. This lets the model adapt.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# 1. Whitened GP: Transform data into space where prior = N(0,I)
# ============================================================

class WhitenedGP:
    """
    Whitened GP prior via Cholesky decomposition.

    If K = L @ L^T is the GP covariance, then:
    - x_white = L^{-1} @ x  transforms data so that the GP prior becomes N(0,I)
    - x = L @ x_white  transforms back

    We train MeanFlow in the whitened space (where noise IS N(0,I)),
    then un-whiten predictions at inference.

    This is mathematically exact: the flow model learns in a space where
    the target distribution is simpler (whitened by the GP structure).
    """

    def __init__(self, kernel_type="ou", length_scale=None, pred_len=24,
                 ctx_len=72, device='cuda'):
        self.kernel_type = kernel_type
        self.pred_len = pred_len
        self.ctx_len = ctx_len
        self.device = device

        # Auto length scale based on dataset
        if length_scale is None:
            length_scale = pred_len / 8.0  # reasonable default

        self.length_scale = length_scale

        # Precompute Cholesky of prediction covariance
        K = self._kernel_matrix(pred_len, device)
        K = K + 1e-4 * torch.eye(pred_len, device=device)  # jitter
        self.L = torch.linalg.cholesky(K)  # (pred_len, pred_len)
        self.L_inv = torch.linalg.solve_triangular(
            self.L, torch.eye(pred_len, device=device), upper=False
        )

        # For conditional GP
        total = ctx_len + pred_len
        K_full = self._kernel_matrix(total, device)
        K_full = K_full + 1e-4 * torch.eye(total, device=device)
        self.K_oo = K_full[:ctx_len, :ctx_len]
        self.K_op = K_full[:ctx_len, ctx_len:]
        self.K_po = K_full[ctx_len:, :ctx_len]
        self.K_pp = K_full[ctx_len:, ctx_len:]

    def _kernel_matrix(self, n, device):
        t = torch.arange(n, device=device, dtype=torch.float32)
        diff = (t.unsqueeze(0) - t.unsqueeze(1)).abs()
        if self.kernel_type == "ou":
            return torch.exp(-diff / self.length_scale)
        elif self.kernel_type == "se":
            return torch.exp(-0.5 * (diff / self.length_scale) ** 2)
        elif self.kernel_type == "matern32":
            r = diff / self.length_scale * math.sqrt(3)
            return (1 + r) * torch.exp(-r)
        else:
            return torch.exp(-diff / self.length_scale)

    def whiten(self, x):
        """Transform data to whitened space. x: (B, pred_len)."""
        return x @ self.L_inv.T  # (B, pred_len)

    def unwhiten(self, x_white):
        """Transform from whitened space back to original. x_white: (B, pred_len)."""
        return x_white @ self.L.T  # (B, pred_len)

    def conditional_mean(self, context_normed):
        """
        Compute conditional GP mean for prediction given context.
        mu_pred = K_po @ K_oo^{-1} @ context

        context_normed: (B, ctx_len) — normalized context
        Returns: (B, pred_len) — conditional mean (float32)
        """
        # Solve in float64 for stability, then cast back to float32
        alpha = torch.linalg.solve(
            self.K_oo.double(), context_normed.double().T
        ).T.float()
        return (alpha @ self.K_op.float()).float()

    def _cond_cholesky(self):
        """Compute conditional covariance Cholesky (cached, float32)."""
        alpha = torch.linalg.solve(self.K_oo.double(), self.K_op.double()).float()
        K_cond = self.K_pp - self.K_po @ alpha
        K_cond = (K_cond + 1e-4 * torch.eye(self.pred_len, device=K_cond.device)).float()
        return torch.linalg.cholesky(K_cond).float()

    def conditional_sample(self, context_normed):
        """Sample from conditional GP posterior. Returns: (B, pred_len) float32."""
        mu = self.conditional_mean(context_normed)
        L_cond = self._cond_cholesky()
        z = torch.randn(context_normed.shape[0], self.pred_len, device=context_normed.device)
        return (mu + z @ L_cond.T).float()

    def conditional_whiten(self, x, context_normed):
        """Whiten in conditional GP space. Returns float32."""
        mu = self.conditional_mean(context_normed)
        L_cond = self._cond_cholesky()
        L_inv = torch.linalg.solve_triangular(
            L_cond, torch.eye(self.pred_len, device=x.device), upper=False
        ).float()
        return ((x.float() - mu) @ L_inv.T).float()

    def conditional_unwhiten(self, x_white, context_normed):
        """Unwhiten from conditional GP space. Returns float32."""
        mu = self.conditional_mean(context_normed)
        L_cond = self._cond_cholesky()
        return (x_white.float() @ L_cond.T + mu).float()


# ============================================================
# 2. Conditional GP Inference (free lunch)
# ============================================================

class ConditionalGPInference:
    """
    Use GP posterior at inference time only — training stays with N(0,I).

    At inference:
    1. Compute conditional GP mean from context: mu = K_po @ K_oo^{-1} @ ctx
    2. Sample noise: e ~ N(0, K_cond) where K_cond = K_pp - K_po @ K_oo^{-1} @ K_op
    3. Initialize: z_1 = mu + L_cond @ e_white  (instead of z_1 ~ N(0,I))
    4. One-step: x_0 = z_1 - u(z_1, t=1, h=1, context)

    The GP-initialized z_1 is already closer to the target distribution,
    so the 1-step MeanFlow correction needs to do less work.
    """

    def __init__(self, kernel_type="ou", length_scale=None, pred_len=24,
                 ctx_len=72, device='cuda'):
        self.gp = WhitenedGP(kernel_type, length_scale, pred_len, ctx_len, device)

    def sample_init(self, context_normed):
        """
        Get GP-conditioned initialization for inference.
        Returns: (B, pred_len) — better starting point than N(0,I)
        """
        return self.gp.conditional_sample(context_normed)

    def sample_init_blended(self, context_normed, blend=0.5):
        """
        Blend between N(0,I) and conditional GP.
        blend=0: pure N(0,I) (standard)
        blend=1: pure conditional GP
        blend=0.5: 50/50 mix

        This is a smooth interpolation that can be tuned.
        """
        B = context_normed.shape[0]
        device = context_normed.device
        z_iid = torch.randn(B, self.gp.pred_len, device=device)
        z_gp = self.gp.conditional_sample(context_normed)
        return (1 - blend) * z_iid + blend * z_gp


# ============================================================
# 3. Whitened-Space MeanFlow Training
# ============================================================

def whitened_meanflow_loss(net, future_clean, context_channels, gp, context_normed,
                           norm_p=0.75, norm_eps=1e-3):
    """
    MeanFlow JVP loss in the whitened GP space.

    1. Whiten the future: future_white = L_cond^{-1} @ (future - mu_cond)
    2. Train with standard N(0,I) noise in whitened space
    3. The model learns to transport N(0,I) -> whitened data distribution

    At inference: sample z ~ N(0,I), predict x_white = z - u(z), unwhiten.

    This is exact — no approximation — and fully compatible with MeanFlow's JVP loss.
    """
    from .model_v3 import sample_t_r

    B = future_clean.shape[0]
    device = future_clean.device

    # Whiten the future target using conditional GP
    future_white = gp.conditional_whiten(future_clean, context_normed)

    # Standard MeanFlow in whitened space (noise is N(0,I))
    e = torch.randn_like(future_white)
    t, r = sample_t_r(B, device)
    t_bc, r_bc = t.unsqueeze(-1), r.unsqueeze(-1)

    z = (1 - t_bc) * future_white + t_bc * e
    v = e - future_white

    def u_func(z, t_bc, r_bc):
        h_bc = t_bc - r_bc
        return net(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)), context_channels)

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
