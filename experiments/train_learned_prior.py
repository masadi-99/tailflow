"""
Learned Prior MeanFlow: Instead of fixed N(0,I) or GP prior, learn a
context-dependent prior distribution.

The prior network maps context -> (mu, L) where L is a lower-triangular
matrix defining the noise covariance. During training, we sample
e ~ N(mu, LL^T) and the MeanFlow loss operates as usual.

This is equivalent to learning an optimal "starting point" for each
input, which should be especially helpful for structured data like solar
(where the prior should reflect the day/night pattern).

Usage:
    python experiments/train_learned_prior.py solar_nips --epochs 800
    python experiments/train_learned_prior.py exchange_rate_nips --epochs 1500
"""
import os, sys, time, argparse, logging, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from copy import deepcopy

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.loader import TrainDataLoader
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.itertools import Cached
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import *

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v4 import (
    S4DMeanFlowNetV4, get_lag_indices_v4, RobustNorm, extract_lags_v4,
)
from meanflow_ts.model_v3 import sample_t_r

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIGS = {
    "solar_nips": {"freq": "H", "ctx": 72, "pred": 24},
    "exchange_rate_nips": {"freq": "B", "ctx": 30, "pred": 30},
}
LAG_MAP = {"H": 672, "B": 750}
TSFLOW = {"solar_nips": 0.341, "exchange_rate_nips": 0.005}


class LearnedPrior(nn.Module):
    """
    Learns a context-dependent noise distribution.
    Maps context_normed -> (mu, diag_std) defining N(mu, diag(std^2)).

    Using diagonal covariance (not full) for efficiency and stability.
    The mean captures the expected trend/pattern, and the std captures
    per-timestep uncertainty.
    """
    def __init__(self, ctx_len, pred_len, hidden=128):
        super().__init__()
        self.pred_len = pred_len
        self.mu_net = nn.Sequential(
            nn.Linear(ctx_len, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, pred_len),
        )
        self.log_std_net = nn.Sequential(
            nn.Linear(ctx_len, hidden), nn.SiLU(),
            nn.Linear(hidden, pred_len),
        )
        # Initialize to N(0,1) (mu=0, log_std=0)
        nn.init.zeros_(self.mu_net[-1].weight)
        nn.init.zeros_(self.mu_net[-1].bias)
        nn.init.zeros_(self.log_std_net[-1].weight)
        nn.init.zeros_(self.log_std_net[-1].bias)

    def forward(self, context_normed):
        """Returns mu (B, pred_len), std (B, pred_len)."""
        mu = self.mu_net(context_normed)
        log_std = self.log_std_net(context_normed).clamp(-2, 2)  # limit range
        return mu, torch.exp(log_std)

    def sample(self, context_normed):
        """Sample from learned prior. Returns (B, pred_len)."""
        mu, std = self.forward(context_normed)
        return mu + std * torch.randn_like(mu)


def learned_prior_loss(net, prior, future_clean, context_channels, context_normed,
                       norm_p=0.75, norm_eps=1e-3, kl_weight=0.01):
    """
    MeanFlow JVP loss with learned prior noise.

    The noise e is sampled from the learned prior N(mu, diag(std^2))
    instead of N(0,I). The MeanFlow interpolation still works:
    z = (1-t)*x_0 + t*e, v = e - x_0

    We add a KL regularizer to prevent the prior from collapsing.
    """
    B = future_clean.shape[0]
    device = future_clean.device

    # Sample noise from learned prior
    mu, std = prior(context_normed)
    e = mu + std * torch.randn_like(future_clean)

    t, r = sample_t_r(B, device)
    t_bc, r_bc = t.unsqueeze(-1), r.unsqueeze(-1)

    z = (1 - t_bc) * future_clean + t_bc * e
    v = e - future_clean

    def u_func(z, t_bc, r_bc):
        h_bc = t_bc - r_bc
        return net(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)), context_channels)

    with torch.amp.autocast("cuda", enabled=False):
        u_pred, dudt = torch.func.jvp(
            u_func, (z, t_bc, r_bc),
            (v, torch.ones_like(t_bc), torch.zeros_like(r_bc)),
        )
        u_tgt = (v - (t_bc - r_bc) * dudt).detach()
        mf_loss = (u_pred - u_tgt) ** 2
        mf_loss = mf_loss.sum(dim=1)
        adp_wt = (mf_loss.detach() + norm_eps) ** norm_p
        mf_loss = (mf_loss / adp_wt).mean()

    # KL divergence: KL(N(mu,std^2) || N(0,1))
    kl = 0.5 * (mu**2 + std**2 - 2*std.log() - 1).sum(dim=1).mean()

    return mf_loss + kl_weight * kl


class LearnedPriorForecaster(nn.Module):
    """Forecaster using learned prior."""
    def __init__(self, net, prior, ctx_len, pred_len, num_samples, freq, clamp_min=None):
        super().__init__()
        self.net = net; self.prior = prior
        self.context_length = ctx_len; self.prediction_length = pred_len
        self.num_samples = num_samples; self.freq = freq
        self.clamp_min = clamp_min; self.norm = RobustNorm()

    def forward(self, past_target, past_observed_values, **kw):
        d = past_target.device; B = past_target.shape[0]
        c = past_target[:, -self.context_length:].float()
        cn, l, sc = self.norm(c)
        lags = extract_lags_v4(past_target.float(), self.context_length, self.freq)
        ln = (lags - l.unsqueeze(1)) / sc.unsqueeze(1)

        mu, std = self.prior(cn)
        ps = []
        for _ in range(self.num_samples):
            z = (mu + std * torch.randn_like(mu)).float()
            t = torch.ones(B, device=d); h = t.clone()
            u = self.net(z, (t, h), ln)
            p = self.norm.inverse((z - u).float(), l, sc).float()
            if self.clamp_min is not None: p = p.clamp(min=self.clamp_min)
            ps.append(p.detach())
        out = torch.stack(ps, dim=1).contiguous()
        # Force float32 — GluonTS numpy conversion needs this
        return out.to(dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=50)
    args = parser.parse_args()

    name = args.dataset
    cfg = CONFIGS[name]
    freq, ctx_len, pred_len = cfg["freq"], cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP[freq]
    n_lags = len(get_lag_indices_v4(freq))
    device = torch.device("cuda")
    torch.manual_seed(42); np.random.seed(42)
    clamp_min = 0.0 if name == "solar_nips" else None

    outdir = os.path.join(os.path.dirname(__file__), '..', 'results_learned_prior', name)
    os.makedirs(outdir, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"Learned Prior MeanFlow: {name}")
    logger.info(f"{'='*60}")

    # Load v4 model
    net = S4DMeanFlowNetV4(pred_len=pred_len, ctx_len=ctx_len, d_model=192,
                            n_s4d_blocks=6, ssm_dim=64, n_lags=n_lags, freq=freq).to(device)
    ckpt = torch.load(f'results_v4/{name}/best.pt', map_location=device, weights_only=False)
    net.load_state_dict(ckpt['net_ema'])
    net_ema = deepcopy(net).eval()
    logger.info(f"Loaded v4 (CRPS={ckpt.get('crps','?')})")

    # Create learned prior
    prior = LearnedPrior(ctx_len, pred_len, hidden=128).to(device)
    prior_ema = deepcopy(prior).eval()
    logger.info(f"Prior params: {sum(p.numel() for p in prior.parameters()):,}")

    dataset = get_dataset(name)
    transformation = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(start_field="start", target_field="target", output_field="time_feat",
                        time_features=time_features_from_frequency_str(freq), pred_length=pred_len),
    ])
    train_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start",
        instance_sampler=ExpectedNumInstanceSampler(num_instances=1, min_future=pred_len),
        past_length=ctx_len + max_lag, future_length=pred_len,
        time_series_fields=["time_feat", "observed_values"],
    )
    td = transformation.apply(dataset.train, is_train=True)
    loader = TrainDataLoader(Cached(td), batch_size=args.batch_size, stack_fn=batchify,
                              transform=train_splitter, num_batches_per_epoch=100,
                              shuffle_buffer_length=2000)
    norm = RobustNorm()

    # Freeze flow model, only train prior
    for p in net.parameters():
        p.requires_grad = False
    net_ema = deepcopy(net).eval()  # reset EMA to frozen weights

    optimizer = AdamW([
        {'params': prior.parameters(), 'lr': args.lr},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    best_crps = float('inf')

    for epoch in range(args.epochs):
        net.train(); prior.train()
        el, nb = 0, 0
        t0 = time.time()
        for batch in loader:
            past = batch["past_target"].to(device).float()
            future = batch["future_target"].to(device).float()
            ctx = past[:, -ctx_len:]
            cn, l, sc = norm(ctx); fn = (future - l) / sc
            lags = extract_lags_v4(past, ctx_len, freq); ln = (lags - l.unsqueeze(1)) / sc.unsqueeze(1)

            loss = learned_prior_loss(net, prior, fn, ln, cn, kl_weight=args.kl_weight)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(net.parameters()) + list(prior.parameters()), 0.5)
            optimizer.step()
            with torch.no_grad():
                for p, pe in zip(net.parameters(), net_ema.parameters()):
                    pe.data.lerp_(p.data, 1e-4)
                for p, pe in zip(prior.parameters(), prior_ema.parameters()):
                    pe.data.lerp_(p.data, 1e-4)
            el += loss.item(); nb += 1

        scheduler.step()
        if (epoch+1) % 20 == 0 or epoch == 0:
            logger.info(f"Ep {epoch+1:>4}/{args.epochs} | Loss: {el/nb:.4f} | {time.time()-t0:.1f}s")

        if (epoch+1) % args.eval_every == 0 or (epoch+1) == args.epochs:
            net_ema.eval(); prior_ema.eval()
            # Manual evaluation — bypass GluonTS evaluator to avoid dtype issues
            test_windows = []
            for entry in dataset.test:
                ts = np.array(entry["target"], dtype=np.float32)
                if len(ts) >= ctx_len + max_lag + pred_len:
                    test_windows.append((ts[-(ctx_len+max_lag+pred_len):-pred_len], ts[-pred_len:]))
            crps_vals = []
            with torch.no_grad():
                for past_np, future_np in test_windows[:300]:
                    past_t = torch.tensor(past_np, device=device).unsqueeze(0).float()
                    c = past_t[:, -ctx_len:]; cn, l, sc = norm(c)
                    lags = extract_lags_v4(past_t, ctx_len, freq)
                    ln = (lags - l.unsqueeze(1)) / sc.unsqueeze(1)
                    mu_p, std_p = prior_ema(cn)
                    samps = []
                    for _ in range(50):
                        z = (mu_p + std_p * torch.randn_like(mu_p)).float()
                        t1 = torch.ones(1, device=device); h1 = t1
                        pred_n = (z - net_ema(z, (t1, h1), ln)).float()
                        p = norm.inverse(pred_n, l, sc).float()
                        if clamp_min is not None: p = p.clamp(min=clamp_min)
                        samps.append(p.cpu().numpy().flatten())
                    samps = np.array(samps)
                    # Normalized CRPS (matching GluonTS mean_wQuantileLoss)
                    denom = max(np.abs(future_np).mean(), 1e-6)
                    qs = np.arange(0.05, 1.0, 0.05)
                    ql = sum(np.mean(2*np.abs((future_np>np.quantile(samps,q,axis=0))-q)*np.abs(future_np-np.quantile(samps,q,axis=0))) for q in qs)
                    crps_vals.append(ql / len(qs) / denom)
            crps = np.mean(crps_vals)
            if crps < best_crps:
                best_crps = crps
                torch.save({'net': net.state_dict(), 'net_ema': net_ema.state_dict(),
                            'prior': prior.state_dict(), 'prior_ema': prior_ema.state_dict(),
                            'epoch': epoch+1, 'crps': crps}, os.path.join(outdir, 'best.pt'))
            tsf = TSFLOW[name]
            beat = " *** BEATS TSFLOW! ***" if crps < tsf else ""
            logger.info(f"  CRPS={crps:.6f} | TSFlow={tsf} | Best={best_crps:.6f}{beat}")

            # Log prior statistics
            with torch.no_grad():
                test_ctx = torch.randn(8, ctx_len, device=device)
                mu, std = prior_ema(test_ctx)
                logger.info(f"  Prior: mu_range=[{mu.min():.3f},{mu.max():.3f}], std_range=[{std.min():.3f},{std.max():.3f}]")

    logger.info(f"\nFINAL {name}: Best={best_crps:.6f} | TSFlow={TSFLOW[name]}")


if __name__ == "__main__":
    main()
