"""
Fine-tune the v4 base with noise scale sampled UNIFORMLY from [1, 3]
during training, instead of the fixed n=1. This directly tests the
"prior-tail reachability" claim: if at inference we now use n=1 and
reach the tail, the reachability argument is wrong and the base model
can be trained to cover the tail without inference-time widening.
Conversely, if n=1 inference still can't reach the tail after training
with a wider prior, the argument is supported.

Usage:
  python experiments/train_v4_wide_prior.py solar_nips --epochs 200 --n-min 1.0 --n-max 3.0
"""
from __future__ import annotations
import os, sys, time, argparse, logging
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from copy import deepcopy

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.loader import TrainDataLoader
from gluonts.itertools import Cached
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify
from gluonts.transform import (
    AddObservedValuesIndicator, AddTimeFeatures, AsNumpyArray,
    Chain, ExpectedNumInstanceSampler, InstanceSplitter,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v4 import (
    S4DMeanFlowNetV4, get_lag_indices_v4, extract_lags_v4, RobustNorm, sample_t_r,
)
from experiments.train_v4 import CONFIGS, LAG_MAP

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def wide_prior_meanflow_loss(net, future_clean, context_with_lags,
                              n_min=1.0, n_max=3.0,
                              norm_p=0.75, norm_eps=1e-3):
    """MeanFlow JVP loss where the starting noise e has variance n²I
    with n ~ U[n_min, n_max]. One n per sample."""
    B = future_clean.shape[0]
    device = future_clean.device
    n_sample = n_min + (n_max - n_min) * torch.rand(B, 1, device=device)
    e = torch.randn_like(future_clean) * n_sample

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-batches", type=int, default=128)
    p.add_argument("--n-min", type=float, default=1.0)
    p.add_argument("--n-max", type=float, default=3.0)
    args = p.parse_args()

    name = args.dataset
    cfg = CONFIGS[name]
    freq, ctx_len, pred_len = cfg["freq"], cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP[freq]
    n_lags = len(get_lag_indices_v4(freq))
    device = torch.device("cuda")
    torch.manual_seed(42); np.random.seed(42)

    outdir = os.path.join(os.path.dirname(__file__), "..", "results_v4_wide",
                          f"{name}_n{args.n_min}-{args.n_max}")
    os.makedirs(outdir, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"Wide-prior fine-tune: {name}  n~U[{args.n_min},{args.n_max}]")
    logger.info(f"{'='*60}")

    dataset = get_dataset(name)
    transformation = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(start_field="start", target_field="target", output_field="time_feat",
                        time_features=time_features_from_frequency_str(freq), pred_length=pred_len),
    ])
    splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start",
        instance_sampler=ExpectedNumInstanceSampler(num_instances=1, min_future=pred_len),
        past_length=ctx_len + max_lag, future_length=pred_len,
        time_series_fields=["time_feat", "observed_values"],
    )
    transformed = transformation.apply(dataset.train, is_train=True)
    loader = TrainDataLoader(
        Cached(transformed), batch_size=args.batch_size, stack_fn=batchify,
        transform=splitter, num_batches_per_epoch=args.num_batches,
        shuffle_buffer_length=2000,
    )

    net = S4DMeanFlowNetV4(
        pred_len=pred_len, ctx_len=ctx_len, d_model=192, n_s4d_blocks=6,
        ssm_dim=64, n_lags=n_lags, freq=freq,
    ).to(device)
    v4p = os.path.join(os.path.dirname(__file__), "..", "results_v4", name, "best.pt")
    ck = torch.load(v4p, map_location=device, weights_only=False)
    net.load_state_dict(ck["net_ema"])
    net_ema = deepcopy(net).eval()
    logger.info(f"Loaded v4 base (CRPS={ck.get('crps','?')})")

    opt = AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    norm = RobustNorm()

    for epoch in range(args.epochs):
        net.train()
        sum_loss, nb = 0.0, 0
        t0 = time.time()
        for batch in loader:
            past = batch["past_target"].to(device).float()
            fut = batch["future_target"].to(device).float()
            ctx = past[:, -ctx_len:]
            _, loc, scale = norm(ctx)
            fut_n = (fut - loc) / scale
            lags = extract_lags_v4(past, ctx_len, freq)
            lags_n = (lags - loc.unsqueeze(1)) / scale.unsqueeze(1)
            loss = wide_prior_meanflow_loss(
                net, fut_n, lags_n, n_min=args.n_min, n_max=args.n_max)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            with torch.no_grad():
                for p_, pe in zip(net.parameters(), net_ema.parameters()):
                    pe.data.lerp_(p_.data, 1e-4)
            sum_loss += loss.item(); nb += 1
        sched.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Ep {epoch+1}/{args.epochs}  loss={sum_loss/max(nb,1):.4f}  "
                        f"lr={opt.param_groups[0]['lr']:.2e}  {time.time()-t0:.1f}s")
        if (epoch + 1) % 20 == 0 or (epoch + 1) == args.epochs:
            torch.save({
                "net": net.state_dict(),
                "net_ema": net_ema.state_dict(),
                "epoch": epoch + 1,
                "loss": sum_loss / max(nb, 1),
                "n_min": args.n_min,
                "n_max": args.n_max,
            }, os.path.join(outdir, f"wide_prior_latest.pt"))


if __name__ == "__main__":
    main()
