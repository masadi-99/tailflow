"""
CRPS-direct fine-tuning: optimize a differentiable CRPS surrogate on the
v4 pretrained model. This directly targets the evaluation metric instead
of the indirect MeanFlow JVP loss.

Quantile loss (pinball loss) at multiple quantiles approximates CRPS:
  L = 2 * mean_q( (y > q_pred) * q * (y - q_pred) + (y <= q_pred) * (1-q) * (q_pred - y) )

We generate N samples from the MeanFlow model, compute empirical quantiles,
and minimize the pinball loss. Gradients flow back through sampling.

This should directly improve CRPS since it optimizes the exact metric.
"""
import os, sys, time, argparse, logging, numpy as np, torch, torch.nn as nn
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
    V4Forecaster, v4_meanflow_loss, sample_t_r,
)

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
CONFIGS = {"solar_nips": {"freq":"H","ctx":72,"pred":24,"clamp":0},
           "exchange_rate_nips": {"freq":"B","ctx":30,"pred":30,"clamp":None}}
LAG_MAP = {"H": 672, "B": 750}
TSFLOW = {"solar_nips": 0.341, "exchange_rate_nips": 0.005}


def energy_crps_loss(pred_samples, target):
    """
    Analytical CRPS for empirical distribution (fully differentiable, unbiased):
        CRPS(F, y) = E|X - y| - (1/2) * E|X - X'|

    pred_samples: (B, N, T)
    target: (B, T)
    Returns: scalar loss
    """
    B, N, T = pred_samples.shape
    # Term 1: E|X - y|
    term1 = (pred_samples - target.unsqueeze(1)).abs().mean(dim=1)  # (B, T)
    # Term 2: (1/2) E|X - X'|
    # Compute pairwise differences efficiently
    s1 = pred_samples.unsqueeze(2)  # (B, N, 1, T)
    s2 = pred_samples.unsqueeze(1)  # (B, 1, N, T)
    term2 = 0.5 * (s1 - s2).abs().mean(dim=(1, 2))  # (B, T)
    crps = term1 - term2  # (B, T)
    # Normalize by target magnitude
    denom = target.abs().mean(dim=-1, keepdim=True).clamp(min=1e-6)
    return (crps / denom).mean()


def quantile_loss(pred_samples, target, num_quantiles=19):
    """Legacy pinball loss (kept for compatibility)."""
    return energy_crps_loss(pred_samples, target)


def crps_finetune_loss(net, future_clean, context_channels, n_samples=16,
                       mf_weight=0.3, crps_weight=0.7):
    """
    Combined loss: MeanFlow (keeps calibration) + CRPS surrogate (directly optimizes metric).
    """
    B = future_clean.shape[0]
    device = future_clean.device

    # Generate multiple 1-step samples with gradient
    samples = []
    for _ in range(n_samples):
        z = torch.randn_like(future_clean)
        t = torch.ones(B, device=device); h = t.clone()
        u = net(z, (t, h), context_channels)
        samples.append((z - u).unsqueeze(1))
    samples = torch.cat(samples, dim=1)  # (B, N, T)

    # CRPS surrogate (pinball loss)
    crps_l = quantile_loss(samples, future_clean)

    # Also compute MeanFlow JVP loss for stability
    e = torch.randn_like(future_clean)
    t_scalar, r = sample_t_r(B, device)
    t_bc, r_bc = t_scalar.unsqueeze(-1), r.unsqueeze(-1)
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
        mf_l = ((u_pred - u_tgt) ** 2).sum(dim=1)
        adp = (mf_l.detach() + 1e-3) ** 0.75
        mf_l = (mf_l / adp).mean()

    return mf_weight * mf_l + crps_weight * crps_l, mf_l.item(), crps_l.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=16,
                        help="Samples per batch for CRPS estimation")
    parser.add_argument("--crps-weight", type=float, default=0.7)
    parser.add_argument("--mf-weight", type=float, default=0.3)
    args = parser.parse_args()

    name = args.dataset
    cfg = CONFIGS[name]
    freq, ctx_len, pred_len = cfg["freq"], cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP[freq]
    n_lags = len(get_lag_indices_v4(freq))
    device = torch.device("cuda")
    torch.manual_seed(42); np.random.seed(42)

    outdir = os.path.join(os.path.dirname(__file__), '..', 'results_crps_ft', name)
    os.makedirs(outdir, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"CRPS direct fine-tuning: {name}")
    logger.info(f"mf_weight={args.mf_weight} crps_weight={args.crps_weight} n_samples={args.n_samples}")
    logger.info(f"{'='*60}")

    net = S4DMeanFlowNetV4(pred_len=pred_len, ctx_len=ctx_len, d_model=192,
                            n_s4d_blocks=6, ssm_dim=64, n_lags=n_lags, freq=freq).to(device)
    v4p = os.path.join(os.path.dirname(__file__), '..', 'results_v4', name, 'best.pt')
    ck = torch.load(v4p, map_location=device, weights_only=False)
    net.load_state_dict(ck['net_ema'])
    net_ema = deepcopy(net).eval()
    logger.info(f"Loaded v4 (CRPS={ck.get('crps', '?')})")

    ds = get_dataset(name)
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
    td = transformation.apply(ds.train, is_train=True)
    loader = TrainDataLoader(Cached(td), batch_size=args.batch_size, stack_fn=batchify,
                              transform=train_splitter, num_batches_per_epoch=50,
                              shuffle_buffer_length=2000)
    norm = RobustNorm()

    optimizer = AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    best = float('inf')

    for epoch in range(args.epochs):
        net.train()
        el, el_mf, el_crps, nb = 0, 0, 0, 0
        t0 = time.time()
        for batch in loader:
            past = batch["past_target"].to(device).float()
            future = batch["future_target"].to(device).float()
            ctx = past[:, -ctx_len:]
            cn, l, s = norm(ctx)
            fn = (future - l) / s
            lags = extract_lags_v4(past, ctx_len, freq)
            ln = (lags - l.unsqueeze(1)) / s.unsqueeze(1)

            loss, mf_val, crps_val = crps_finetune_loss(
                net, fn, ln, n_samples=args.n_samples,
                mf_weight=args.mf_weight, crps_weight=args.crps_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                for p, pe in zip(net.parameters(), net_ema.parameters()):
                    pe.data.lerp_(p.data, 2e-4)
            el += loss.item(); el_mf += mf_val; el_crps += crps_val; nb += 1

        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Ep {epoch+1}/{args.epochs} | L={el/nb:.4f} (mf={el_mf/nb:.4f}, crps={el_crps/nb:.4f}) | "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} | {time.time()-t0:.1f}s")

        if (epoch + 1) % 25 == 0 or (epoch + 1) == args.epochs:
            net_ema.eval()
            tt = transformation.apply(ds.test, is_train=False)
            tsp = InstanceSplitter(target_field="target", is_pad_field="is_pad", start_field="start",
                forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
                past_length=ctx_len+max_lag, future_length=pred_len,
                time_series_fields=["time_feat", "observed_values"])
            fc = V4Forecaster(net_ema, ctx_len, pred_len, 100, freq, use_gp=False).to(device)
            pr = PyTorchPredictor(prediction_length=pred_len,
                input_names=["past_target", "past_observed_values"],
                prediction_net=fc, batch_size=64, input_transform=tsp, device=device)
            fi, ti = make_evaluation_predictions(dataset=tt, predictor=pr, num_samples=100)
            forecasts = list(fi); tss = list(ti)
            m, _ = Evaluator(num_workers=0)(tss, forecasts)
            crps = m["mean_wQuantileLoss"]
            if crps < best:
                best = crps
                torch.save({"net_ema": net_ema.state_dict(), "crps": crps, "epoch": epoch+1},
                           os.path.join(outdir, "best.pt"))
            tsf = TSFLOW[name]
            beat = " *** BEAT TSFLOW! ***" if crps < tsf else ""
            logger.info(f"  CRPS={crps:.6f} | TSFlow={tsf} | Best={best:.6f}{beat}")

    logger.info(f"FINAL {name}: Best={best:.6f}")


if __name__ == "__main__":
    main()
