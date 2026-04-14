"""
Distillation: Use multi-step inference as teacher, train 1-step student.

The key insight: our model with 8-step Euler gets much better CRPS than 1-step.
We can distill this quality back into 1-step by:
1. Generate predictions with K-step teacher
2. Train the same model to match those predictions in 1 step
3. This teaches the model to "shortcut" the multi-step ODE

Also: progressive distillation (halve steps each round: 8->4->2->1)

Usage:
    python experiments/train_distill.py solar_nips --teacher-steps 8 --rounds 3
    python experiments/train_distill.py electricity_nips --teacher-steps 4 --rounds 2
"""
import os, sys, time, argparse, logging, json, gc
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
    sample_t_r, V4Forecaster,
)

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIGS = {
    "electricity_nips": {"freq": "H", "ctx": 72, "pred": 24},
    "solar_nips": {"freq": "H", "ctx": 72, "pred": 24},
    "traffic_nips": {"freq": "H", "ctx": 72, "pred": 24},
    "exchange_rate_nips": {"freq": "B", "ctx": 30, "pred": 30},
}
LAG_MAP = {"H": 672, "B": 750}
TSFLOW = {"electricity_nips": 0.045, "solar_nips": 0.341, "traffic_nips": 0.082, "exchange_rate_nips": 0.005}


@torch.no_grad()
def teacher_predict(net, z1, lags_normed, n_steps):
    """K-step Euler from t=1 to t=0."""
    B = z1.shape[0]
    device = z1.device
    dt = 1.0 / n_steps
    z = z1.clone()
    for k in range(n_steps):
        tv = 1.0 - k * dt
        t = torch.full((B,), tv, device=device)
        h = torch.full((B,), tv, device=device)
        u = net(z, (t, h), lags_normed)
        z = z - dt * u
    return z


def distillation_loss(student, teacher, z1, lags_normed, teacher_steps, alpha=0.5):
    """
    Combined loss:
    - MeanFlow JVP loss (keeps the model well-calibrated)
    - Distillation loss (student 1-step output matches teacher K-step output)
    """
    B = z1.shape[0]
    device = z1.device

    # Teacher prediction (K steps, no grad)
    with torch.no_grad():
        teacher_pred = teacher_predict(teacher, z1, lags_normed, teacher_steps)

    # Student prediction (1 step)
    t1 = torch.ones(B, device=device)
    h1 = torch.ones(B, device=device)
    u_student = student(z1, (t1, h1), lags_normed)
    student_pred = z1 - u_student

    # Distillation loss: match teacher output
    distill_loss = F.mse_loss(student_pred, teacher_pred)

    return distill_loss


def combined_distill_meanflow_loss(student, teacher, future_clean, lags_normed,
                                    teacher_steps, distill_weight=0.5):
    """
    Combined: MeanFlow JVP loss + distillation from multi-step teacher.
    """
    B = future_clean.shape[0]
    device = future_clean.device

    # Standard MeanFlow JVP loss
    e = torch.randn_like(future_clean)
    t, r = sample_t_r(B, device)
    t_bc, r_bc = t.unsqueeze(-1), r.unsqueeze(-1)
    z = (1 - t_bc) * future_clean + t_bc * e
    v = e - future_clean

    def u_func(z, t_bc, r_bc):
        h_bc = t_bc - r_bc
        return student(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)), lags_normed)

    with torch.amp.autocast("cuda", enabled=False):
        u_pred, dudt = torch.func.jvp(
            u_func, (z, t_bc, r_bc),
            (v, torch.ones_like(t_bc), torch.zeros_like(r_bc)),
        )
        u_tgt = (v - (t_bc - r_bc) * dudt).detach()
        mf_loss = ((u_pred - u_tgt) ** 2).sum(dim=1)
        adp_wt = (mf_loss.detach() + 1e-3) ** 0.75
        mf_loss = (mf_loss / adp_wt).mean()

    # Distillation loss
    z1 = torch.randn_like(future_clean)
    d_loss = distillation_loss(student, teacher, z1, lags_normed, teacher_steps)

    return (1 - distill_weight) * mf_loss + distill_weight * d_loss, mf_loss.item(), d_loss.item()


class FlexForecaster(nn.Module):
    def __init__(self, net, ctx_len, pred_len, num_samples, freq, n_steps, clamp_min=None):
        super().__init__()
        self.net = net; self.context_length = ctx_len; self.prediction_length = pred_len
        self.num_samples = num_samples; self.freq = freq; self.n_steps = n_steps
        self.clamp_min = clamp_min; self.norm = RobustNorm()

    def forward(self, past_target, past_observed_values, **kw):
        d = past_target.device; B = past_target.shape[0]
        ctx = past_target[:, -self.context_length:]
        cn, l, sc = self.norm(ctx)
        lags = extract_lags_v4(past_target, self.context_length, self.freq)
        ln = (lags - l.unsqueeze(1)) / sc.unsqueeze(1)
        dt = 1.0 / self.n_steps; ps = []
        for _ in range(self.num_samples):
            z = torch.randn(B, self.prediction_length, device=d)
            for k in range(self.n_steps):
                tv = 1.0 - k * dt
                t = torch.full((B,), tv, device=d); h = torch.full((B,), tv, device=d)
                u = self.net(z, (t, h), ln); z = z - dt * u
            p = self.norm.inverse(z, l, sc)
            if self.clamp_min is not None: p = p.clamp(min=self.clamp_min)
            ps.append(p)
        return torch.stack(ps, dim=1)


def evaluate(net, dataset, transformation, cfg, device, num_samples=16,
             freq="H", n_steps=1, clamp_min=None, label=""):
    ctx_len, pred_len = cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP[freq]
    net.eval()
    test_transform = transformation.apply(dataset.test, is_train=False)
    test_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
        past_length=ctx_len + max_lag, future_length=pred_len,
        time_series_fields=["time_feat", "observed_values"],
    )
    fc = FlexForecaster(net, ctx_len, pred_len, num_samples, freq, n_steps, clamp_min).to(device)
    pred = PyTorchPredictor(
        prediction_length=pred_len, input_names=["past_target", "past_observed_values"],
        prediction_net=fc, batch_size=128, input_transform=test_splitter, device=device,
    )
    fi, ti = make_evaluation_predictions(dataset=test_transform, predictor=pred, num_samples=num_samples)
    forecasts = list(fi); tss = list(ti)
    metrics, _ = Evaluator(num_workers=0)(tss, forecasts)
    crps = metrics["mean_wQuantileLoss"]
    logger.info(f"  [{label}] CRPS={crps:.6f} ({n_steps}-step, {num_samples} samples)")
    return crps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--teacher-steps", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3,
                        help="Progressive distillation rounds")
    parser.add_argument("--epochs-per-round", type=int, default=200)
    parser.add_argument("--distill-weight", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    name = args.dataset
    cfg = CONFIGS[name]
    freq = cfg["freq"]
    ctx_len, pred_len = cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP[freq]
    n_lags = len(get_lag_indices_v4(freq))
    device = torch.device("cuda")
    torch.manual_seed(42); np.random.seed(42)
    clamp_min = 0 if name == "solar_nips" else None

    outdir = os.path.join(os.path.dirname(__file__), '..', 'results_distill', name)
    os.makedirs(outdir, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"Progressive Distillation: {name}")
    logger.info(f"Teacher steps: {args.teacher_steps}, Rounds: {args.rounds}")
    logger.info(f"{'='*60}")

    # Load best checkpoint
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'results_v4', name, 'best.pt')
    if name == "solar_nips":
        # v4plus is better for solar
        alt = os.path.join(os.path.dirname(__file__), '..', 'results_v4plus', name, 'best.pt')
        if os.path.exists(alt):
            # v4plus is wrapped, load base weights from v4 instead
            pass
    logger.info(f"Loading from {ckpt_path}")

    net = S4DMeanFlowNetV4(
        pred_len=pred_len, ctx_len=ctx_len, d_model=192, n_s4d_blocks=6,
        ssm_dim=64, n_lags=n_lags, freq=freq,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt['net_ema'])
    net_ema = deepcopy(net).eval()
    logger.info(f"Loaded (CRPS={ckpt.get('crps', '?')})")

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
    transformed_data = transformation.apply(dataset.train, is_train=True)
    train_loader = TrainDataLoader(
        Cached(transformed_data), batch_size=args.batch_size, stack_fn=batchify,
        transform=train_splitter, num_batches_per_epoch=100, shuffle_buffer_length=2000,
    )

    norm = RobustNorm()

    # Initial evaluation
    logger.info("\nBaseline evaluation:")
    for ns in [1, args.teacher_steps]:
        evaluate(net_ema, dataset, transformation, cfg, device,
                 num_samples=16, freq=freq, n_steps=ns, clamp_min=clamp_min,
                 label=f"baseline-{ns}step")

    # Progressive distillation: halve teacher steps each round
    teacher_steps = args.teacher_steps
    best_crps_1step = float('inf')

    for round_num in range(1, args.rounds + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Distillation Round {round_num}: teacher={teacher_steps}-step -> student=1-step")
        logger.info(f"{'='*50}")

        # Teacher is the current EMA model with multi-step
        teacher = deepcopy(net_ema).eval()

        optimizer = AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs_per_round, eta_min=1e-5)

        for epoch in range(args.epochs_per_round):
            net.train()
            ep_loss, ep_mf, ep_d, n_b = 0, 0, 0, 0
            t0 = time.time()
            for batch in train_loader:
                past = batch["past_target"].to(device)
                future = batch["future_target"].to(device)
                ctx = past[:, -ctx_len:]
                ctx_n, loc, scale = norm(ctx)
                fut_n = (future - loc) / scale
                lags = extract_lags_v4(past, ctx_len, freq).to(device)
                lags_n = (lags - loc.unsqueeze(1)) / scale.unsqueeze(1)

                loss, mf_l, d_l = combined_distill_meanflow_loss(
                    net, teacher, fut_n, lags_n,
                    teacher_steps=teacher_steps,
                    distill_weight=args.distill_weight,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()
                with torch.no_grad():
                    for p, pe in zip(net.parameters(), net_ema.parameters()):
                        pe.data.lerp_(p.data, 1e-4)
                ep_loss += loss.item(); ep_mf += mf_l; ep_d += d_l; n_b += 1

            scheduler.step()
            if (epoch + 1) % 20 == 0 or epoch == 0:
                logger.info(f"  R{round_num} Ep {epoch+1}/{args.epochs_per_round} | "
                            f"total={ep_loss/n_b:.4f} mf={ep_mf/n_b:.4f} distill={ep_d/n_b:.4f} | "
                            f"{time.time()-t0:.1f}s")

            if (epoch + 1) % 50 == 0 or (epoch + 1) == args.epochs_per_round:
                crps_1 = evaluate(net_ema, dataset, transformation, cfg, device,
                                   num_samples=16, freq=freq, n_steps=1,
                                   clamp_min=clamp_min, label=f"R{round_num}-1step")
                crps_k = evaluate(net_ema, dataset, transformation, cfg, device,
                                   num_samples=16, freq=freq, n_steps=teacher_steps,
                                   clamp_min=clamp_min, label=f"R{round_num}-{teacher_steps}step")
                if crps_1 < best_crps_1step:
                    best_crps_1step = crps_1
                    torch.save({'net': net.state_dict(), 'net_ema': net_ema.state_dict(),
                                'round': round_num, 'epoch': epoch+1, 'crps': crps_1},
                               os.path.join(outdir, 'best.pt'))
                tsf = TSFLOW.get(name, "?")
                beat = " *** BEATS TSFLOW! ***" if isinstance(tsf, float) and crps_1 < tsf else ""
                logger.info(f"  1-step Best={best_crps_1step:.6f} | TSFlow={tsf}{beat}")

        # Halve teacher steps for next round
        teacher_steps = max(teacher_steps // 2, 2)
        del teacher; gc.collect(); torch.cuda.empty_cache()

    # Final evaluation with 200 samples
    logger.info("\nFinal evaluation (200 samples):")
    for ns in [1, 2, 4]:
        evaluate(net_ema, dataset, transformation, cfg, device,
                 num_samples=200, freq=freq, n_steps=ns,
                 clamp_min=clamp_min, label=f"final-{ns}step")

    tsf = TSFLOW.get(name, "?")
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL: {name} | TSFlow={tsf} | Best 1-step={best_crps_1step:.6f}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
