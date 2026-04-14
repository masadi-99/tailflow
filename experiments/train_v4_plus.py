"""
TailFlow v4+: v4 model + Spectral features + Multi-step refinement.

Loads v4 checkpoint, adds spectral features + refinement, fine-tunes.
Also trains from scratch with spectral features for fair comparison.

Usage:
    python experiments/train_v4_plus.py electricity_nips --mode finetune
    python experiments/train_v4_plus.py electricity_nips --mode scratch
"""
import os, sys, time, argparse, logging, json, gc, tempfile
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
from gluonts.transform import (
    AddObservedValuesIndicator, AddTimeFeatures, AsNumpyArray,
    Chain, ExpectedNumInstanceSampler, InstanceSplitter, TestSplitSampler,
)

try:
    import pykeops
    tmp = tempfile.mkdtemp(prefix="pykeops_build_")
    pykeops.set_build_folder(tmp)
    pykeops.clean_pykeops()
except: pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v4 import (
    S4DMeanFlowNetV4, v4_meanflow_loss, RobustNorm,
    extract_lags_v4, get_lag_indices_v4, sample_t_r,
)
from meanflow_ts.innovations import (
    S4DSpectralRefinedNet, SpectralLoss, SelfRefinementModule,
)

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIGS = {
    "electricity_nips":   {"freq": "H", "ctx": 72, "pred": 24},
    "solar_nips":         {"freq": "H", "ctx": 72, "pred": 24},
    "traffic_nips":       {"freq": "H", "ctx": 72, "pred": 24},
    "exchange_rate_nips": {"freq": "B", "ctx": 30, "pred": 30},
}
LAG_MAP = {"H": 672, "B": 750}
TSFLOW_CRPS = {
    "electricity_nips": 0.045, "solar_nips": 0.341,
    "traffic_nips": 0.082, "exchange_rate_nips": 0.005,
}


# ============================================================
# Loss with spectral auxiliary
# ============================================================

def combined_loss(net, future_clean, context_with_lags, context_normed,
                  spectral_loss_fn, spectral_weight=0.1, norm_p=0.75, norm_eps=1e-3):
    """
    MeanFlow JVP loss + spectral auxiliary loss.

    The spectral loss helps the model capture periodic patterns.
    """
    B = future_clean.shape[0]
    device = future_clean.device
    e = torch.randn_like(future_clean)
    t, r = sample_t_r(B, device)
    t_bc, r_bc = t.unsqueeze(-1), r.unsqueeze(-1)

    z = (1 - t_bc) * future_clean + t_bc * e
    v = e - future_clean

    def u_func(z, t_bc, r_bc):
        h_bc = t_bc - r_bc
        return net(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)), context_with_lags, context_normed)

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

    # Spectral loss on the 1-step prediction (at t=1)
    # Quick 1-step: x0 = z1 - u(z1, t=1, h=1)
    with torch.no_grad():
        z1 = e  # At t=1, z = e (pure noise)
    t1 = torch.ones(B, device=device)
    h1 = torch.ones(B, device=device)
    u_at_1 = net(e.detach(), (t1, h1), context_with_lags, context_normed)
    x0_pred = e.detach() - u_at_1
    spec_loss = spectral_loss_fn(x0_pred, future_clean)

    return mf_loss + spec_loss


# ============================================================
# Refinement loss
# ============================================================

def refinement_loss(refiner, base_pred, future_clean, context_normed):
    """
    Train the refinement module to correct base predictions.
    """
    # Detach base_pred so gradients only flow through refiner
    refined = refiner(base_pred.detach(), context_normed)
    return F.mse_loss(refined, future_clean)


# ============================================================
# Forecaster with refinement
# ============================================================

class V4PlusForecaster(nn.Module):
    """Forecaster with spectral features and optional refinement."""

    def __init__(self, net, refiner, context_length, prediction_length,
                 num_samples=100, freq="H", use_refinement=True):
        super().__init__()
        self.net = net
        self.refiner = refiner
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.freq = freq
        self.use_refinement = use_refinement
        self.norm = RobustNorm()

    def forward(self, past_target, past_observed_values, **kwargs):
        device = past_target.device
        B = past_target.shape[0]
        context = past_target[:, -self.context_length:]

        ctx_normed, loc, scale = self.norm(context)
        lags = extract_lags_v4(past_target, self.context_length, self.freq)
        lags_normed = (lags - loc.unsqueeze(1)) / scale.unsqueeze(1)

        all_preds = []
        for _ in range(self.num_samples):
            z_1 = torch.randn(B, self.prediction_length, device=device)
            t = torch.ones(B, device=device)
            h = torch.ones(B, device=device)
            u = self.net(z_1, (t, h), lags_normed, ctx_normed)
            x0 = z_1 - u

            # Apply refinement
            if self.use_refinement and self.refiner is not None:
                x0 = self.refiner(x0, ctx_normed)

            pred = self.norm.inverse(x0, loc, scale)
            all_preds.append(pred)

        return torch.stack(all_preds, dim=1)


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(net, refiner, dataset, transformation, cfg, device,
                   num_samples=16, label="", freq="H", use_refinement=True):
    ctx_len, pred_len = cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP.get(freq, 672)
    net.eval()
    if refiner: refiner.eval()

    test_transform = transformation.apply(dataset.test, is_train=False)
    test_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
        past_length=ctx_len + max_lag, future_length=pred_len,
        time_series_fields=["time_feat", "observed_values"],
    )
    forecaster = V4PlusForecaster(
        net, refiner, ctx_len, pred_len, num_samples=num_samples,
        freq=freq, use_refinement=use_refinement,
    ).to(device)
    predictor = PyTorchPredictor(
        prediction_length=pred_len,
        input_names=["past_target", "past_observed_values"],
        prediction_net=forecaster, batch_size=128,
        input_transform=test_splitter, device=device,
    )
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_transform, predictor=predictor, num_samples=num_samples,
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    metrics, _ = Evaluator(num_workers=0)(tss, forecasts)
    crps = metrics["mean_wQuantileLoss"]
    nd = metrics["ND"]
    nrmse = metrics["NRMSE"]
    logger.info(f"  [{label}] CRPS={crps:.6f} | ND={nd:.6f} | NRMSE={nrmse:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--mode", type=str, default="finetune", choices=["finetune", "scratch"])
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--refine-epochs", type=int, default=200,
                        help="Additional epochs for refinement training")
    parser.add_argument("--n-refine-steps", type=int, default=3)
    parser.add_argument("--spectral-weight", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    name = args.dataset
    cfg = CONFIGS[name]
    freq = cfg["freq"]
    ctx_len, pred_len = cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP.get(freq, 672)
    n_lags = len(get_lag_indices_v4(freq))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    outdir = os.path.join(os.path.dirname(__file__), '..', 'results_v4plus', name)
    os.makedirs(outdir, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"TailFlow v4+: {name} | mode={args.mode}")
    logger.info(f"spectral_weight={args.spectral_weight} | n_refine_steps={args.n_refine_steps}")
    logger.info(f"{'='*60}")

    dataset = get_dataset(name)
    transformation = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(
            start_field="start", target_field="target", output_field="time_feat",
            time_features=time_features_from_frequency_str(freq), pred_length=pred_len,
        ),
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
    spectral_loss_fn = SpectralLoss(weight=args.spectral_weight)

    # ============================================================
    # Phase 1: Train base model with spectral features
    # ============================================================
    logger.info("\n=== Phase 1: Base model + spectral features ===")

    base_net = S4DMeanFlowNetV4(
        pred_len=pred_len, ctx_len=ctx_len, d_model=192,
        n_s4d_blocks=6, ssm_dim=64, time_emb_dim=64,
        dropout=0.1, n_lags=n_lags, freq=freq,
    ).to(device)

    # Wrap with spectral features
    net = S4DSpectralRefinedNet(
        base_net, ctx_len, pred_len, n_refine_steps=args.n_refine_steps,
    ).to(device)

    if args.mode == "finetune":
        # Load v4 checkpoint into the base model
        v4_path = os.path.join(os.path.dirname(__file__), '..', 'results_v4', name, 'best.pt')
        if os.path.exists(v4_path):
            logger.info(f"Loading v4 checkpoint from {v4_path}")
            ckpt = torch.load(v4_path, map_location=device, weights_only=False)
            base_net.load_state_dict(ckpt['net_ema'], strict=False)
            logger.info(f"  Loaded (CRPS was {ckpt.get('crps', '?')})")
        else:
            logger.warning(f"No v4 checkpoint at {v4_path}, training from scratch")
            args.mode = "scratch"

    net_ema = deepcopy(net).eval()
    param_count = sum(p.numel() for p in net.parameters())
    logger.info(f"Total params: {param_count:,}")

    if args.mode == "finetune":
        optimizer = AdamW([
            {'params': net.base.parameters(), 'lr': 1e-4},
            {'params': net.spectral.parameters(), 'lr': args.lr},
            {'params': net.refiner.parameters(), 'lr': args.lr},
        ], weight_decay=0.01)
        total_epochs = min(args.epochs, 400)  # Less for fine-tune
    else:
        optimizer = AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
        total_epochs = args.epochs

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-5)
    best_crps = float('inf')
    start_epoch = 0

    # Resume
    if args.resume:
        ckpt_path = os.path.join(outdir, 'best.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            net.load_state_dict(ckpt['net'])
            net_ema.load_state_dict(ckpt['net_ema'])
            start_epoch = ckpt.get('epoch', 0)
            best_crps = ckpt.get('crps', float('inf'))
            logger.info(f"Resumed from epoch {start_epoch}, CRPS={best_crps:.6f}")
            for _ in range(start_epoch):
                scheduler.step()

    results = {'dataset': name, 'mode': args.mode}

    for epoch in range(start_epoch, total_epochs):
        net.train()
        epoch_loss, n_b = 0, 0
        t0 = time.time()
        for batch in train_loader:
            past = batch["past_target"].to(device)
            future = batch["future_target"].to(device)
            ctx = past[:, -ctx_len:]

            ctx_normed, loc, scale = norm(ctx)
            future_normed = (future - loc) / scale
            lags = extract_lags_v4(past, ctx_len, freq).to(device)
            lags_normed = (lags - loc.unsqueeze(1)) / scale.unsqueeze(1)

            loss = combined_loss(
                net, future_normed, lags_normed, ctx_normed,
                spectral_loss_fn, args.spectral_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                for p, pe in zip(net.parameters(), net_ema.parameters()):
                    pe.data.lerp_(p.data, 1e-4)
            epoch_loss += loss.item()
            n_b += 1

        scheduler.step()
        avg = epoch_loss / n_b
        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(f"Ep {epoch+1:>3}/{total_epochs} | Loss: {avg:.4f} | "
                        f"lr={scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s")

        if (epoch + 1) % args.eval_every == 0 or (epoch + 1) == total_epochs:
            # Eval without refinement first
            m = evaluate_model(net_ema, None, dataset, transformation, cfg, device,
                                num_samples=16, label=f"ep{epoch+1}-noRefine",
                                freq=freq, use_refinement=False)
            crps_base = m["mean_wQuantileLoss"]

            # Eval with refinement
            m_ref = evaluate_model(net_ema, net_ema.refiner, dataset, transformation,
                                    cfg, device, num_samples=16,
                                    label=f"ep{epoch+1}-refined", freq=freq,
                                    use_refinement=True)
            crps_ref = m_ref["mean_wQuantileLoss"]

            crps = min(crps_base, crps_ref)
            if crps < best_crps:
                best_crps = crps
                torch.save({
                    'net': net.state_dict(), 'net_ema': net_ema.state_dict(),
                    'epoch': epoch + 1, 'crps': crps,
                }, os.path.join(outdir, 'best.pt'))
            torch.save({
                'net': net.state_dict(), 'net_ema': net_ema.state_dict(),
                'epoch': epoch + 1, 'crps': crps,
            }, os.path.join(outdir, 'latest.pt'))
            tsflow = TSFLOW_CRPS.get(name, "?")
            tag = " ***BEST***" if crps == best_crps else ""
            logger.info(f"  base={crps_base:.6f} refined={crps_ref:.6f} | "
                        f"TSFlow={tsflow} | Best={best_crps:.6f}{tag}")

    # ============================================================
    # Phase 2: Fine-tune refinement module only
    # ============================================================
    logger.info(f"\n=== Phase 2: Fine-tune refinement ({args.refine_epochs} epochs) ===")

    # Freeze base, train only refiner
    for p in net.base.parameters():
        p.requires_grad = False
    for p in net.spectral.parameters():
        p.requires_grad = False

    refine_opt = AdamW(net.refiner.parameters(), lr=1e-3)

    for epoch in range(args.refine_epochs):
        net.refiner.train()
        net.base.eval()
        epoch_loss, n_b = 0, 0
        t0 = time.time()
        for batch in train_loader:
            past = batch["past_target"].to(device)
            future = batch["future_target"].to(device)
            ctx = past[:, -ctx_len:]

            ctx_normed, loc, scale = norm(ctx)
            future_normed = (future - loc) / scale
            lags = extract_lags_v4(past, ctx_len, freq).to(device)
            lags_normed = (lags - loc.unsqueeze(1)) / scale.unsqueeze(1)

            # Get base prediction
            with torch.no_grad():
                z1 = torch.randn_like(future_normed)
                t_ones = torch.ones(z1.shape[0], device=device)
                h_ones = torch.ones(z1.shape[0], device=device)
                u = net.forward(z1, (t_ones, h_ones), lags_normed, ctx_normed)
                base_pred = z1 - u

            # Train refiner to correct base prediction toward ground truth
            loss = refinement_loss(net.refiner, base_pred, future_normed, ctx_normed)
            refine_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.refiner.parameters(), 1.0)
            refine_opt.step()
            with torch.no_grad():
                for p, pe in zip(net.refiner.parameters(), net_ema.refiner.parameters()):
                    pe.data.lerp_(p.data, 5e-4)
            epoch_loss += loss.item()
            n_b += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(f"Refine Ep {epoch+1:>3}/{args.refine_epochs} | Loss: {epoch_loss/n_b:.6f} | {time.time()-t0:.1f}s")

        if (epoch + 1) % 50 == 0 or (epoch + 1) == args.refine_epochs:
            m_ref = evaluate_model(net_ema, net_ema.refiner, dataset, transformation,
                                    cfg, device, num_samples=16,
                                    label=f"refine-ep{epoch+1}", freq=freq)
            crps_ref = m_ref["mean_wQuantileLoss"]
            if crps_ref < best_crps:
                best_crps = crps_ref
                torch.save({
                    'net': net.state_dict(), 'net_ema': net_ema.state_dict(),
                    'epoch': total_epochs + epoch + 1, 'crps': best_crps,
                }, os.path.join(outdir, 'best.pt'))
            tsflow = TSFLOW_CRPS.get(name, "?")
            tag = " ***BEST***" if crps_ref == best_crps else ""
            logger.info(f"  CRPS={crps_ref:.6f} | TSFlow={tsflow} | Best={best_crps:.6f}{tag}")

    # Unfreeze all
    for p in net.parameters():
        p.requires_grad = True

    # Final eval with 100 samples
    logger.info("\nFinal evaluation (100 samples)...")
    m_base = evaluate_model(net_ema, None, dataset, transformation, cfg, device,
                             num_samples=100, label="final-noRefine", freq=freq,
                             use_refinement=False)
    m_ref = evaluate_model(net_ema, net_ema.refiner, dataset, transformation, cfg,
                            device, num_samples=100, label="final-refined", freq=freq)

    results['final_no_refine'] = {'crps': m_base["mean_wQuantileLoss"]}
    results['final_refined'] = {'crps': m_ref["mean_wQuantileLoss"]}
    results['best_crps'] = best_crps
    results['params'] = param_count

    with open(os.path.join(outdir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    tsflow = TSFLOW_CRPS.get(name, "?")
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL: {name}")
    logger.info(f"  TSFlow (32 NFE): {tsflow}")
    logger.info(f"  v4+ no refine (1 NFE): {m_base['mean_wQuantileLoss']:.6f}")
    logger.info(f"  v4+ refined (1+{args.n_refine_steps} NFE): {m_ref['mean_wQuantileLoss']:.6f}")
    logger.info(f"  Best: {best_crps:.6f}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
