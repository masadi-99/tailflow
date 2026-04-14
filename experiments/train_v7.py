"""
TailFlow v7: Targeted fixes for solar and exchange.

Solar: time features + LogScaleNorm + non-negative clamping
Exchange: adaptive noise scaling + time features + small model option

Usage:
    python experiments/train_v7.py solar_nips --epochs 1500
    python experiments/train_v7.py exchange_rate_nips --epochs 2000 --d-model 96 --n-blocks 4
"""
import os, sys, time, argparse, logging, json, tempfile
import numpy as np
import torch
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

try:
    import pykeops; tmp=tempfile.mkdtemp(); pykeops.set_build_folder(tmp); pykeops.clean_pykeops()
except: pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v7 import (
    S4DMeanFlowNetV7, v7_meanflow_loss, V7Forecaster,
    LogScaleNorm,
)
from meanflow_ts.model_v4 import extract_lags_v4, get_lag_indices_v4, RobustNorm

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIGS = {
    "solar_nips": {"freq": "H", "ctx": 72, "pred": 24, "norm": "robust",
                   "adaptive_noise": False, "clamp": 0},
    "exchange_rate_nips": {"freq": "B", "ctx": 30, "pred": 30, "norm": "robust",
                           "adaptive_noise": False, "clamp": None},
}
LAG_MAP = {"H": 672, "B": 750}
TSFLOW = {"solar_nips": 0.341, "exchange_rate_nips": 0.005}


def evaluate(net_ema, dataset, transformation, cfg, device, num_samples=16,
             freq="H", n_steps=1, label=""):
    ctx_len, pred_len = cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP[freq]
    net_ema.eval()
    tt = transformation.apply(dataset.test, is_train=False)
    sp = InstanceSplitter(target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
        past_length=ctx_len+max_lag, future_length=pred_len,
        time_series_fields=["time_feat", "observed_values"])

    input_names = ["past_target", "past_observed_values"]
    n_tf = len(time_features_from_frequency_str(freq))
    if n_tf > 0:
        input_names += ["past_time_feat", "future_time_feat"]

    fc = V7Forecaster(net_ema, ctx_len, pred_len, num_samples, freq, n_steps,
                       clamp_min=cfg.get("clamp"), norm_type=cfg.get("norm", "robust")).to(device)
    pr = PyTorchPredictor(prediction_length=pred_len, input_names=input_names,
        prediction_net=fc, batch_size=128, input_transform=sp, device=device)
    fi, ti = make_evaluation_predictions(dataset=tt, predictor=pr, num_samples=num_samples)
    forecasts = list(fi); tss = list(ti)
    metrics, _ = Evaluator(num_workers=0)(tss, forecasts)
    crps = metrics["mean_wQuantileLoss"]
    logger.info(f"  [{label}] CRPS={crps:.6f} ({n_steps}-step, {num_samples} samp)")
    return crps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    name = args.dataset
    cfg = CONFIGS[name]
    freq = cfg["freq"]
    ctx_len, pred_len = cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP[freq]
    n_lags = len(get_lag_indices_v4(freq))
    n_tf = len(time_features_from_frequency_str(freq))
    device = torch.device("cuda")
    torch.manual_seed(42); np.random.seed(42)

    outdir = os.path.join(os.path.dirname(__file__), '..', 'results_v7', name)
    os.makedirs(outdir, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"TailFlow v7: {name}")
    logger.info(f"ctx={ctx_len} pred={pred_len} d_model={args.d_model} blocks={args.n_blocks}")
    logger.info(f"adaptive_noise={cfg['adaptive_noise']} norm={cfg['norm']} clamp={cfg['clamp']}")
    logger.info(f"n_time_features={n_tf}")
    logger.info(f"{'='*60}")

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
        past_length=ctx_len+max_lag, future_length=pred_len,
        time_series_fields=["time_feat", "observed_values"],
    )
    td = transformation.apply(dataset.train, is_train=True)
    loader = TrainDataLoader(Cached(td), batch_size=args.batch_size, stack_fn=batchify,
        transform=train_splitter, num_batches_per_epoch=100, shuffle_buffer_length=2000)

    if cfg["norm"] == "log":
        norm = LogScaleNorm()
    else:
        norm = RobustNorm()

    net = S4DMeanFlowNetV7(
        pred_len=pred_len, ctx_len=ctx_len, d_model=args.d_model,
        n_s4d_blocks=args.n_blocks, ssm_dim=64 if args.d_model >= 128 else 32,
        time_emb_dim=64 if args.d_model >= 128 else 32,
        dropout=0.1 if args.d_model >= 128 else 0.2,
        n_lags=n_lags, freq=freq, n_time_features=n_tf,
        use_adaptive_noise=cfg["adaptive_noise"],
    ).to(device)
    net_ema = deepcopy(net).eval()
    logger.info(f"Params: {sum(p.numel() for p in net.parameters()):,}")

    # Optionally load v4 weights (skip adaptive noise and time feature layers)
    v4_path = os.path.join(os.path.dirname(__file__), '..', 'results_v4', name, 'best.pt')
    if os.path.exists(v4_path) and not args.resume:
        logger.info(f"Warm-starting from v4: {v4_path}")
        v4_ckpt = torch.load(v4_path, map_location=device, weights_only=False)
        # Load matching keys only
        v4_state = v4_ckpt['net_ema']
        my_state = net.state_dict()
        loaded = 0
        for k, v in v4_state.items():
            if k in my_state and my_state[k].shape == v.shape:
                my_state[k] = v
                loaded += 1
        net.load_state_dict(my_state)
        net_ema = deepcopy(net).eval()
        logger.info(f"  Loaded {loaded}/{len(v4_state)} weights from v4")

    optimizer = AdamW(net.parameters(), lr=args.lr,
                       weight_decay=0.05 if args.d_model < 128 else 0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)
    best_crps = float('inf')
    start_epoch = 0

    if args.resume:
        ckpt_path = os.path.join(outdir, 'best.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            net.load_state_dict(ckpt['net']); net_ema.load_state_dict(ckpt['net_ema'])
            start_epoch = ckpt.get('epoch', 0); best_crps = ckpt.get('crps', float('inf'))
            logger.info(f"Resumed from epoch {start_epoch}, CRPS={best_crps:.6f}")
            for _ in range(start_epoch): scheduler.step()

    for epoch in range(start_epoch, args.epochs):
        net.train()
        el, nb = 0, 0
        t0 = time.time()
        for batch in loader:
            past = batch["past_target"].to(device)
            future = batch["future_target"].to(device)
            ctx = past[:, -ctx_len:]
            ctx_n, loc, scale = norm(ctx)
            fut_n = (future - loc) / scale

            lags = extract_lags_v4(past, ctx_len, freq).to(device)
            lags_n = (lags - loc.unsqueeze(1)) / scale.unsqueeze(1)

            # Time features
            past_tf = batch.get("past_time_feat")
            future_tf = batch.get("future_time_feat")
            if past_tf is not None:
                past_tf = past_tf.to(device)
                ptf = past_tf.transpose(1,2) if past_tf.shape[2] < past_tf.shape[1] else past_tf
                tf_ctx = ptf[:, :, -ctx_len:]
                ctx_ch = torch.cat([lags_n, tf_ctx], dim=1)
            else:
                ctx_ch = lags_n

            ft_feat = None
            if future_tf is not None:
                future_tf = future_tf.to(device)
                ftf = future_tf.transpose(1,2) if future_tf.shape[2] < future_tf.shape[1] else future_tf
                ft_feat = ftf[:, :, :pred_len]

            loss = v7_meanflow_loss(net, fut_n, ctx_ch, ctx_n, ft_feat)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                for p, pe in zip(net.parameters(), net_ema.parameters()):
                    pe.data.lerp_(p.data, 1e-4)
            el += loss.item(); nb += 1

        scheduler.step()
        if (epoch+1) % 20 == 0 or epoch == 0:
            logger.info(f"Ep {epoch+1:>4}/{args.epochs} | Loss: {el/nb:.4f} | "
                        f"lr={scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s")

        if (epoch+1) % args.eval_every == 0 or (epoch+1) == args.epochs:
            crps = evaluate(net_ema, dataset, transformation, cfg, device,
                             num_samples=16, freq=freq, n_steps=1, label=f"ep{epoch+1}")
            if crps < best_crps:
                best_crps = crps
                torch.save({'net': net.state_dict(), 'net_ema': net_ema.state_dict(),
                            'epoch': epoch+1, 'crps': crps}, os.path.join(outdir, 'best.pt'))
            torch.save({'net': net.state_dict(), 'net_ema': net_ema.state_dict(),
                        'epoch': epoch+1, 'crps': crps}, os.path.join(outdir, 'latest.pt'))
            tsf = TSFLOW[name]
            beat = " *** BEATS TSFLOW! ***" if crps < tsf else ""
            logger.info(f"  CRPS={crps:.6f} | TSFlow={tsf} | Best={best_crps:.6f}{beat}")

    # Final
    logger.info("\nFinal eval (200 samples):")
    for ns in [1, 4]:
        evaluate(net_ema, dataset, transformation, cfg, device,
                  num_samples=200, freq=freq, n_steps=ns, label=f"final-{ns}step")

    logger.info(f"\nFINAL {name}: Best={best_crps:.6f} | TSFlow={TSFLOW[name]}")


if __name__ == "__main__":
    main()
