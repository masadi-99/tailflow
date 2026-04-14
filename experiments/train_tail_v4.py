"""
Phase-2 extremity-conditioned fine-tune on top of a pre-trained v4 base.

Loads results_v4/<dataset>/best.pt as the Phase-1 initialization, wraps it
with a zero-init extremity adapter, and fine-tunes with CFG dropout using
the same RobustNorm + extended-lag pipeline as train_v4. No changes to the
context encoder, prediction decoder, or lag indices — only the adapter is
new, and only the adapter + base network weights are updated.

Key differences from the original train_tail_v2.py:
  - 2.2M-param S4D backbone instead of 500K ResBlock1D
  - Context length 72 (v4) instead of 24 (small backbone)
  - RobustNorm with extended lag features (get_lag_indices_v4)
  - Cosine LR schedule matching train_v4

Usage:
  python experiments/train_tail_v4.py solar_nips --epochs 200
"""
import os, sys, time, argparse, logging, pickle, json
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
from meanflow_ts.model_v4_tail import (
    ConditionedS4DMeanFlowNetV4, conditioned_v4_meanflow_loss,
)
from meanflow_ts.model_tail import compute_raw_extremity, QuantileMapper

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIGS = {
    "solar_nips":         {"freq": "H", "ctx": 72, "pred": 24, "d_model": 192, "n_blocks": 6},
    "electricity_nips":   {"freq": "H", "ctx": 72, "pred": 24, "d_model": 192, "n_blocks": 6},
    "traffic_nips":       {"freq": "H", "ctx": 72, "pred": 24, "d_model": 192, "n_blocks": 6},
    "exchange_rate_nips": {"freq": "B", "ctx": 30, "pred": 30, "d_model": 192, "n_blocks": 6},
}
LAG_MAP = {"H": 672, "B": 750}


def precompute_extremity_scores(dataset, ctx_len, pred_len):
    """Compute per-window extremity on training data for the quantile mapper."""
    logger.info("Precomputing extremity scores on training set...")
    futures = []
    for entry in dataset.train:
        ts = np.array(entry["target"], dtype=np.float32)
        stride = max(pred_len // 2, 1)
        for start in range(0, len(ts) - ctx_len - pred_len + 1, stride):
            ctx = ts[start:start + ctx_len]
            fut = ts[start + ctx_len:start + ctx_len + pred_len]
            loc = max(np.abs(ctx).mean(), 1e-6)
            futures.append(fut / loc)
    futures = np.array(futures, dtype=np.float32)
    scores = compute_raw_extremity(torch.tensor(futures)).numpy()
    logger.info(f"  {len(futures)} windows, extremity mean={scores.mean():.3f} std={scores.std():.3f}")
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=str)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--cfg-drop", type=float, default=0.2)
    p.add_argument("--freeze-base", action="store_true",
                   help="Freeze v4 base weights; only train adapter.")
    p.add_argument("--num-batches-per-epoch", type=int, default=128)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to v4 base checkpoint (defaults to results_v4/<ds>/best.pt)")
    args = p.parse_args()

    name = args.dataset
    cfg = CONFIGS[name]
    freq, ctx_len, pred_len = cfg["freq"], cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP[freq]
    device = torch.device("cuda")
    torch.manual_seed(42); np.random.seed(42)

    outdir = os.path.join(os.path.dirname(__file__), "..", "results_v4_tail", name)
    os.makedirs(outdir, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"train_tail_v4: {name}  ctx={ctx_len} pred={pred_len} d={cfg['d_model']}")
    logger.info(f"  epochs={args.epochs} lr={args.lr} freeze_base={args.freeze_base}")
    logger.info(f"{'='*60}")

    dataset = get_dataset(name)
    norm = RobustNorm()

    # Extremity pipeline
    scores = precompute_extremity_scores(dataset, ctx_len, pred_len)
    qmap = QuantileMapper(n_bins=10)
    qmap.fit(scores)
    with open(os.path.join(outdir, "quantile_mapper.pkl"), "wb") as f:
        pickle.dump(qmap, f)

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
    transformed = transformation.apply(dataset.train, is_train=True)
    train_loader = TrainDataLoader(
        Cached(transformed), batch_size=args.batch_size, stack_fn=batchify,
        transform=train_splitter,
        num_batches_per_epoch=args.num_batches_per_epoch,
        shuffle_buffer_length=2000,
    )

    n_lags = len(get_lag_indices_v4(freq))
    base_net = S4DMeanFlowNetV4(
        pred_len=pred_len, ctx_len=ctx_len,
        d_model=cfg["d_model"], n_s4d_blocks=cfg["n_blocks"],
        ssm_dim=64, n_lags=n_lags, freq=freq,
    ).to(device)

    resume_path = args.resume or os.path.join(
        os.path.dirname(__file__), "..", "results_v4", name, "best.pt")
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"v4 base checkpoint not found: {resume_path}")
    ck = torch.load(resume_path, map_location=device, weights_only=False)
    base_state = ck.get("net_ema", ck.get("net"))
    base_net.load_state_dict(base_state)
    logger.info(f"Loaded Phase-1 base from {resume_path}  (reported CRPS={ck.get('crps','?')})")

    cond_net = ConditionedS4DMeanFlowNetV4(base_net, cfg_drop_prob=args.cfg_drop).to(device)
    if args.freeze_base:
        for p_ in cond_net.base.parameters():
            p_.requires_grad = False
        logger.info("  base frozen; training adapter only")

    cond_ema = deepcopy(cond_net).eval()
    trainable = [p for p in cond_net.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    logger.info(f"Trainable params: {n_train:,}")

    opt = AdamW(trainable, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        cond_net.train()
        ep_loss, nb = 0.0, 0
        t0 = time.time()
        for batch in train_loader:
            past = batch["past_target"].to(device).float()
            fut = batch["future_target"].to(device).float()
            ctx = past[:, -ctx_len:]
            ctx_n, loc, scale = norm(ctx)
            fut_n = (fut - loc) / scale

            lags = extract_lags_v4(past, ctx_len, freq)
            lags_n = (lags - loc.unsqueeze(1)) / scale.unsqueeze(1)

            # Per-window extremity on normalized futures (same pipeline as train)
            raw_scores = compute_raw_extremity(fut_n).cpu().numpy()
            q_vals = qmap.to_quantile(raw_scores)
            ext_q = torch.tensor(q_vals, device=device)

            loss = conditioned_v4_meanflow_loss(cond_net, fut_n, lags_n, ext_q)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 0.5)
            opt.step()
            with torch.no_grad():
                for p, pe in zip(cond_net.parameters(), cond_ema.parameters()):
                    pe.data.lerp_(p.data, 1e-4)
            ep_loss += loss.item()
            nb += 1
        sched.step()
        avg = ep_loss / max(nb, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Ep {epoch+1}/{args.epochs}  loss={avg:.4f}  "
                        f"lr={opt.param_groups[0]['lr']:.2e}  {time.time()-t0:.1f}s")
        if avg < best_loss:
            best_loss = avg
            torch.save({
                "cond_net": cond_net.state_dict(),
                "cond_ema": cond_ema.state_dict(),
                "epoch": epoch + 1,
                "loss": avg,
                "config": cfg,
            }, os.path.join(outdir, "phase2_v4_best.pt"))

    logger.info(f"DONE {name}: best_loss={best_loss:.4f}")


if __name__ == "__main__":
    main()
