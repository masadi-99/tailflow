"""
Train TSFlow (Kollovieh et al. ICLR 2025) from their official code on one
of our benchmark datasets, then evaluate with our tail_metrics code so the
numbers are directly comparable to TailFlow's.

This is a thin wrapper around `tsflow.model.TSFlowCond` that:
  - reproduces the canonical `configs_local/train_conditional.yaml` config
    (with per-dataset ctx/pred adjustments),
  - trains with PyTorch Lightning (no aim/seml logger),
  - at eval time dumps raw forecasts to a numpy archive, and
  - scores them with our `meanflow_ts.tail_metrics` for twCRPS, qloss, etc.

We deliberately keep the TSFlow architecture + hyperparameters as close to
their default as possible so reviewers can see we didn't cripple the
baseline.
"""
from __future__ import annotations
import os, sys, time, argparse, logging, json, tempfile, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore", category=FutureWarning)

import pytorch_lightning as pl
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.loader import TrainDataLoader
from gluonts.dataset.split import OffsetSplitter
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.itertools import Cached
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'third_party', 'TSFlow'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pykeops
pykeops.set_build_folder(tempfile.mkdtemp(prefix="pykeops_build_"))

from tsflow.model import TSFlowCond
from tsflow.utils import create_transforms
from tsflow.utils.util import create_splitter, ConcatDataset
from tsflow.utils.variables import get_season_length

from meanflow_ts.tail_metrics import compute_all_tail_metrics, compute_train_thresholds

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


DATASET_CONFIGS = {
    "solar_nips":                   {"freq": "H", "ctx": 24, "pred": 24},
    "electricity_nips":             {"freq": "H", "ctx": 24, "pred": 24},
    "traffic_nips":                 {"freq": "H", "ctx": 24, "pred": 24},
    "exchange_rate_nips":           {"freq": "B", "ctx": 30, "pred": 30},
    "kdd_cup_2018_without_missing": {"freq": "H", "ctx": 48, "pred": 48},
    "uber_tlc_hourly":              {"freq": "H", "ctx": 48, "pred": 24},
}


def default_model_params(freq: str, context_length: int, prediction_length: int):
    return {
        "backbone_params": {
            "input_dim": 1,
            "output_dim": 1,
            "step_emb": 64,
            "num_residual_blocks": 3,
            "residual_block": "s4",
            "hidden_dim": 64,
            "dropout": 0.0,
            "init_skip": False,
            "feature_skip": True,
        },
        "freq": freq,
        "normalization": "longmean",
        "context_length": context_length,
        "prediction_length": prediction_length,
        "use_ema": True,
        "use_lags": True,
        "num_steps": 32,
        "solver": "euler",
        "matching": "random",
        "device": "cuda:0",
        "optimizer_params": {"lr": 1e-3},
        "prior_params": {"kernel": "ou", "gamma": 1, "context_freqs": 14},
        "ema_params": {"beta": 0.9999, "update_after_step": 128, "update_every": 1},
    }


def train_and_evaluate(name: str, epochs: int, num_samples: int,
                        logdir: str, seed: int = 6432):
    cfg = DATASET_CONFIGS[name]
    freq, ctx_len, pred_len = cfg["freq"], cfg["ctx"], cfg["pred"]
    pl.seed_everything(seed)

    dataset = get_dataset(name)
    target_dim = 1

    model_params = default_model_params(freq, ctx_len, pred_len)
    model = TSFlowCond(
        setting="univariate",
        target_dim=target_dim,
        context_length=model_params["context_length"],
        prediction_length=model_params["prediction_length"],
        backbone_params=model_params["backbone_params"],
        prior_params=model_params["prior_params"],
        optimizer_params=model_params["optimizer_params"],
        ema_params=model_params["ema_params"],
        frequency=model_params["freq"],
        normalization=model_params["normalization"],
        use_lags=model_params["use_lags"],
        use_ema=model_params["use_ema"],
        num_steps=model_params["num_steps"],
        solver=model_params["solver"],
        matching=model_params["matching"],
    )
    model.to(model_params["device"])
    logger.info(f"TSFlowCond params: {sum(p.numel() for p in model.parameters()):,}")

    transformation = create_transforms(
        time_features=time_features_from_frequency_str(freq),
        prediction_length=pred_len,
        freq=get_season_length(freq),
        train_length=len(dataset.train),
    )
    train_splitter = create_splitter(
        past_length=max(ctx_len + max(model.lags_seq), model.prior_context_length),
        future_length=pred_len,
        mode="train",
    )
    transformed = transformation.apply(dataset.train, is_train=True)
    loader = TrainDataLoader(
        Cached(transformed), batch_size=64, stack_fn=batchify,
        transform=train_splitter, num_batches_per_epoch=128,
        shuffle_buffer_length=10000,
    )

    ckdir = os.path.join(logdir, name)
    os.makedirs(ckdir, exist_ok=True)
    trainer = pl.Trainer(
        accelerator="gpu", devices=[0],
        max_epochs=epochs,
        enable_progress_bar=False,
        logger=False,
        default_root_dir=ckdir,
        gradient_clip_val=0.5,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
    )
    t0 = time.time()
    trainer.fit(model, train_dataloaders=loader)
    logger.info(f"[{name}] training done in {time.time()-t0:.1f}s")

    # Evaluate
    test_transform = transformation.apply(dataset.test, is_train=False)
    test_splitter = create_splitter(
        past_length=max(ctx_len + max(model.lags_seq), model.prior_context_length),
        future_length=pred_len, mode="test",
    )
    predictor = model.get_predictor(
        test_splitter, batch_size=max(1, 1024 * 64 // num_samples),
        device=model_params["device"],
    )
    fi, ti = make_evaluation_predictions(
        dataset=test_transform, predictor=predictor, num_samples=num_samples,
    )
    forecasts = list(fi); tss = list(ti)

    # Standard gluonts metric
    gluon_metrics, _ = Evaluator(num_workers=0)(tss, forecasts)

    # Pack samples for our tail metrics (N, S, T) / (N, T)
    samples = np.stack([f.samples for f in forecasts], axis=0).astype(np.float32)
    targets = np.stack([ts.values[-pred_len:].flatten() for ts in tss], axis=0).astype(np.float32)
    train_thr = compute_train_thresholds(dataset.train, quantiles=(0.9, 0.95, 0.99))
    tail_m = compute_all_tail_metrics(samples, targets, train_thresholds=train_thr)
    tail_m["gluonts_mean_wQuantileLoss"] = float(gluon_metrics["mean_wQuantileLoss"])

    # Save checkpoint + samples
    torch.save(model.state_dict(), os.path.join(ckdir, "tsflow_final.pt"))
    np.savez_compressed(os.path.join(ckdir, "samples.npz"),
                        samples=samples, targets=targets)
    logger.info(f"[{name}] CRPS={tail_m['gluonts_mean_wQuantileLoss']:.4f}  "
                f"twCRPS95={tail_m['twCRPS_q095']:.4f}  "
                f"twCRPS99={tail_m['twCRPS_q099']:.4f}")
    return tail_m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["solar_nips", "electricity_nips",
                            "traffic_nips", "exchange_rate_nips"])
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--logdir", default="tsflow_runs")
    p.add_argument("--output", default="results_tsflow_baseline.json")
    args = p.parse_args()

    out = {"tsflow": {}}
    for name in args.datasets:
        logger.info(f"{'='*60}\n{name}\n{'='*60}")
        try:
            out["tsflow"][name] = train_and_evaluate(
                name, args.epochs, args.num_samples, args.logdir)
        except Exception as exc:
            logger.exception(f"failed on {name}: {exc}")
            out["tsflow"][name] = {"error": str(exc)}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, default=float)
    logger.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
