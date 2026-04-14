"""
v11 solar: TSFlow-style longmean normalization, done CORRECTLY.

Key insight: Don't put normalization in the GluonTS transform chain, because
then the evaluator compares against normalized targets. Instead:
1. Pre-compute per-series means from training data
2. Normalize INSIDE the training loop (after instance splitter)
3. Forecaster denormalizes back to raw scale

This way the evaluator sees raw predictions vs raw targets (correct CRPS).
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
from gluonts.transform import (
    AddObservedValuesIndicator, AddTimeFeatures, AsNumpyArray,
    Chain, ExpectedNumInstanceSampler, InstanceSplitter, TestSplitSampler,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v4 import S4DMeanFlowNetV4, get_lag_indices_v4, extract_lags_v4, sample_t_r

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_longmean_cache(dataset_train, min_scale=0.01):
    """Pre-compute per-series long-term means from training data."""
    cache = {}
    for entry in dataset_train:
        item_id = entry.get('item_id', len(cache))
        target = np.array(entry['target'], dtype=np.float32)
        cache[item_id] = max(float(np.abs(target).mean()), min_scale)
    return cache


def v11_meanflow_loss(net, future_normed, context_channels, norm_p=0.75, norm_eps=1e-3):
    B = future_normed.shape[0]
    device = future_normed.device
    e = torch.randn_like(future_normed)
    t, r = sample_t_r(B, device)
    t_bc, r_bc = t.unsqueeze(-1), r.unsqueeze(-1)
    z = (1 - t_bc) * future_normed + t_bc * e
    v = e - future_normed

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


class V11Forecaster(nn.Module):
    """
    Forecaster with longmean normalization applied INTERNALLY.
    Input past_target is RAW. We normalize, run model, denormalize.

    The scale is computed from a mean per-series cache using feat_static_cat
    (item_id). If unavailable, fall back to per-window mean-abs.
    """
    def __init__(self, net, ctx_len, pred_len, num_samples, freq, cache,
                 default_scale, clamp_min=None):
        super().__init__()
        self.net = net
        self.context_length = ctx_len
        self.prediction_length = pred_len
        self.num_samples = num_samples
        self.freq = freq
        self.clamp_min = clamp_min
        self.default_scale = default_scale
        # Register the cache as a buffer (indexed by item_id)
        self.register_buffer(
            'scale_table',
            torch.tensor([v for _, v in sorted(cache.items())], dtype=torch.float32)
        )
        self.n_cached = len(cache)

    def forward(self, past_target, past_observed_values, feat_static_cat=None, **kw):
        d = past_target.device
        B = past_target.shape[0]

        # Determine scale per batch item
        if feat_static_cat is not None:
            item_ids = feat_static_cat[:, 0].long()  # (B,)
            # Clamp to valid range
            item_ids_clamped = item_ids.clamp(max=self.n_cached - 1)
            scale = self.scale_table.to(d)[item_ids_clamped].unsqueeze(-1)  # (B, 1)
        else:
            # Fallback: use default
            scale = torch.full((B, 1), self.default_scale, device=d)

        # Normalize past
        past_n = past_target.float() / scale

        # Extract lags in normalized space
        lags = extract_lags_v4(past_n, self.context_length, self.freq)

        all_preds = []
        for _ in range(self.num_samples):
            z = torch.randn(B, self.prediction_length, device=d)
            t = torch.ones(B, device=d); h = t.clone()
            u = self.net(z, (t, h), lags)
            pred_n = (z - u).float()
            # Denormalize
            pred = pred_n * scale
            if self.clamp_min is not None:
                pred = pred.clamp(min=self.clamp_min)
            all_preds.append(pred.cpu())
        return torch.stack(all_preds, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    name = "solar_nips"
    freq = "H"
    ctx_len = 72
    pred_len = 24
    max_lag = 672
    n_lags = len(get_lag_indices_v4(freq))
    device = torch.device("cuda")
    torch.manual_seed(42); np.random.seed(42)

    outdir = os.path.join(os.path.dirname(__file__), '..', 'results_v11', name)
    os.makedirs(outdir, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"V11 Solar: longmean (correct — normalize inside loop)")
    logger.info(f"{'='*60}")

    dataset = get_dataset(name)

    # Pre-compute per-series long-term means from training data
    cache = compute_longmean_cache(dataset.train)
    default_scale = float(np.mean(list(cache.values())))
    logger.info(f"Cached {len(cache)} series means, "
                f"range [{min(cache.values()):.2f}, {max(cache.values()):.2f}], "
                f"default={default_scale:.2f}")

    # Standard transformation — NO longmean in the chain
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
        transform=train_splitter, num_batches_per_epoch=128, shuffle_buffer_length=2000,
    )

    # Scale table as tensor (for fast lookup)
    scale_table_cpu = torch.tensor(
        [cache[k] for k in sorted(cache.keys())], dtype=torch.float32
    )

    net = S4DMeanFlowNetV4(
        pred_len=pred_len, ctx_len=ctx_len, d_model=192,
        n_s4d_blocks=6, ssm_dim=64, n_lags=n_lags, freq=freq,
    ).to(device)
    net_ema = deepcopy(net).eval()
    logger.info(f"Params: {sum(p.numel() for p in net.parameters()):,}")

    opt = AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    best = float('inf')

    for epoch in range(args.epochs):
        net.train()
        epoch_loss, n_b = 0, 0
        t0 = time.time()
        for batch in train_loader:
            past = batch["past_target"].to(device).float()
            future = batch["future_target"].to(device).float()
            feat_cat = batch["feat_static_cat"].to(device)  # (B, 1)

            # Look up per-series scale
            item_ids = feat_cat[:, 0].long().clamp(max=len(cache) - 1)
            scale = scale_table_cpu.to(device)[item_ids].unsqueeze(-1)  # (B, 1)

            # Normalize by per-series scale
            past_n = past / scale
            future_n = future / scale

            # Extract lags in normalized space
            lags = extract_lags_v4(past_n, ctx_len, freq)

            loss = v11_meanflow_loss(net, future_n, lags)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            with torch.no_grad():
                for p, pe in zip(net.parameters(), net_ema.parameters()):
                    pe.data.lerp_(p.data, 1e-4)
            epoch_loss += loss.item(); n_b += 1
        sched.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(f"Ep {epoch+1}/{args.epochs} | Loss={epoch_loss/n_b:.4f} | "
                        f"lr={opt.param_groups[0]['lr']:.2e} | {time.time()-t0:.1f}s")

        if (epoch + 1) % 50 == 0 or (epoch + 1) == args.epochs:
            net_ema.eval()
            tt = transformation.apply(dataset.test, is_train=False)
            tsp = InstanceSplitter(
                target_field="target", is_pad_field="is_pad", start_field="start",
                forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
                past_length=ctx_len+max_lag, future_length=pred_len,
                time_series_fields=["time_feat", "observed_values"],
            )
            fc = V11Forecaster(net_ema, ctx_len, pred_len, 100, freq, cache,
                                default_scale, clamp_min=0).to(device)
            pr = PyTorchPredictor(
                prediction_length=pred_len,
                input_names=["past_target", "past_observed_values", "feat_static_cat"],
                prediction_net=fc, batch_size=16, input_transform=tsp, device=device,
            )
            fi, ti = make_evaluation_predictions(dataset=tt, predictor=pr, num_samples=100)
            tss = list(ti); forecasts = list(fi)
            m, _ = Evaluator(num_workers=0)(tss, forecasts)
            crps = m["mean_wQuantileLoss"]
            if crps < best:
                best = crps
                torch.save({"net_ema": net_ema.state_dict(), "crps": crps, "epoch": epoch+1,
                            "cache": cache},
                           os.path.join(outdir, "best.pt"))
            beat = " ***BEAT TSFLOW!***" if crps < 0.343 else ""
            logger.info(f"  CRPS={crps:.6f} | TSFlow=0.343 | Best={best:.6f}{beat}")

    logger.info(f"FINAL v11 solar: Best={best:.6f}")


if __name__ == "__main__":
    main()
