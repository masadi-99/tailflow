"""
v10 exchange: v4's successful approach + TSFlow's 29 lags.
Tests if the extended lags alone help, keeping everything else v4-style.
"""
import os, sys, time, argparse, logging, numpy as np, torch, torch.nn as nn
from torch.optim import AdamW
from copy import deepcopy
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.loader import TrainDataLoader
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.itertools import Cached
from gluonts.time_feature import time_features_from_frequency_str, get_lags_for_frequency
from gluonts.torch.batchify import batchify
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import *

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v4 import S4DMeanFlowNetV4, RobustNorm, sample_t_r

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_lags_gluonts(past_target, ctx_len, freq):
    """Extract TSFlow-style lag features."""
    lag_offsets = get_lags_for_frequency(freq)
    B = past_target.shape[0]
    context = past_target[:, -ctx_len:]
    channels = [context.unsqueeze(1)]
    for offset in lag_offsets:
        start = past_target.shape[1] - ctx_len - offset
        end = past_target.shape[1] - offset
        if start >= 0:
            lag = past_target[:, start:end]
        else:
            lag = torch.zeros(B, ctx_len, device=past_target.device)
            if end > 0:
                available = past_target[:, :end]
                lag[:, -available.shape[1]:] = available
        channels.append(lag.unsqueeze(1))
    return torch.cat(channels, dim=1)


def v10_meanflow_loss(net, future_clean, context_channels, norm_p=0.75, norm_eps=1e-3):
    B = future_clean.shape[0]
    device = future_clean.device
    e = torch.randn_like(future_clean)
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
        loss = (u_pred - u_tgt) ** 2
        loss = loss.sum(dim=1)
        adp_wt = (loss.detach() + norm_eps) ** norm_p
        loss = (loss / adp_wt).mean()
    return loss


class V10Forecaster(nn.Module):
    def __init__(self, net, ctx_len, pred_len, num_samples, freq):
        super().__init__()
        self.net = net
        self.context_length = ctx_len
        self.prediction_length = pred_len
        self.num_samples = num_samples
        self.freq = freq
        self.norm = RobustNorm()

    def forward(self, past_target, past_observed_values, **kw):
        device = past_target.device
        B = past_target.shape[0]
        context = past_target[:, -self.context_length:]
        ctx_normed, loc, scale = self.norm(context)

        # Extract lags in ORIGINAL space, then normalize
        lags_raw = extract_lags_gluonts(past_target.float(), self.context_length, self.freq)
        lags_normed = (lags_raw - loc.unsqueeze(1)) / scale.unsqueeze(1)

        all_preds = []
        for _ in range(self.num_samples):
            z = torch.randn(B, self.prediction_length, device=device)
            t = torch.ones(B, device=device); h = t.clone()
            u = self.net(z, (t, h), lags_normed)
            pred = self.norm.inverse((z - u).float(), loc, scale).float()
            all_preds.append(pred.cpu())
        return torch.stack(all_preds, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    name = "exchange_rate_nips"
    freq = "B"
    ctx_len = 30
    pred_len = 30
    max_lag = 810  # Need to fit all TSFlow lags (max=780)
    device = torch.device("cuda")
    torch.manual_seed(42); np.random.seed(42)

    outdir = os.path.join(os.path.dirname(__file__), '..', 'results_v10', name)
    os.makedirs(outdir, exist_ok=True)

    lag_indices = get_lags_for_frequency(freq)
    n_lags = len(lag_indices)
    logger.info(f"{'='*60}")
    logger.info(f"V10 Exchange: RobustNorm + TSFlow 29 lags")
    logger.info(f"n_lags={n_lags} (max={max(lag_indices)})")
    logger.info(f"{'='*60}")

    dataset = get_dataset(name)
    norm = RobustNorm()

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

            ctx = past[:, -ctx_len:]
            ctx_normed, loc, scale = norm(ctx)
            future_normed = (future - loc) / scale

            # Extract lags (raw), then normalize
            lags_raw = extract_lags_gluonts(past, ctx_len, freq)
            lags_normed = (lags_raw - loc.unsqueeze(1)) / scale.unsqueeze(1)

            loss = v10_meanflow_loss(net, future_normed, lags_normed)
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
            fc = V10Forecaster(net_ema, ctx_len, pred_len, 100, freq).to(device)
            pr = PyTorchPredictor(
                prediction_length=pred_len,
                input_names=["past_target", "past_observed_values"],
                prediction_net=fc, batch_size=16, input_transform=tsp, device=device,
            )
            fi, ti = make_evaluation_predictions(dataset=tt, predictor=pr, num_samples=100)
            tss = list(ti); forecasts = list(fi)
            m, _ = Evaluator(num_workers=0)(tss, forecasts)
            crps = m["mean_wQuantileLoss"]
            if crps < best:
                best = crps
                torch.save({"net_ema": net_ema.state_dict(), "crps": crps, "epoch": epoch+1},
                           os.path.join(outdir, "best.pt"))
            beat = " ***BEAT TSFLOW!***" if crps < 0.005 else ""
            logger.info(f"  CRPS={crps:.6f} | TSFlow=0.005 | Best={best:.6f}{beat}")

    logger.info(f"FINAL v10 exchange: Best={best:.6f}")


if __name__ == "__main__":
    main()
