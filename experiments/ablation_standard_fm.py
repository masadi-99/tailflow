"""
Ablation: Same architecture as electricity_v2, but trained with STANDARD flow matching loss.
No JVP, no dual time, h always = 0.

If this achieves similar CRPS, then MeanFlow's JVP isn't helping.
If this is significantly worse, MeanFlow genuinely contributes.
"""
import os, sys, time, math, logging, tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from copy import deepcopy
from tqdm.auto import tqdm

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
except:
    pass

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))
from meanflow_ts.model import ConditionalMeanFlowNet, MeanFlowForecaster

# ============================================================
# Standard FM loss — NO JVP, h=0 always
# ============================================================
def standard_fm_loss(net, future_clean, context, norm_p=0.75, norm_eps=1e-3):
    """Standard conditional flow matching. Same net, but h=0 and target=v.
    Uses same adaptive weighting as MeanFlow for fair comparison."""
    B = future_clean.shape[0]
    device = future_clean.device
    e = torch.randn_like(future_clean)
    t = torch.rand(B, device=device)  # uniform [0,1]
    t_bc = t.unsqueeze(-1)

    z = (1 - t_bc) * future_clean + t_bc * e
    v = e - future_clean

    # h = 0 always (instantaneous velocity)
    h = torch.zeros(B, device=device)
    pred = net(z, (t, h), context)

    # Same adaptive weighting as MeanFlow for fair comparison
    loss = (pred - v) ** 2
    loss = loss.sum(dim=1)
    adp_wt = (loss.detach() + norm_eps) ** norm_p
    loss = (loss / adp_wt).mean()
    return loss


# ============================================================
# Multi-step FM inference (since standard FM can't do 1-step)
# ============================================================
class FMForecaster(nn.Module):
    """Standard FM forecaster with multi-step ODE."""
    def __init__(self, net, context_length, prediction_length, num_samples=100, num_steps=32):
        super().__init__()
        self.net = net
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.num_steps = num_steps

    def forward(self, past_target, past_observed_values, **kwargs):
        device = past_target.device
        B = past_target.shape[0]
        context = past_target[:, -self.context_length:]
        loc = context.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
        scaled_ctx = context / loc

        all_preds = []
        for _ in range(self.num_samples):
            # Start from noise (t=1), integrate backward to data (t=0)
            # v = e - x points from data toward noise
            # So to go noise→data: z = z - dt * v
            z = torch.randn(B, self.prediction_length, device=device)
            dt = 1.0 / self.num_steps
            for step in range(self.num_steps):
                t_val = 1.0 - step * dt  # t goes from 1 → 0
                t_tensor = torch.full((B,), t_val, device=device)
                h_tensor = torch.zeros(B, device=device)
                v = self.net(z, (t_tensor, h_tensor), scaled_ctx)
                z = z - dt * v  # subtract: move toward data
            pred = z * loc
            all_preds.append(pred)
        return torch.stack(all_preds, dim=1)


class FM1StepForecaster(nn.Module):
    """Standard FM with 1-step."""
    def __init__(self, net, context_length, prediction_length, num_samples=100):
        super().__init__()
        self.net = net
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples

    def forward(self, past_target, past_observed_values, **kwargs):
        device = past_target.device
        B = past_target.shape[0]
        context = past_target[:, -self.context_length:]
        loc = context.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
        scaled_ctx = context / loc

        all_preds = []
        for _ in range(self.num_samples):
            # Start at noise (t=1), one step toward data (t=0)
            z = torch.randn(B, self.prediction_length, device=device)
            t_tensor = torch.ones(B, device=device)   # t=1 (noise)
            h_tensor = torch.zeros(B, device=device)
            v = self.net(z, (t_tensor, h_tensor), scaled_ctx)
            pred = (z - v) * loc  # z_0 = z_1 - v
            all_preds.append(pred)
        return torch.stack(all_preds, dim=1)


def evaluate(net_ema, forecaster_cls, dataset, transformation, device, label, **kwargs):
    net_ema.eval()
    test_transform = transformation.apply(dataset.test, is_train=False)
    test_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
        past_length=24 + 672, future_length=24,
        time_series_fields=["time_feat", "observed_values"],
    )
    forecaster = forecaster_cls(net_ema, 24, 24, num_samples=16, **kwargs).to(device)
    predictor = PyTorchPredictor(
        prediction_length=24, input_names=["past_target", "past_observed_values"],
        prediction_net=forecaster, batch_size=512, input_transform=test_splitter, device=device,
    )
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_transform, predictor=predictor, num_samples=16,
    )
    forecasts = list(tqdm(forecast_it, total=len(test_transform), desc=label))
    tss = list(ts_it)
    evaluator = Evaluator(num_workers=1)
    metrics, _ = evaluator(tss, forecasts)
    return metrics["mean_wQuantileLoss"], metrics["ND"], metrics["NRMSE"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = "electricity_nips"
    context_length = 24
    prediction_length = 24
    batch_size = 64
    num_batches_per_epoch = 128
    max_epochs = 600  # same as MeanFlow
    lr = 6e-4  # same as MeanFlow
    ema_decay = 0.9999
    seed = 6432

    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = get_dataset(dataset_name)
    transformation = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(
            start_field="start", target_field="target", output_field="time_feat",
            time_features=time_features_from_frequency_str("H"), pred_length=prediction_length,
        ),
    ])

    train_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start",
        instance_sampler=ExpectedNumInstanceSampler(num_instances=1, min_future=prediction_length),
        past_length=context_length + 672, future_length=prediction_length,
        time_series_fields=["time_feat", "observed_values"],
    )
    transformed_data = transformation.apply(dataset.train, is_train=True)
    train_loader = TrainDataLoader(
        Cached(transformed_data), batch_size=batch_size, stack_fn=batchify,
        transform=train_splitter, num_batches_per_epoch=num_batches_per_epoch,
        shuffle_buffer_length=10000,
    )

    # SAME architecture as MeanFlow
    net = ConditionalMeanFlowNet(
        pred_len=prediction_length, ctx_len=context_length,
        model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
    ).to(device)
    net_ema = deepcopy(net).eval()
    n_params = sum(p.numel() for p in net.parameters())
    logger.info(f"Standard FM ablation — same arch, {n_params:,} params")

    optimizer = AdamW(net.parameters(), lr=lr)

    for epoch in range(max_epochs):
        net.train()
        epoch_loss = 0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            past_target = batch["past_target"].to(device)
            future_target = batch["future_target"].to(device)
            context = past_target[:, -context_length:]
            loc = context.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
            scaled_ctx = context / loc
            scaled_future = future_target / loc

            loss = standard_fm_loss(net, scaled_future, scaled_ctx)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for p, pe in zip(net.parameters(), net_ema.parameters()):
                    pe.data.lerp_(p.data, 1 - ema_decay)
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        elapsed = time.time() - t0

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:>3}/{max_epochs} | FM Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s")

        # Evaluate every 50 epochs — test BOTH 1-step and 32-step
        if (epoch + 1) % 50 == 0:
            logger.info("Evaluating...")
            net_ema.eval()

            # 1-step (same as how MeanFlow is evaluated)
            crps_1, nd_1, nrmse_1 = evaluate(
                net_ema, FM1StepForecaster, dataset, transformation, device, "FM-1step"
            )
            logger.info(f"  FM 1-step:  CRPS={crps_1:.6f} | ND={nd_1:.6f} | NRMSE={nrmse_1:.4f}")

            # 32-step ODE (fair for standard FM)
            crps_32, nd_32, nrmse_32 = evaluate(
                net_ema, FMForecaster, dataset, transformation, device, "FM-32step", num_steps=32
            )
            logger.info(f"  FM 32-step: CRPS={crps_32:.6f} | ND={nd_32:.6f} | NRMSE={nrmse_32:.4f}")

            logger.info(f"  MeanFlow (1-step) best: CRPS=0.0558")
            logger.info(f"  TSFlow (32-step) best:  CRPS=0.0446")

    logger.info("Done.")


if __name__ == "__main__":
    main()
