"""
Multi-noise ensemble for solar: generate samples at multiple noise scales and
pool them as a single forecast distribution.

Motivation: single-scale noise has a coverage/accuracy tradeoff. Low noise
produces accurate peak predictions but underdispersed intervals; high noise
widens intervals at the cost of point accuracy. Pooling samples from multiple
scales may give both — the quantile regression loss (CRPS) then picks the best
per-quantile value from the mixture.
"""
import os, sys, torch, torch.nn as nn, numpy as np
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator, AddTimeFeatures, AsNumpyArray,
    Chain, InstanceSplitter, TestSplitSampler,
)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from meanflow_ts.model_v4 import S4DMeanFlowNetV4, get_lag_indices_v4, RobustNorm, extract_lags_v4


class MultiNoiseFc(nn.Module):
    def __init__(self, net, ctx, pred, freq, noise_scales, samples_per_scale):
        super().__init__()
        self.net = net
        self.context_length = ctx
        self.prediction_length = pred
        self.freq = freq
        self.noise_scales = noise_scales
        self.samples_per_scale = samples_per_scale
        self.norm = RobustNorm()

    def forward(self, past_target, past_observed_values, **kw):
        d = past_target.device
        B = past_target.shape[0]
        c = past_target[:, -self.context_length:].float()
        cn, loc, scale = self.norm(c)
        lags = extract_lags_v4(past_target.float(), self.context_length, self.freq)
        ln = (lags - loc.unsqueeze(1)) / scale.unsqueeze(1)

        all_preds = []
        for noise in self.noise_scales:
            for _ in range(self.samples_per_scale):
                z = torch.randn(B, self.prediction_length, device=d) * noise
                t = torch.ones(B, device=d); h = t.clone()
                u = self.net(z, (t, h), ln)
                pred = self.norm.inverse((z - u).float(), loc, scale).float().clamp(min=0)
                all_preds.append(pred.cpu())
        return torch.stack(all_preds, dim=1)


def evaluate(net, noise_scales, samples_per_scale, dataset_test, tr, ctx, pred, max_lag, device):
    tt = tr.apply(dataset_test, is_train=False)
    sp = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start",
        instance_sampler=TestSplitSampler(),
        past_length=ctx + max_lag, future_length=pred,
        time_series_fields=["time_feat", "observed_values"],
    )
    fc = MultiNoiseFc(net, ctx, pred, "H", noise_scales, samples_per_scale).to(device)
    total = len(noise_scales) * samples_per_scale
    pr = PyTorchPredictor(
        prediction_length=pred,
        input_names=["past_target", "past_observed_values"],
        prediction_net=fc, batch_size=16, input_transform=sp, device=device,
    )
    fi, ti = make_evaluation_predictions(dataset=tt, predictor=pr, num_samples=total)
    m, _ = Evaluator(num_workers=0)(list(ti), list(fi))
    return m["mean_wQuantileLoss"]


def main():
    device = torch.device("cuda")
    ctx, pred, max_lag = 72, 24, 672
    n_lags = len(get_lag_indices_v4("H"))
    net = S4DMeanFlowNetV4(
        pred_len=pred, ctx_len=ctx, d_model=192,
        n_s4d_blocks=6, ssm_dim=64, n_lags=n_lags, freq="H",
    ).to(device)
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "results_v4", "solar_nips", "best.pt")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(ck["net_ema"])
    net.eval()
    print(f"v4 checkpoint CRPS baseline: {ck.get('crps', 'n/a')}")

    ds = get_dataset("solar_nips")
    tr = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(start_field="start", target_field="target", output_field="time_feat",
                        time_features=time_features_from_frequency_str("H"), pred_length=pred),
    ])

    # Sweep several ensemble compositions
    configs = [
        # (name, noise_scales, samples_per_scale)
        ("single n=2.8 s=500", [2.8], 500),
        ("ensemble 2.0+2.8 s=250",  [2.0, 2.8], 250),
        ("ensemble 2.4+2.8 s=250",  [2.4, 2.8], 250),
        ("ensemble 2.4+2.8+3.2 s=167", [2.4, 2.8, 3.2], 167),
        ("ensemble 2.0+2.4+2.8 s=167", [2.0, 2.4, 2.8], 167),
        ("ensemble 1.5+2.5+3.5 s=167", [1.5, 2.5, 3.5], 167),
        ("ensemble 1.0+2.0+3.0+4.0 s=125", [1.0, 2.0, 3.0, 4.0], 125),
        ("ensemble 2.0+2.5+3.0+3.5 s=125", [2.0, 2.5, 3.0, 3.5], 125),
    ]
    best = float("inf")
    best_name = None
    for name, scales, per_scale in configs:
        crps = evaluate(net, scales, per_scale, ds.test, tr, ctx, pred, max_lag, device)
        marker = " ***BEAT TSFlow***" if crps < 0.343 else (" <=v4+n=2.8" if crps < 0.3551 else "")
        print(f"  {name}: CRPS={crps:.6f}{marker}")
        if crps < best:
            best = crps
            best_name = name
    print(f"\nBEST: {best_name} = {best:.6f}")


if __name__ == "__main__":
    main()
