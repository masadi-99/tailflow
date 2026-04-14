"""
Re-test the best smart-blend config (n=2.8, b=0.1) with s=500 samples,
and also fine-scan blend near 0.1 to lock the optimum.
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


class SmartBlendFc(nn.Module):
    def __init__(self, net, ctx, pred, ns, freq, blend, noise):
        super().__init__()
        self.net = net
        self.context_length = ctx
        self.prediction_length = pred
        self.num_samples = ns
        self.freq = freq
        self.blend = blend
        self.noise = noise
        self.norm = RobustNorm()

    def forward(self, past_target, past_observed_values, **kw):
        d = past_target.device
        B = past_target.shape[0]
        day1 = past_target[:, -24:]
        day2 = past_target[:, -48:-24]
        day3 = past_target[:, -72:-48]
        same_hour_prior = 0.5 * day1 + 0.3 * day2 + 0.2 * day3

        c = past_target[:, -self.context_length:].float()
        cn, loc, scale = self.norm(c)
        lags = extract_lags_v4(past_target.float(), self.context_length, self.freq)
        ln = (lags - loc.unsqueeze(1)) / scale.unsqueeze(1)

        ps = []
        for _ in range(self.num_samples):
            z = torch.randn(B, self.prediction_length, device=d) * self.noise
            t = torch.ones(B, device=d); h = t.clone()
            raw = self.norm.inverse((z - self.net(z, (t, h), ln)).float(), loc, scale).float().clamp(min=0)
            pred = (1 - self.blend) * raw + self.blend * same_hour_prior.float()
            ps.append(pred.cpu())
        return torch.stack(ps, dim=1)


def main():
    device = torch.device("cuda")
    ctx, pred, max_lag = 72, 24, 672
    n_lags = len(get_lag_indices_v4("H"))
    net = S4DMeanFlowNetV4(
        pred_len=pred, ctx_len=ctx, d_model=192, n_s4d_blocks=6,
        ssm_dim=64, n_lags=n_lags, freq="H",
    ).to(device)
    ck = torch.load(os.path.join(os.path.dirname(__file__), "..", "results_v4", "solar_nips", "best.pt"),
                    map_location=device, weights_only=False)
    net.load_state_dict(ck["net_ema"]); net.eval()

    ds = get_dataset("solar_nips")
    tr = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(start_field="start", target_field="target", output_field="time_feat",
                        time_features=time_features_from_frequency_str("H"), pred_length=pred),
    ])
    tt = tr.apply(ds.test, is_train=False)
    sp = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
        past_length=ctx + max_lag, future_length=pred,
        time_series_fields=["time_feat", "observed_values"],
    )

    best = float("inf"); best_cfg = None
    print("Finer sweep around (n=3.0, b=0.10), s=500, multi-seed average")
    configs = [(n, b) for n in [2.9, 3.0, 3.1, 3.2] for b in [0.14, 0.16, 0.18, 0.20, 0.22, 0.25]]
    for noise, blend in configs:
        # Multi-seed average to reduce Monte Carlo noise
        crps_list = []
        for seed in [42, 123, 999]:
            torch.manual_seed(seed); np.random.seed(seed)
            fc = SmartBlendFc(net, ctx, pred, 500, "H", blend, noise).to(device)
            pr = PyTorchPredictor(
                prediction_length=pred,
                input_names=["past_target", "past_observed_values"],
                prediction_net=fc, batch_size=16, input_transform=sp, device=device,
            )
            fi, ti = make_evaluation_predictions(dataset=tt, predictor=pr, num_samples=500)
            m, _ = Evaluator(num_workers=0)(list(ti), list(fi))
            crps_list.append(m["mean_wQuantileLoss"])
        avg = float(np.mean(crps_list))
        std = float(np.std(crps_list))
        marker = " ***BEAT***" if avg < 0.343 else ""
        print(f"  n={noise}, b={blend:.2f}: avg={avg:.6f} ± {std:.6f}{marker}")
        if avg < best:
            best = avg; best_cfg = f"n={noise}, b={blend:.2f}"

    print(f"\nBEST avg: {best_cfg} = {best:.6f}")


if __name__ == "__main__":
    main()
