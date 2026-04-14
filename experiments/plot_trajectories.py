"""
Qualitative trajectory visualization for the paper.

Loads a v4 checkpoint and plots a small grid of forecast samples at several
inference-time noise scales, overlayed on the ground-truth window. This
shows visually that the noise scale `n` is a genuine dial over the
body-tail trade-off on the same trained model.
"""
from __future__ import annotations
import os, sys, argparse, logging
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.transform import (
    AddObservedValuesIndicator, AddTimeFeatures, AsNumpyArray,
    Chain, InstanceSplitter, TestSplitSampler,
)
from gluonts.time_feature import time_features_from_frequency_str

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v4 import (
    S4DMeanFlowNetV4, get_lag_indices_v4, extract_lags_v4, RobustNorm,
)
from experiments.train_v4 import CONFIGS, LAG_MAP

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


@torch.no_grad()
def sample_at_noise(net, past_target, ctx_len, pred_len, freq, noise,
                    n_samples, device):
    B = past_target.shape[0]
    ctx = past_target[:, -ctx_len:]
    norm = RobustNorm()
    _, loc, scale = norm(ctx)
    lags = extract_lags_v4(past_target.float(), ctx_len, freq)
    lags_n = (lags - loc.unsqueeze(1)) / scale.unsqueeze(1)
    samples = []
    for _ in range(n_samples):
        z = torch.randn(B, pred_len, device=device) * noise
        t = torch.ones(B, device=device); h = t.clone()
        u = net(z, (t, h), lags_n)
        pred = norm.inverse((z - u).float(), loc, scale).float()
        samples.append(pred.cpu())
    return torch.stack(samples, dim=1).numpy()  # (B, S, T)


def collect_n_windows(dataset_test, cfg, n_windows, device):
    """Pick the n_windows with the highest ground-truth max."""
    pred_len = cfg["pred"]
    items = []
    for entry in dataset_test:
        full = np.asarray(entry["target"], dtype=np.float32)
        if full.ndim == 2: full = full[0]
        if len(full) <= pred_len:
            continue
        past = full[:-pred_len]
        fut = full[-pred_len:]
        items.append((float(fut.max()), past, fut))
    items.sort(reverse=True, key=lambda x: x[0])
    return items[:n_windows]


def plot_panel(ax, past, fut, samples_by_n, pred_len, max_show=100):
    T_past = len(past)
    xs_past = np.arange(T_past)
    xs_fut = np.arange(T_past, T_past + pred_len)

    # Plot only the last 72 of past for readability
    show_past = min(72, T_past)
    ax.plot(xs_past[-show_past:], past[-show_past:], "k-", lw=1.5, label="context")
    ax.plot(xs_fut, fut, "k-", lw=2.0, label="ground truth")
    ax.axvline(T_past - 0.5, color="gray", ls=":", alpha=0.5)

    colors = {"n=1.0": "C0", "n=2.0": "C1", "n=3.0": "C2"}
    for name, s in samples_by_n.items():
        # s: (S, T), S samples for THIS window
        median = np.median(s, axis=0)
        q05 = np.quantile(s, 0.05, axis=0)
        q95 = np.quantile(s, 0.95, axis=0)
        ax.fill_between(xs_fut, q05, q95, color=colors.get(name, "gray"),
                        alpha=0.2)
        ax.plot(xs_fut, median, "--", color=colors.get(name, "gray"),
                lw=1.8, label=f"{name} median")

    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="solar_nips")
    p.add_argument("--ckpt", default="results_v4/solar_nips/best.pt")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--n-windows", type=int, default=4)
    p.add_argument("--noise-grid", type=float, nargs="+",
                   default=[1.0, 2.0, 3.0])
    p.add_argument("--out", default="trajectories_solar.png")
    args = p.parse_args()

    cfg = CONFIGS[args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_lags = len(get_lag_indices_v4(cfg["freq"]))
    net = S4DMeanFlowNetV4(
        pred_len=cfg["pred"], ctx_len=cfg["ctx"],
        d_model=192, n_s4d_blocks=6, ssm_dim=64,
        n_lags=n_lags, freq=cfg["freq"],
    ).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    net.load_state_dict(ck.get("net_ema") or ck.get("net"))
    net.eval()

    ds = get_dataset(args.dataset)
    windows = collect_n_windows(ds.test, cfg, args.n_windows, device)
    logger.info(f"Selected {len(windows)} highest-peak test windows")

    fig, axes = plt.subplots(len(windows), 1, figsize=(10, 3 * len(windows)),
                              sharex=False)
    if len(windows) == 1:
        axes = [axes]
    for i, (peak, past, fut) in enumerate(windows):
        past_t = torch.tensor(past, device=device).unsqueeze(0)
        samples_by_n = {}
        for n in args.noise_grid:
            s = sample_at_noise(net, past_t, cfg["ctx"], cfg["pred"],
                                 cfg["freq"], float(n), args.n_samples, device)
            samples_by_n[f"n={n}"] = s[0]  # (S, T)
        plot_panel(axes[i], past, fut, samples_by_n, cfg["pred"])
        axes[i].set_title(f"{args.dataset}: gt peak={peak:.1f}")
    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    logger.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
