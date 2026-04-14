"""
Build the Pareto plot for the paper: body CRPS vs tail twCRPS on solar.

Three families are overlaid:
  (A) v4 base + inference-time noise-scale sweep n ∈ {1.0, 1.5, 2.0, 2.5, 3.0}
      — the *inference-time dial*.
  (B) v4 + twCRPS fine-tune (Wessel et al. 2025 style) at several τ values,
      each also scored across noise scales. One point per (τ, n) pair.
      — the *training-time retraining* family.
  (C) Simple baselines (seasonal naïve, bootstrap) for orientation.

The claim we want to empirically support: family (A) is competitive with or
Pareto-dominates family (B) without requiring retraining.

Reads JSON files produced by:
  eval_v4_tail.py  (v4 base + noise sweep, and twCRPS fine-tunes)
  eval_baselines_tail.py
"""
from __future__ import annotations
import json, os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def extract_sweep(data, dataset_key_prefix):
    """For /tmp/v4_solar_tail.json style files keyed like 'solar_nips_n1.0'."""
    rows = []
    for k, v in data.items():
        if not k.startswith(dataset_key_prefix):
            continue
        if "noise_scale" in v:
            rows.append((float(v["noise_scale"]),
                         float(v["gluonts_mean_wQuantileLoss"]),
                         float(v["twCRPS_q095"]),
                         float(v["twCRPS_q099"]),
                         float(v["coverage_90"])))
    rows.sort()
    return rows


def plot_pareto(v4_rows, twcrps_rows_by_tau, baseline_rows, out_path,
                 dataset_label="Solar"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes

    # Panel 1: CRPS vs twCRPS95
    if v4_rows:
        ns = [r[0] for r in v4_rows]
        crps = [r[1] for r in v4_rows]
        tw95 = [r[2] for r in v4_rows]
        ax1.plot(crps, tw95, "o-", color="C0", label="v4 + inference-time n",
                 lw=2, ms=8, zorder=3)
        for n, c, t in zip(ns, crps, tw95):
            ax1.annotate(f"n={n}", (c, t), fontsize=8, xytext=(5, 5),
                          textcoords="offset points")
    colors = ["C1", "C2", "C3"]
    for i, (tau, rows) in enumerate(twcrps_rows_by_tau.items()):
        if not rows:
            continue
        ns = [r[0] for r in rows]
        crps = [r[1] for r in rows]
        tw95 = [r[2] for r in rows]
        ax1.plot(crps, tw95, "s--", color=colors[i % len(colors)],
                 label=f"v4 + twCRPS(τ={tau})", lw=1.8, ms=7, alpha=0.85)
    for name, m in baseline_rows.items():
        ax1.scatter(m.get("crps_norm", np.nan), m.get("twCRPS_q095", np.nan),
                    marker="x", s=100, color="grey", label=name, zorder=4)
    ax1.set_xlabel("Body CRPS (gluonts mean_wQuantileLoss)")
    ax1.set_ylabel("Upper-tail twCRPS₉₅")
    ax1.set_title(f"{dataset_label}: body vs tail, τ=0.95")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="best")

    # Panel 2: CRPS vs twCRPS99
    if v4_rows:
        ns = [r[0] for r in v4_rows]
        crps = [r[1] for r in v4_rows]
        tw99 = [r[3] for r in v4_rows]
        ax2.plot(crps, tw99, "o-", color="C0", label="v4 + inference-time n",
                 lw=2, ms=8, zorder=3)
    for i, (tau, rows) in enumerate(twcrps_rows_by_tau.items()):
        if not rows:
            continue
        ns = [r[0] for r in rows]
        crps = [r[1] for r in rows]
        tw99 = [r[3] for r in rows]
        ax2.plot(crps, tw99, "s--", color=colors[i % len(colors)],
                 label=f"v4 + twCRPS(τ={tau})", lw=1.8, ms=7, alpha=0.85)
    ax2.set_xlabel("Body CRPS")
    ax2.set_ylabel("Upper-tail twCRPS₉₉")
    ax2.set_title(f"{dataset_label}: body vs tail, τ=0.99")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v4-json", default="/tmp/v4_solar_tail.json")
    p.add_argument("--twcrps-json", default="/tmp/twcrps_eval_final.json")
    p.add_argument("--baselines-json", default="results_baselines_tail.json")
    p.add_argument("--out", default="pareto_solar.png")
    p.add_argument("--dataset", default="solar_nips")
    args = p.parse_args()

    v4_data = load_json(args.v4_json)
    twcrps_data = load_json(args.twcrps_json)
    baselines_data = load_json(args.baselines_json)

    v4_rows = extract_sweep(v4_data, f"{args.dataset}_n")
    # twCRPS entries keyed as 'solar_nips_tau0.9_n1.0' etc. Group by tau.
    twcrps_by_tau = {}
    for k, v in twcrps_data.items():
        if not k.startswith(args.dataset):
            continue
        # key format: <ds>_tau<TAU>_n<N>
        try:
            tau = k.split("_tau")[1].split("_n")[0]
            n = float(k.split("_n")[1])
        except Exception:
            continue
        row = (n,
               float(v["gluonts_mean_wQuantileLoss"]),
               float(v["twCRPS_q095"]),
               float(v["twCRPS_q099"]),
               float(v["coverage_90"]))
        twcrps_by_tau.setdefault(tau, []).append(row)
    for t in twcrps_by_tau:
        twcrps_by_tau[t].sort()

    baseline_rows = {}
    ds_base = (baselines_data.get("baselines") or {}).get(args.dataset, {})
    for method, m in ds_base.items():
        baseline_rows[method] = m

    plot_pareto(v4_rows, twcrps_by_tau, baseline_rows, args.out,
                 dataset_label=args.dataset)


if __name__ == "__main__":
    main()
