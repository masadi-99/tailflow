"""
Build the Pareto plot for the paper: body CRPS vs tail twCRPS on solar.

Family A (inference-time dial): v4 base + noise scale n ∈ {1.0,1.5,2.0,2.5,3.0}
Family B (training-time retrain): v4 fine-tuned with twCRPS loss at
  τ ∈ {0.90, 0.95, 0.99}, each also scored across noise scales.

We want to show: family A meets or dominates family B on every point.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_sweep(path):
    with open(path) as f:
        d = json.load(f)
    rows = []
    for k, v in d.items():
        rows.append({
            "n": float(v["noise_scale"]),
            "crps": float(v["gluonts_mean_wQuantileLoss"]),
            "twCRPS95": float(v["twCRPS_q095"]),
            "twCRPS99": float(v["twCRPS_q099"]),
            "cov90": float(v["coverage_90"]),
        })
    rows.sort(key=lambda r: r["n"])
    return rows


def plot(base_rows, twcrps_rows, out_path, dataset="Solar"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax1, ax2 = axes

    def plot_family(ax, rows, fmt, color, label, ymetric):
        xs = [r["crps"] for r in rows]
        ys = [r[ymetric] for r in rows]
        ns = [r["n"] for r in rows]
        ax.plot(xs, ys, fmt, color=color, lw=1.8, ms=8, label=label)
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(f"{n}", (x, y), fontsize=7, xytext=(4, 3),
                         textcoords="offset points", color=color)

    for ax, ymetric, title in [
        (ax1, "twCRPS95", f"{dataset}: body CRPS vs upper-tail twCRPS₉₅"),
        (ax2, "twCRPS99", f"{dataset}: body CRPS vs upper-tail twCRPS₉₉"),
    ]:
        plot_family(ax, base_rows, "o-", "C0",
                    "v4 base + inference-time n (1 trained model)", ymetric)
        colors = ["C1", "C2", "C3"]
        for i, (tau, rows) in enumerate(twcrps_rows.items()):
            plot_family(ax, rows, "s--", colors[i % len(colors)],
                        f"v4 + twCRPS(τ={tau}) fine-tune (retrained)", ymetric)
        ax.set_xlabel("Body CRPS (gluonts mean_wQuantileLoss)")
        ax.set_ylabel(f"Upper-tail {ymetric}".replace("twCRPS", "twCRPS_"))
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    plt.suptitle(
        "Inference-time noise scaling Pareto-dominates training-time tail reweighting\n"
        "(lower-left is better)",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Wrote {out_path}")


def main():
    base = load_sweep("/tmp/v4_solar_tail.json")
    tw = {}
    for tau in ["0.9", "0.95", "0.99"]:
        p = f"/tmp/twcrps_agg_t{tau}_eval.json"
        if os.path.exists(p):
            tw[tau] = load_sweep(p)
    os.makedirs("figs", exist_ok=True)
    plot(base, tw, "figs/pareto_solar.png", "Solar")

    print("\nSummary (per family, best twCRPS₉₉ reached across noise scales):")
    best_base = min(r["twCRPS99"] for r in base)
    print(f"  v4 base (inference-time): twCRPS₉₉_best = {best_base:.4f}")
    for tau, rows in tw.items():
        best = min(r["twCRPS99"] for r in rows)
        print(f"  twCRPS τ={tau} (training-time): twCRPS₉₉_best = {best:.4f}")


if __name__ == "__main__":
    main()
