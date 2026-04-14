# TailFlow

One-step probabilistic time-series forecasting via MeanFlow with an S4D
backbone, plus an extremity-conditioned extension that enables
controllable rare-event generation via classifier-free guidance.

**Headline claim.** TailFlow is the only method in our evaluation that is
simultaneously (a) calibrated at the 90% PI level, (b) competitive or
better on upper-tail twCRPS, and (c) uses a single neural function
evaluation at inference — 32× fewer than TSFlow (Kollovieh et al., ICLR
2025). TailFlow matches or beats TSFlow on calibration CRPS, coverage,
Winkler interval score, and threshold-weighted CRPS across the four NIPS
benchmark datasets (solar, electricity, traffic, exchange_rate), and
cleanly dominates seasonal-naïve and bootstrap baselines on a heavy-tailed
air-quality benchmark (KDD Cup 2018).

Full method description, result tables, and an honest discussion of the
limitations are in [RESULTS.md](RESULTS.md). Start there.

## Repository layout

| Path                                | Purpose                                      |
|-------------------------------------|----------------------------------------------|
| [`meanflow_ts/model_v4.py`](meanflow_ts/model_v4.py)           | Production backbone (S4D-MeanFlow v4, 2.2M params) |
| [`meanflow_ts/model_v4_tail.py`](meanflow_ts/model_v4_tail.py) | Extremity adapter wrapping the v4 backbone (WIP) |
| [`meanflow_ts/model_tail.py`](meanflow_ts/model_tail.py)       | Extremity functional + QuantileMapper + tilted resampling |
| [`meanflow_ts/tail_metrics.py`](meanflow_ts/tail_metrics.py)   | twCRPS, quantile loss, Winkler, coverage, Brier — used by *every* row in RESULTS.md |
| [`experiments/train_v4.py`](experiments/train_v4.py)           | Train the unconditional v4 base on any GluonTS dataset |
| [`experiments/train_tail_v2.py`](experiments/train_tail_v2.py) | Extremity-conditioned pipeline on the small 500K backbone (Phase 1/2/3) |
| [`experiments/train_tail_v4.py`](experiments/train_tail_v4.py) | Extremity adapter fine-tune on the 2.2M v4 backbone (WIP — did not converge in our attempt) |
| [`experiments/eval_v4_tail.py`](experiments/eval_v4_tail.py)   | Score any v4 checkpoint with our tail metrics at a given noise scale |
| [`experiments/eval_tail_metrics.py`](experiments/eval_tail_metrics.py) | Score the extremity-conditioned model + CFG guidance sweep |
| [`experiments/eval_baselines_tail.py`](experiments/eval_baselines_tail.py) | Seasonal-naïve and bootstrap baselines |
| [`experiments/train_tsflow_baseline.py`](experiments/train_tsflow_baseline.py) | Thin wrapper that retrains TSFlow (ICLR 2025) from the official repo and scores its outputs with our tail_metrics |
| [`experiments/eval_smart_blend_s500.py`](experiments/eval_smart_blend_s500.py) | Solar diurnal blend sweep (the final 0.3513 leaderboard config) |
| `results_*.json`                    | Small numeric snapshots backing the RESULTS.md tables |

Checkpoints, dataset caches, and the third-party TSFlow clone are
intentionally excluded from the repo — reproduce them via the commands in
[RESULTS.md §6](RESULTS.md).

## Quick reproduction

```bash
# Install deps (pytorch + gluonts + a few extras for TSFlow)
pip install gluonts torch torchvision einops opt_einsum lightning

# Train the v4 base on one dataset (~30 min on a modern GPU)
python experiments/train_v4.py solar_nips --epochs 800 --no-gp

# Score it with our tail metrics at several noise scales
python experiments/eval_v4_tail.py \
    --dataset solar_nips \
    --ckpt results_v4/solar_nips/best.pt \
    --num-samples 200 --noise 2.0 \
    --output results_v4_tail.json

# Leaderboard config for solar (noise + diurnal blend)
python experiments/eval_smart_blend_s500.py

# Simple baselines (seasonal naïve + bootstrap) on the same test splits
python experiments/eval_baselines_tail.py \
    --num-samples 200 \
    --output results_baselines_tail.json

# External learned baseline (TSFlow retrained from official repo).
# Requires: git clone https://github.com/marcelkollovieh/TSFlow third_party/TSFlow
# and installing its deps (pykeops, torchdyn, ema_pytorch, linear_attention_transformer, POT).
python experiments/train_tsflow_baseline.py \
    --datasets solar_nips --epochs 400 --num-samples 100 \
    --logdir tsflow_runs --output results_tsflow_baseline.json
```

## Acknowledgements

This repo builds on:

- **MeanFlow** — Geng et al., *Mean Flows for One-step Generative
  Modeling*, 2025.
- **TSFlow** — Kollovieh et al., *Flow Matching with Gaussian Process
  Priors for Probabilistic Time Series Forecasting*, ICLR 2025.
  ([GitHub](https://github.com/marcelkollovieh/TSFlow))
- **S4D** — Gu et al., *On the Parameterization and Initialization of
  Diagonal State Space Models*, NeurIPS 2022.
- **Threshold-weighted CRPS** — Gneiting & Ranjan, *Comparing Density
  Forecasts Using Threshold- and Quantile-Weighted Scoring Rules*, JBES
  2011. Implementation in [`meanflow_ts/tail_metrics.py`](meanflow_ts/tail_metrics.py).
- **Tail calibration** — Allen, Ziegel & Ginsbourger, *Tail Calibration
  of Probabilistic Forecasts*, JASA 2025.
- **Extreme Value Loss (EVL)** — Ding et al., *Modeling Extreme Events
  in Time Series Prediction*, KDD 2019.

The `.gluonts` data pipeline and the GluonTS `Evaluator.mean_wQuantileLoss`
metric are from [awslabs/gluonts](https://github.com/awslabs/gluonts).
