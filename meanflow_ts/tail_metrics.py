"""
Tail-aware probabilistic forecast metrics.

Implements:
- Threshold-weighted CRPS (twCRPS) via the sample-based estimator of
  Gneiting & Ranjan 2011 / Allen et al. 2024.
- Quantile (pinball) loss at individual quantile levels.
- Winkler / interval score at a given prediction-interval level.
- Brier score on threshold exceedance.
- Empirical coverage of central prediction intervals.

All functions accept:
    samples: (N_windows, N_samples, pred_len) float array
    target:  (N_windows, pred_len) float array

and return either a scalar or a per-window array (reductions noted per fn).

The twCRPS estimator:

    twCRPS(F, y) = E|v(X) - v(y)| - 0.5 * E|v(X) - v(X')|

where v is a chaining / weight function. For a right-tail-weighted score
above threshold t, the standard choice is v(x) = max(x, t), which gives
an upper-tail-focused strictly proper score. This is the same expression
used in Allen et al. (JASA 2025) tail-calibration paper.
"""
from __future__ import annotations
import numpy as np


def _flatten_windows(samples, target):
    # (N, S, T) x (N, T) -> flatten over (N, T) -> (NT, S) and (NT,)
    N, S, T = samples.shape
    s_flat = samples.transpose(0, 2, 1).reshape(N * T, S)
    y_flat = target.reshape(N * T)
    return s_flat, y_flat


def crps_sample(samples, target):
    """
    Classical CRPS estimator (Gneiting 2007, eq. 2).

    CRPS(F, y) ~ mean_s |X_s - y| - 0.5 * mean_{s,s'} |X_s - X_{s'}|
    """
    s_flat, y_flat = _flatten_windows(samples, target)
    abs_err = np.abs(s_flat - y_flat[:, None]).mean(axis=1)
    # E|X - X'| via sorted diffs (O(S log S))
    s_sorted = np.sort(s_flat, axis=1)
    S = s_sorted.shape[1]
    w = 2.0 * np.arange(S) - (S - 1)
    ex_xprime = (w[None, :] * s_sorted).mean(axis=1) * (2.0 / S)
    return float((abs_err - 0.5 * ex_xprime).mean())


def tw_crps_sample(samples, target, threshold, side="upper"):
    """
    Threshold-weighted CRPS using chaining function v(x) = max(x, t) for
    upper tails, or v(x) = min(x, t) for lower tails.

    samples: (N, S, T), target: (N, T)
    threshold: scalar (applied globally) or per-window (N,) or
               per-(window, time) (N, T).

    Returns a scalar twCRPS — mean over all (window, time) positions.
    """
    s_flat, y_flat = _flatten_windows(samples, target)
    if np.ndim(threshold) == 0:
        t_flat = np.full_like(y_flat, threshold, dtype=np.float64)
    else:
        t_arr = np.asarray(threshold, dtype=np.float64)
        if t_arr.shape == (samples.shape[0],):
            t_flat = np.repeat(t_arr, samples.shape[2])
        elif t_arr.shape == samples.shape[0:1] + samples.shape[2:3]:
            t_flat = t_arr.reshape(-1)
        else:
            raise ValueError(f"threshold shape {t_arr.shape} not understood")

    if side == "upper":
        vx = np.maximum(s_flat, t_flat[:, None])
        vy = np.maximum(y_flat, t_flat)
    elif side == "lower":
        vx = np.minimum(s_flat, t_flat[:, None])
        vy = np.minimum(y_flat, t_flat)
    else:
        raise ValueError(side)

    abs_err = np.abs(vx - vy[:, None]).mean(axis=1)
    vx_sorted = np.sort(vx, axis=1)
    S = vx_sorted.shape[1]
    w = 2.0 * np.arange(S) - (S - 1)
    ex_xprime = (w[None, :] * vx_sorted).mean(axis=1) * (2.0 / S)
    return float((abs_err - 0.5 * ex_xprime).mean())


def quantile_loss(samples, target, q):
    """Pinball loss at level q."""
    s_flat, y_flat = _flatten_windows(samples, target)
    q_hat = np.quantile(s_flat, q, axis=1)
    diff = y_flat - q_hat
    loss = np.where(diff >= 0, q * diff, (q - 1) * diff)
    return float(loss.mean())


def weighted_quantile_loss(samples, target, q):
    """Normalized (mean_wQuantileLoss style) — sum|pinball| / sum|y|."""
    s_flat, y_flat = _flatten_windows(samples, target)
    q_hat = np.quantile(s_flat, q, axis=1)
    diff = y_flat - q_hat
    ql = np.where(diff >= 0, q * diff, (q - 1) * diff)
    denom = np.abs(y_flat).sum()
    if denom < 1e-12:
        return float("nan")
    return float(2.0 * ql.sum() / denom)


def winkler_score(samples, target, alpha):
    """
    Winkler interval score for central (1-alpha) PI.
    alpha=0.1 -> 90% PI; alpha=0.05 -> 95% PI; alpha=0.01 -> 99% PI.
    Lower is better.
    """
    s_flat, y_flat = _flatten_windows(samples, target)
    lo = np.quantile(s_flat, alpha / 2.0, axis=1)
    hi = np.quantile(s_flat, 1.0 - alpha / 2.0, axis=1)
    width = hi - lo
    under = (lo - y_flat) * (y_flat < lo) * (2.0 / alpha)
    over = (y_flat - hi) * (y_flat > hi) * (2.0 / alpha)
    return float((width + under + over).mean())


def interval_coverage(samples, target, level):
    """Empirical coverage of the central `level` prediction interval."""
    alpha = 1.0 - level
    s_flat, y_flat = _flatten_windows(samples, target)
    lo = np.quantile(s_flat, alpha / 2.0, axis=1)
    hi = np.quantile(s_flat, 1.0 - alpha / 2.0, axis=1)
    return float(((y_flat >= lo) & (y_flat <= hi)).mean())


def exceedance_brier(samples, target, threshold):
    """Brier score on P(Y > threshold)."""
    s_flat, y_flat = _flatten_windows(samples, target)
    if np.ndim(threshold) == 0:
        t_flat = np.full_like(y_flat, threshold, dtype=np.float64)
    else:
        t_flat = np.asarray(threshold).reshape(-1).astype(np.float64)
    p_hat = (s_flat > t_flat[:, None]).mean(axis=1)
    y_bin = (y_flat > t_flat).astype(np.float64)
    return float(((p_hat - y_bin) ** 2).mean())


def compute_all_tail_metrics(samples, target, thresholds=(0.9, 0.95, 0.99),
                              quantiles=(0.9, 0.95, 0.99),
                              train_thresholds=None):
    """
    Convenience wrapper.

    Upper-tail chaining function `v(x) = max(x, t)` requires a *fixed*
    threshold `t`. For propriety of the twCRPS scoring rule (Gneiting &
    Ranjan 2011), the threshold must not depend on the evaluation target.
    Supply `train_thresholds` as a dict {0.90: t1, 0.95: t2, 0.99: t3}
    where each `ti` is the absolute threshold computed on the TRAINING
    set at marginal quantile `i`. These are then used uniformly across
    every test window.

    Legacy fallback: if `train_thresholds` is not provided, thresholds
    are estimated from the current `target` array. This is NOT strictly
    proper and should be used only for quick exploration; production
    numbers must pass `train_thresholds`.

    Returns a dict with:
      crps, twCRPS_q090, twCRPS_q095, twCRPS_q099,
      qloss_q090, qloss_q095, qloss_q099,
      wqloss_q090, wqloss_q095, wqloss_q099,
      winkler_90, winkler_95, winkler_99,
      coverage_90, coverage_95, coverage_99,
      brier_q090, brier_q095, brier_q099.
    Each twCRPS/brier entry also has a suffix `_threshold` giving the
    absolute threshold used.
    """
    out = {"crps": crps_sample(samples, target)}
    y_flat = target.reshape(-1)
    for q in thresholds:
        if train_thresholds is not None and q in train_thresholds:
            t = float(train_thresholds[q])
            src = "train"
        else:
            t = float(np.quantile(y_flat, q))
            src = "test"  # legacy / improper
        out[f"twCRPS_q{int(q*100):03d}"] = tw_crps_sample(samples, target, t, side="upper")
        out[f"twCRPS_q{int(q*100):03d}_threshold"] = t
        out[f"twCRPS_q{int(q*100):03d}_threshold_src"] = src
        out[f"brier_q{int(q*100):03d}"] = exceedance_brier(samples, target, t)
    for q in quantiles:
        out[f"qloss_q{int(q*100):03d}"] = quantile_loss(samples, target, q)
        out[f"wqloss_q{int(q*100):03d}"] = weighted_quantile_loss(samples, target, q)
    for alpha, lvl in [(0.1, 90), (0.05, 95), (0.01, 99)]:
        out[f"winkler_{lvl}"] = winkler_score(samples, target, alpha)
        out[f"coverage_{lvl}"] = interval_coverage(samples, target, 1.0 - alpha)
    return out


def compute_train_thresholds(dataset_train, quantiles=(0.9, 0.95, 0.99)):
    """
    Compute absolute thresholds on the training distribution for each
    quantile level. Operates on the pooled flattened training-target
    values — the same marginal we'd use if we had infinite training data.
    This is the correct anchor for the Gneiting & Ranjan twCRPS chaining
    function.
    """
    vals = []
    for entry in dataset_train:
        t = np.asarray(entry["target"], dtype=np.float32)
        if t.ndim == 2: t = t[0]
        vals.extend(t[np.isfinite(t)].tolist())
    vals = np.asarray(vals, dtype=np.float64)
    return {q: float(np.quantile(vals, q)) for q in quantiles}
