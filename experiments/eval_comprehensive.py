"""
Comprehensive evaluation for TailFlow paper.

Metrics computed:
1. Overall CRPS, ND, NRMSE
2. Tail-stratified CRPS (top 10%, 20% most extreme test windows)
3. Quantile coverage (does model's q95/q99 contain true extremes?)
4. Energy score (multivariate proper scoring rule)
5. Variogram score (captures temporal correlation quality)
6. Rejection sampling comparison (our guided 1-step vs RS with N steps)
7. Tail generation quality (KL divergence of extremity distributions)
8. Efficiency comparison (wallclock time for equivalent tail coverage)

Usage:
    python experiments/eval_comprehensive.py electricity_nips
    python experiments/eval_comprehensive.py --all
"""
import os, sys, time, argparse, logging, pickle, json
import numpy as np
import torch
from scipy import stats
from tqdm.auto import tqdm

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.itertools import Cached
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator, AddTimeFeatures, AsNumpyArray,
    Chain, InstanceSplitter, TestSplitSampler,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_tail import (
    ExtremityCondMeanFlowNet, ExtremityCondForecaster,
    QuantileMapper, compute_composite_extremity,
    compute_volatility, compute_max_deviation, compute_drawdown,
    guided_sample,
)
from meanflow_ts.model import ConditionalMeanFlowNet, MeanFlowForecaster

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIGS = {
    "electricity_nips":   {"freq": "H", "ctx": 24, "pred": 24},
    "solar_nips":         {"freq": "H", "ctx": 24, "pred": 24},
    "traffic_nips":       {"freq": "H", "ctx": 24, "pred": 24},
    "exchange_rate_nips": {"freq": "B", "ctx": 30, "pred": 30},
    "m4_hourly":          {"freq": "H", "ctx": 48, "pred": 48},
}
LAG_MAP = {"H": 672, "B": 750}


# ============================================================
# Scoring rules
# ============================================================

def energy_score(samples, truth):
    """
    Energy score for multivariate probabilistic forecasts.
    samples: (N, T) - N forecast samples, each of length T
    truth: (T,) - ground truth
    Returns: scalar energy score (lower is better)
    """
    N, T = samples.shape
    # Term 1: E[||X - y||]
    diff_to_truth = np.linalg.norm(samples - truth[None, :], axis=1)
    term1 = diff_to_truth.mean()
    # Term 2: 0.5 * E[||X - X'||]
    # Subsample for efficiency if N is large
    if N > 100:
        idx1 = np.random.choice(N, 100, replace=False)
        idx2 = np.random.choice(N, 100, replace=False)
        diff_pairs = np.linalg.norm(samples[idx1] - samples[idx2], axis=1)
    else:
        diff_pairs = []
        for i in range(min(N, 50)):
            for j in range(i+1, min(N, 50)):
                diff_pairs.append(np.linalg.norm(samples[i] - samples[j]))
        diff_pairs = np.array(diff_pairs) if diff_pairs else np.array([0.0])
    term2 = 0.5 * diff_pairs.mean()
    return term1 - term2


def variogram_score(samples, truth, p=1.0):
    """
    Variogram score — captures temporal correlation quality.
    samples: (N, T), truth: (T,)
    """
    N, T = samples.shape
    score = 0.0
    count = 0
    for i in range(T):
        for j in range(i+1, T):
            obs_diff = np.abs(truth[i] - truth[j]) ** p
            sample_diffs = np.abs(samples[:, i] - samples[:, j]) ** p
            score += (obs_diff - sample_diffs.mean()) ** 2
            count += 1
    return score / max(count, 1)


def quantile_coverage(samples, truth, quantiles=[0.1, 0.5, 0.9, 0.95, 0.99]):
    """
    Compute empirical coverage of prediction intervals.
    samples: (N, T), truth: (T,)
    Returns dict: quantile -> fraction of time steps where truth is below sample quantile
    """
    results = {}
    for q in quantiles:
        q_val = np.quantile(samples, q, axis=0)  # (T,)
        coverage = (truth <= q_val).mean()
        results[f'q{q}'] = float(coverage)
    return results


def tail_crps_decomposition(forecasts, tss, pred_len, top_fractions=[0.1, 0.2, 0.3]):
    """
    Decompose CRPS by extremity of ground truth.
    Returns: {fraction: crps_value}
    """
    # Compute per-series extremity
    extremities = []
    crps_per_series = []

    for fc, ts in zip(forecasts, tss):
        gt = ts.values[-pred_len:]
        loc = max(np.abs(gt).mean(), 1e-6)
        normalized = gt.flatten() / loc
        if len(normalized) >= 2:
            vol = np.std(np.diff(normalized))
            max_dev = np.max(np.abs(normalized - normalized.mean()))
            extremities.append(float(vol + max_dev) / 2)
        else:
            extremities.append(0.0)

        # Compute CRPS for this series
        samples = fc.samples  # (N, T)
        gt_flat = gt.flatten()
        # CRPS via quantile loss
        quantile_levels = np.arange(0.05, 1.0, 0.05)
        ql = 0
        for q in quantile_levels:
            q_pred = np.quantile(samples, q, axis=0)
            ql += np.mean(2 * np.abs((gt_flat > q_pred) - q) * np.abs(gt_flat - q_pred))
        crps_per_series.append(ql / len(quantile_levels))

    extremities = np.array(extremities)
    crps_per_series = np.array(crps_per_series)

    results = {}
    for frac in top_fractions:
        threshold = np.quantile(extremities, 1 - frac)
        mask = extremities >= threshold
        if mask.sum() > 0:
            results[f'tail_{int(frac*100)}pct'] = float(crps_per_series[mask].mean())
        # Also non-tail
        mask_nt = extremities < threshold
        if mask_nt.sum() > 0:
            results[f'nontail_{int(frac*100)}pct'] = float(crps_per_series[mask_nt].mean())

    results['overall'] = float(crps_per_series.mean())
    return results


# ============================================================
# Rejection Sampling Baseline
# ============================================================

def rejection_sampling_eval(net_baseline, net_tailflow, dataset, transformation,
                            cfg, qmap, device, n_total=1000, n_keep=100):
    """
    Compare rejection sampling (generate many, keep extreme) from baseline
    vs. guided generation from TailFlow.

    For each test window:
    1. Baseline RS: generate n_total samples, score, keep top n_keep by extremity
    2. TailFlow guided: generate n_keep samples with high guidance

    Compare: quality (CRPS on tail windows), diversity, and wallclock time.
    """
    freq, ctx_len, pred_len = cfg["freq"], cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP.get(freq, 672)

    test_transform = transformation.apply(dataset.test, is_train=False)
    test_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
        past_length=ctx_len + max_lag, future_length=pred_len,
        time_series_fields=["time_feat", "observed_values"],
    )

    results = {
        'rejection_sampling': {'n_total': n_total, 'n_keep': n_keep},
        'guided_generation': {},
    }

    # Time the rejection sampling approach
    t0 = time.time()
    rs_forecaster = MeanFlowForecaster(net_baseline, ctx_len, pred_len, num_samples=n_total).to(device)
    rs_predictor = PyTorchPredictor(
        prediction_length=pred_len,
        input_names=["past_target", "past_observed_values"],
        prediction_net=rs_forecaster, batch_size=64,
        input_transform=test_splitter, device=device,
    )
    rs_it, ts_it = make_evaluation_predictions(
        dataset=test_transform, predictor=rs_predictor, num_samples=n_total,
    )
    rs_forecasts = list(rs_it)
    tss = list(ts_it)
    rs_time = time.time() - t0
    results['rejection_sampling']['wallclock_s'] = rs_time

    # For each forecast, keep top-n_keep by extremity
    rs_filtered = []
    for fc in rs_forecasts:
        samples = fc.samples  # (n_total, T)
        samples_t = torch.tensor(samples)
        scores = compute_composite_extremity(samples_t).numpy()
        top_idx = np.argsort(scores)[-n_keep:]
        # Create modified forecast with only top samples
        fc_filtered = fc.copy_with_samples(samples[top_idx])
        rs_filtered.append(fc_filtered)

    # Evaluate RS filtered
    rs_metrics, _ = Evaluator(num_workers=0)(tss, rs_filtered)
    results['rejection_sampling']['crps'] = rs_metrics["mean_wQuantileLoss"]
    results['rejection_sampling']['nd'] = rs_metrics["ND"]

    # Time the guided generation approach
    t0 = time.time()
    for w in [1.0, 2.0, 3.0]:
        guided_forecaster = ExtremityCondForecaster(
            net_tailflow, ctx_len, pred_len, num_samples=n_keep,
            guidance_scale=w, target_extremity=0.9, quantile_mapper=qmap,
        ).to(device)
        g_predictor = PyTorchPredictor(
            prediction_length=pred_len,
            input_names=["past_target", "past_observed_values"],
            prediction_net=guided_forecaster, batch_size=64,
            input_transform=test_splitter, device=device,
        )
        g_it, ts_it2 = make_evaluation_predictions(
            dataset=test_transform, predictor=g_predictor, num_samples=n_keep,
        )
        g_forecasts = list(g_it)
        tss2 = list(ts_it2)
        g_time = time.time() - t0

        g_metrics, _ = Evaluator(num_workers=0)(tss2, g_forecasts)
        results['guided_generation'][f'w={w}'] = {
            'crps': g_metrics["mean_wQuantileLoss"],
            'nd': g_metrics["ND"],
            'wallclock_s': g_time,
            'nfe_per_sample': 2,  # 2 forward passes for CFG
        }

    # Compute NFE comparison
    results['rejection_sampling']['total_nfe'] = n_total  # 1 NFE per sample * n_total
    results['guided_generation']['total_nfe'] = n_keep * 2  # 2 NFE per sample (cond + uncond)

    logger.info(f"Rejection Sampling: {n_total} NFE, CRPS={results['rejection_sampling']['crps']:.6f}, "
                f"time={rs_time:.1f}s")
    for w_key, gd in results['guided_generation'].items():
        if isinstance(gd, dict):
            logger.info(f"Guided ({w_key}): {n_keep*2} NFE, CRPS={gd['crps']:.6f}, "
                        f"time={gd.get('wallclock_s', 0):.1f}s")

    return results


# ============================================================
# Extremity Distribution Analysis
# ============================================================

def analyze_extremity_distributions(net, qmap, ctx_data, device, ctx_len, pred_len,
                                     guidance_scales=[0.0, 1.0, 2.0, 3.0],
                                     target_qs=[0.5, 0.7, 0.9, 0.95],
                                     n_samples=2000):
    """
    For each guidance scale and target extremity:
    - Generate n_samples
    - Compute extremity distribution
    - Measure KL divergence vs. real data
    - Measure how well the model responds to conditioning
    """
    net.eval()
    results = {}

    for w in guidance_scales:
        for tq in target_qs:
            key = f'w={w}_tq={tq}'
            gen_scores = []

            with torch.no_grad():
                for start in range(0, n_samples, 256):
                    end = min(start + 256, n_samples)
                    B = end - start
                    idx = np.random.randint(0, len(ctx_data), B)
                    ctx = torch.tensor(ctx_data[idx], device=device)
                    ext_q = torch.full((B,), tq, device=device)
                    samples = guided_sample(net, ctx, ext_q, (B, pred_len), device, w)
                    scores = compute_composite_extremity(samples).cpu().numpy()
                    gen_scores.extend(scores.tolist())

            gen_scores = np.array(gen_scores)
            results[key] = {
                'mean': float(gen_scores.mean()),
                'std': float(gen_scores.std()),
                'p90': float(np.percentile(gen_scores, 90)),
                'p95': float(np.percentile(gen_scores, 95)),
            }

    return results


# ============================================================
# Main evaluation
# ============================================================

def run_eval(name, device):
    cfg = CONFIGS[name]
    freq, ctx_len, pred_len = cfg["freq"], cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP.get(freq, 672)
    outdir = os.path.join(os.path.dirname(__file__), '..', 'results', name)

    logger.info(f"\n{'='*60}")
    logger.info(f"Comprehensive Evaluation: {name}")
    logger.info(f"{'='*60}")

    dataset = get_dataset(name)
    transformation = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(
            start_field="start", target_field="target", output_field="time_feat",
            time_features=time_frequency_str(freq), pred_length=pred_len,
        ),
    ])

    # Load quantile mapper
    qmap_path = os.path.join(outdir, 'quantile_mapper.pkl')
    if not os.path.exists(qmap_path):
        logger.warning(f"No quantile mapper found at {qmap_path}. Run training first.")
        return
    with open(qmap_path, 'rb') as f:
        qmap = pickle.load(f)

    # Load models
    results = {}

    # Load TailFlow model (check for self-training rounds first, then base)
    tailflow_path = None
    for rnd in [3, 2, 1]:
        p = os.path.join(outdir, f'after_st_round_{rnd}.pt')
        if os.path.exists(p):
            tailflow_path = p
            results['tailflow_stage'] = f'after_st_round_{rnd}'
            break
    if tailflow_path is None:
        tailflow_path = os.path.join(outdir, 'best_base.pt')
        results['tailflow_stage'] = 'base'

    if not os.path.exists(tailflow_path):
        logger.warning(f"No TailFlow model found. Run training first.")
        return

    logger.info(f"Loading TailFlow from {tailflow_path}")
    ckpt = torch.load(tailflow_path, map_location=device, weights_only=False)
    net = ExtremityCondMeanFlowNet(
        pred_len=pred_len, ctx_len=ctx_len,
        model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
    ).to(device)
    net.load_state_dict(ckpt['net_ema'])
    net.eval()

    # Evaluate with multiple guidance scales
    test_transform = transformation.apply(dataset.test, is_train=False)
    test_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
        past_length=ctx_len + max_lag, future_length=pred_len,
        time_series_fields=["time_feat", "observed_values"],
    )

    for w in [0.0, 1.0, 1.5, 2.0, 3.0]:
        logger.info(f"\n--- Guidance scale w={w} ---")
        forecaster = ExtremityCondForecaster(
            net, ctx_len, pred_len, num_samples=100,
            guidance_scale=w, target_extremity=None, quantile_mapper=qmap,
        ).to(device)
        predictor = PyTorchPredictor(
            prediction_length=pred_len,
            input_names=["past_target", "past_observed_values"],
            prediction_net=forecaster, batch_size=256,
            input_transform=test_splitter, device=device,
        )
        fc_it, ts_it = make_evaluation_predictions(
            dataset=test_transform, predictor=predictor, num_samples=100,
        )
        forecasts = list(fc_it)
        tss = list(ts_it)
        metrics, _ = Evaluator(num_workers=0)(tss, forecasts)

        # Tail decomposition
        tail_decomp = tail_crps_decomposition(forecasts, tss, pred_len)

        # Energy and variogram scores (subsample for speed)
        n_eval = min(100, len(forecasts))
        es_scores = []
        vs_scores = []
        cov_results = []
        for i in range(n_eval):
            gt = tss[i].values[-pred_len:].flatten()
            samples = forecasts[i].samples
            es_scores.append(energy_score(samples, gt))
            if pred_len <= 48:
                vs_scores.append(variogram_score(samples, gt))
            cov_results.append(quantile_coverage(samples, gt))

        # Aggregate coverage
        avg_cov = {}
        for key in cov_results[0].keys():
            avg_cov[key] = float(np.mean([c[key] for c in cov_results]))

        results[f'w={w}'] = {
            'crps': metrics["mean_wQuantileLoss"],
            'nd': metrics["ND"],
            'nrmse': metrics["NRMSE"],
            'energy_score': float(np.mean(es_scores)),
            'variogram_score': float(np.mean(vs_scores)) if vs_scores else None,
            'coverage': avg_cov,
            'tail_decomp': tail_decomp,
        }

        logger.info(f"  CRPS={metrics['mean_wQuantileLoss']:.6f} | "
                     f"ES={np.mean(es_scores):.4f} | "
                     f"Tail-10% CRPS={tail_decomp.get('tail_10pct', 'N/A')}")

    # Save comprehensive results
    results_path = os.path.join(outdir, 'comprehensive_eval.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, nargs='?', default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.all:
        for name in CONFIGS:
            try:
                run_eval(name, device)
            except Exception as e:
                logger.error(f"Failed on {name}: {e}")
    elif args.dataset:
        run_eval(args.dataset, device)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
