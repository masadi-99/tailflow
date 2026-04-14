"""
Resume from Phase 2 checkpoints: run self-training + comprehensive eval + ablation.

For each dataset:
1. Load Phase 2 checkpoint (conditioned model)
2. Run self-training rounds
3. Comprehensive evaluation with guidance sweep + tail metrics
4. Ablation: extended Phase 1 training (no adapter) at lower LR

Usage:
    python experiments/resume_and_eval.py electricity_nips
    python experiments/resume_and_eval.py --all
"""
import os, sys, time, argparse, logging, pickle, json, gc
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model import (
    ConditionalMeanFlowNet, conditional_meanflow_loss, MeanFlowForecaster, sample_t_r,
)
from meanflow_ts.model_tail import (
    QuantileMapper, compute_raw_extremity, tilted_resampling,
)
# Import from train_tail_v2
from experiments.train_tail_v2 import (
    ConditionedMeanFlowNet, conditioned_meanflow_loss, guided_sample,
    TailFlowForecaster, evaluate, tail_stratified_crps,
    CONFIGS, LAG_MAP, TSFLOW_CRPS,
)

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_phase2_model(outdir, cfg, device):
    """Load Phase 2 conditioned model from checkpoint."""
    ctx_len, pred_len = cfg["ctx"], cfg["pred"]
    ckpt_path = os.path.join(outdir, 'phase2_best.pt')
    if not os.path.exists(ckpt_path):
        return None, None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    base_net = ConditionalMeanFlowNet(
        pred_len=pred_len, ctx_len=ctx_len,
        model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
    ).to(device)
    cond_net = ConditionedMeanFlowNet(base_net, cfg_drop_prob=0.2).to(device)
    cond_net.load_state_dict(ckpt['cond_net'])
    cond_ema = deepcopy(cond_net).eval()
    cond_ema.load_state_dict(ckpt['cond_ema'])
    logger.info(f"Loaded Phase 2 model from {ckpt_path} (CRPS={ckpt.get('crps', '?')})")
    return cond_net, cond_ema


def load_phase1_model(outdir, cfg, device):
    """Load Phase 1 base model from checkpoint."""
    ctx_len, pred_len = cfg["ctx"], cfg["pred"]
    ckpt_path = os.path.join(outdir, 'phase1_best.pt')
    if not os.path.exists(ckpt_path):
        return None, None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = ConditionalMeanFlowNet(
        pred_len=pred_len, ctx_len=ctx_len,
        model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
    ).to(device)
    net.load_state_dict(ckpt['net'])
    ema = deepcopy(net).eval()
    ema.load_state_dict(ckpt['net_ema'])
    logger.info(f"Loaded Phase 1 model from {ckpt_path} (CRPS={ckpt.get('crps', '?')})")
    return net, ema


def run_dataset(name, device, self_train_rounds=2, self_train_epochs=80,
                ablation_epochs=200, num_eval_samples=100, alpha=2.0):
    cfg = CONFIGS[name]
    freq, ctx_len, pred_len = cfg["freq"], cfg["ctx"], cfg["pred"]
    max_lag = LAG_MAP.get(freq, 672)
    outdir = os.path.join(os.path.dirname(__file__), '..', 'results', name)

    logger.info(f"\n{'='*60}")
    logger.info(f"Resume: {name} | ctx={ctx_len} pred={pred_len}")
    logger.info(f"{'='*60}")

    dataset = get_dataset(name)
    transformation = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(
            start_field="start", target_field="target", output_field="time_feat",
            time_features=time_features_from_frequency_str(freq), pred_length=pred_len,
        ),
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
        Cached(transformed_data), batch_size=128, stack_fn=batchify,
        transform=train_splitter, num_batches_per_epoch=32, shuffle_buffer_length=5000,
    )

    # Load quantile mapper
    qmap_path = os.path.join(outdir, 'quantile_mapper.pkl')
    with open(qmap_path, 'rb') as f:
        qmap = pickle.load(f)

    results = {'dataset': name}

    # ============================================================
    # Part A: Self-training from Phase 2 checkpoint
    # ============================================================
    cond_net, cond_ema = load_phase2_model(outdir, cfg, device)
    if cond_net is None:
        logger.warning(f"No Phase 2 checkpoint for {name}, skipping self-training")
    else:
        # Evaluate Phase 2 baseline first
        logger.info("\n--- Phase 2 baseline evaluation ---")
        p2_results = {}
        for tq in [0.5, 'marginal']:
            m, fcs, tss, _ = evaluate(cond_ema, dataset, transformation, cfg, device,
                                       num_samples=num_eval_samples, guidance_scale=1.0,
                                       target_extremity=tq, label=f"P2-{tq}")
            td = tail_stratified_crps(fcs, tss, pred_len)
            p2_results[f'tq={tq}_w=1'] = {'crps': m["mean_wQuantileLoss"], 'nd': m["ND"],
                                            'nrmse': m["NRMSE"], 'tail_decomp': td}
        results['phase2_baseline'] = p2_results

        # Self-training rounds
        for round_num in range(1, self_train_rounds + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Self-Training Round {round_num}")
            logger.info(f"{'='*50}")

            # Collect original data
            orig = {'ctx': [], 'fut': [], 'loc': [], 'eq': []}
            cond_ema.eval()
            for batch in train_loader:
                past = batch["past_target"].to(device)
                future = batch["future_target"].to(device)
                ctx = past[:, -ctx_len:]
                loc = ctx.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
                with torch.no_grad():
                    scores = compute_raw_extremity(future / loc)
                    eq = qmap.to_quantile(scores)
                orig['ctx'].append(ctx.cpu())
                orig['fut'].append(future.cpu())
                orig['loc'].append(loc.cpu())
                orig['eq'].append(eq.cpu() if isinstance(eq, torch.Tensor) else torch.tensor(eq))

            orig_ctx = torch.cat(orig['ctx'])
            orig_fut = torch.cat(orig['fut'])
            orig_loc = torch.cat(orig['loc'])
            orig_eq = torch.cat(orig['eq'])

            # Generate synthetic data
            logger.info("Generating synthetic data...")
            syn = {'ctx': [], 'fut': [], 'loc': [], 'eq': []}
            n_gen = max(orig_ctx.shape[0] // 256, 8)
            with torch.no_grad():
                for _ in tqdm(range(n_gen), desc="Gen"):
                    B = min(256, orig_ctx.shape[0])
                    idx = torch.randint(0, orig_ctx.shape[0], (B,))
                    ctx = orig_ctx[idx].to(device)
                    loc_b = orig_loc[idx].to(device)
                    ext_t = torch.rand(B, device=device) * 0.5 + 0.5
                    samples = guided_sample(cond_ema, ctx / loc_b, ext_t,
                                            (B, pred_len), device, 1.5)
                    scores = compute_raw_extremity(samples)
                    eq = qmap.to_quantile(scores)
                    syn['ctx'].append(ctx.cpu())
                    syn['fut'].append((samples * loc_b).cpu())
                    syn['loc'].append(loc_b.cpu())
                    syn['eq'].append(eq.cpu() if isinstance(eq, torch.Tensor) else torch.tensor(eq))

            syn_ctx = torch.cat(syn['ctx'])
            syn_fut = torch.cat(syn['fut'])
            syn_loc = torch.cat(syn['loc'])
            syn_eq = torch.cat(syn['eq'])

            # Tilted resampling
            weights = torch.exp(alpha * syn_eq)
            weights = weights / weights.sum()
            n_syn = min(syn_ctx.shape[0], orig_ctx.shape[0])
            indices = torch.multinomial(weights, n_syn, replacement=True)
            n_orig = orig_ctx.shape[0] // 2
            perm = torch.randperm(orig_ctx.shape[0])[:n_orig]
            mixed_ctx = torch.cat([syn_ctx[indices], orig_ctx[perm]])
            mixed_fut = torch.cat([syn_fut[indices], orig_fut[perm]])
            mixed_loc = torch.cat([syn_loc[indices], orig_loc[perm]])
            mixed_eq = torch.cat([syn_eq[indices], orig_eq[perm]])
            logger.info(f"Mixed: {mixed_ctx.shape[0]} samples, eq_mean={mixed_eq.mean():.3f}")

            # Fine-tune
            ds = TensorDataset(mixed_ctx, mixed_fut, mixed_loc, mixed_eq)
            loader = DataLoader(ds, batch_size=128, shuffle=True, drop_last=True, num_workers=0)
            opt = AdamW(cond_net.parameters(), lr=1e-4)

            for ep in range(self_train_epochs):
                cond_net.train()
                ep_loss, nb = 0, 0
                t0 = time.time()
                for cb, fb, lb, eb in loader:
                    cb, fb, lb, eb = cb.to(device), fb.to(device), lb.to(device), eb.to(device)
                    loss = conditioned_meanflow_loss(cond_net, fb / lb, cb / lb, eb)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(cond_net.parameters(), 1.0)
                    opt.step()
                    with torch.no_grad():
                        for p, pe in zip(cond_net.parameters(), cond_ema.parameters()):
                            pe.data.lerp_(p.data, 2e-4)
                    ep_loss += loss.item(); nb += 1
                if (ep+1) % 20 == 0 or ep == 0:
                    logger.info(f"  ST-R{round_num} Ep {ep+1}/{self_train_epochs} | "
                                f"Loss: {ep_loss/nb:.4f} | {time.time()-t0:.1f}s")

            # Evaluate
            logger.info(f"Evaluating ST-R{round_num}...")
            st_res = {}
            for tq in [0.5, 'marginal', 0.8, 0.9]:
                for w in [1.0, 2.0, 3.0]:
                    key = f'tq={tq}_w={w}'
                    m, fcs, tss, _ = evaluate(cond_ema, dataset, transformation, cfg, device,
                                               num_samples=num_eval_samples, guidance_scale=w,
                                               target_extremity=tq, label=f"ST-R{round_num}-{key}")
                    td = tail_stratified_crps(fcs, tss, pred_len)
                    st_res[key] = {'crps': m["mean_wQuantileLoss"], 'nd': m["ND"],
                                    'nrmse': m["NRMSE"], 'tail_decomp': td}
            results[f'self_train_round_{round_num}'] = st_res

            torch.save({'cond_net': cond_net.state_dict(), 'cond_ema': cond_ema.state_dict(),
                        'round': round_num}, os.path.join(outdir, f'st_round_{round_num}.pt'))
            # Cleanup tensors
            del orig_ctx, orig_fut, orig_loc, orig_eq
            del syn_ctx, syn_fut, syn_loc, syn_eq
            del mixed_ctx, mixed_fut, mixed_loc, mixed_eq, ds, loader
            gc.collect(); torch.cuda.empty_cache()

    # Cleanup conditioned model before ablation
    del cond_net
    gc.collect(); torch.cuda.empty_cache()

    # ============================================================
    # Part B: Ablation — continued training WITHOUT adapter
    # ============================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"ABLATION: Continued training without adapter ({ablation_epochs} epochs)")
    logger.info(f"{'='*60}")

    abl_net, abl_ema = load_phase1_model(outdir, cfg, device)
    if abl_net is not None:
        abl_opt = AdamW(abl_net.parameters(), lr=1e-4)  # Same lower LR as Phase 2
        best_abl = float('inf')

        for epoch in range(ablation_epochs):
            abl_net.train()
            ep_loss, nb = 0, 0
            t0 = time.time()
            for batch in train_loader:
                past = batch["past_target"].to(device)
                future = batch["future_target"].to(device)
                ctx = past[:, -ctx_len:]
                loc = ctx.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
                loss = conditional_meanflow_loss(abl_net, future / loc, ctx / loc)
                abl_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(abl_net.parameters(), 1.0)
                abl_opt.step()
                with torch.no_grad():
                    for p, pe in zip(abl_net.parameters(), abl_ema.parameters()):
                        pe.data.lerp_(p.data, 2e-4)
                ep_loss += loss.item(); nb += 1

            if (epoch+1) % 20 == 0 or epoch == 0:
                logger.info(f"  Ablation Ep {epoch+1}/{ablation_epochs} | "
                            f"Loss: {ep_loss/nb:.4f} | {time.time()-t0:.1f}s")

            if (epoch+1) % 50 == 0 or (epoch+1) == ablation_epochs:
                m, fcs, tss, _ = evaluate(abl_ema, dataset, transformation, cfg, device,
                                           num_samples=16, label="ablation", is_conditioned=False)
                crps = m["mean_wQuantileLoss"]
                if crps < best_abl:
                    best_abl = crps
                logger.info(f"  Ablation CRPS={crps:.6f} | Best={best_abl:.6f}")

        # Final ablation eval with 100 samples
        m, fcs, tss, _ = evaluate(abl_ema, dataset, transformation, cfg, device,
                                   num_samples=num_eval_samples, label="ablation-final",
                                   is_conditioned=False)
        td = tail_stratified_crps(fcs, tss, pred_len)
        results['ablation_no_adapter'] = {
            'crps': m["mean_wQuantileLoss"], 'nd': m["ND"], 'nrmse': m["NRMSE"],
            'tail_decomp': td, 'epochs': ablation_epochs,
        }

    # Cleanup ablation
    del abl_net, abl_ema
    gc.collect(); torch.cuda.empty_cache()

    # ============================================================
    # Part C: Comprehensive eval using best ST checkpoint or Phase 2
    # ============================================================
    # Reload the best available conditioned model
    best_st = None
    for rnd in [2, 1]:
        st_path = os.path.join(outdir, f'st_round_{rnd}.pt')
        if os.path.exists(st_path):
            best_st = st_path
            break
    if best_st:
        logger.info(f"Loading best ST model from {best_st}")
        _, cond_ema = load_phase2_model(outdir, cfg, device)
        ckpt = torch.load(best_st, map_location=device, weights_only=False)
        cond_ema.load_state_dict(ckpt['cond_ema'])
        cond_ema.eval()
    else:
        _, cond_ema = load_phase2_model(outdir, cfg, device)

    if cond_ema is not None:
        logger.info(f"\n--- Comprehensive guidance sweep ---")
        sweep = {}
        for tq in [0.5, 'marginal', 0.3, 0.7, 0.8, 0.9, 0.95]:
            for w in [1.0, 2.0, 3.0]:
                key = f'tq={tq}_w={w}'
                m, fcs, tss, _ = evaluate(cond_ema, dataset, transformation, cfg, device,
                                           num_samples=num_eval_samples, guidance_scale=w,
                                           target_extremity=tq, label=key)
                td = tail_stratified_crps(fcs, tss, pred_len)
                sweep[key] = {'crps': m["mean_wQuantileLoss"], 'nd': m["ND"],
                               'nrmse': m["NRMSE"], 'tail_decomp': td}
                del fcs, tss; gc.collect()
        results['guidance_sweep'] = sweep

    # Save results
    out_path = os.path.join(outdir, 'resume_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info(f"SUMMARY: {name}")
    logger.info(f"{'='*70}")
    tsflow = TSFLOW_CRPS.get(name, "?")
    logger.info(f"TSFlow (32 NFE):                 CRPS={tsflow}")

    if 'ablation_no_adapter' in results:
        logger.info(f"Ablation (no adapter, 1 NFE):    CRPS={results['ablation_no_adapter']['crps']:.6f}")
    if 'phase2_baseline' in results:
        for k, v in results['phase2_baseline'].items():
            logger.info(f"Phase 2 {k} (1-2 NFE):          CRPS={v['crps']:.6f}")
    for rnd in range(1, self_train_rounds + 1):
        key = f'self_train_round_{rnd}'
        if key in results:
            st = results[key]
            for k in ['tq=0.5_w=1.0', 'tq=marginal_w=3.0']:
                if k in st:
                    logger.info(f"ST-R{rnd} {k}:            CRPS={st[k]['crps']:.6f}")
                    td = st[k].get('tail_decomp', {})
                    logger.info(f"  tail-10%={td.get('tail_10pct','N/A')}, "
                                f"tail-20%={td.get('tail_20pct','N/A')}")

    logger.info(f"Results saved to {out_path}")
    logger.info(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs='?', default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--st-rounds", type=int, default=2)
    parser.add_argument("--st-epochs", type=int, default=80)
    parser.add_argument("--ablation-epochs", type=int, default=200)
    parser.add_argument("--num-eval-samples", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(6432)
    np.random.seed(6432)

    if args.all:
        for name in ['electricity_nips', 'solar_nips', 'traffic_nips']:
            try:
                run_dataset(name, device, args.st_rounds, args.st_epochs,
                            args.ablation_epochs, args.num_eval_samples)
            except Exception as e:
                logger.error(f"Failed {name}: {e}", exc_info=True)
    elif args.dataset:
        run_dataset(args.dataset, device, args.st_rounds, args.st_epochs,
                    args.ablation_epochs, args.num_eval_samples)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
