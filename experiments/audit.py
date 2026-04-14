"""
Critical audit of MeanFlow-TS results.

Checks:
1. Is the JVP actually doing something? (ablation: JVP vs no-JVP)
2. Is the CRPS calculation correct? (manual verification)
3. Are predictions actually reasonable? (visual + statistical inspection)
4. Does the model degenerate to something trivial? (e.g., always predict mean)
5. Is the scaling correct? (loc computation)
"""
import os, sys, math, torch, numpy as np
import torch.nn.functional as F
from copy import deepcopy

sys.path.insert(0, os.path.dirname(__file__))
from meanflow_ts.model import (
    ConditionalMeanFlowNet, conditional_meanflow_loss, sample_t_r,
    MeanFlowForecaster, SinusoidalPosEmb
)
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import Evaluator

import tempfile
try:
    import pykeops
    tmp = tempfile.mkdtemp(prefix="pykeops_build_")
    pykeops.set_build_folder(tmp)
    pykeops.clean_pykeops()
except:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
ckpt = torch.load('best_cond_meanflow.pt', map_location=device, weights_only=False)
net = ConditionalMeanFlowNet(
    pred_len=24, ctx_len=24, model_channels=128,
    num_res_blocks=4, time_emb_dim=64, dropout=0.1,
).to(device)
net.load_state_dict(ckpt['net_ema'])
net.eval()
print(f"Loaded model from epoch {ckpt['epoch']}, CRPS={ckpt['crps']:.6f}")

# Load dataset
dataset = get_dataset('electricity_nips')
print(f"Test set: {len(dataset.test)} series")

# ============================================================
# AUDIT 1: Is JVP actually changing the target?
# ============================================================
print("\n" + "="*60)
print("AUDIT 1: JVP contribution analysis")
print("="*60)

# Generate a batch
torch.manual_seed(42)
context = torch.randn(16, 24, device=device)
future = torch.randn(16, 24, device=device)

# Compute loss with full MeanFlow (JVP)
mf_loss = conditional_meanflow_loss(net, future, context)

# Compute what happens WITHOUT JVP (standard FM loss)
e = torch.randn_like(future)
B = future.shape[0]
t, r = sample_t_r(B, device)
t_bc = t.unsqueeze(-1)
r_bc = r.unsqueeze(-1)
z = (1 - t_bc) * future + t_bc * e
v = e - future
h_bc = t_bc - r_bc
u_pred = net(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)), context)
fm_loss = F.mse_loss(u_pred, v)

# Check: when r == t (instantaneous mode), target should equal v
# When r != t, target should differ from v by the JVP correction
def u_func(z, t_bc, r_bc):
    h_bc = t_bc - r_bc
    return net(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)), context)

dtdt = torch.ones_like(t_bc)
drdt = torch.zeros_like(r_bc)

with torch.amp.autocast("cuda", enabled=False):
    u_pred_jvp, dudt = torch.func.jvp(u_func, (z, t_bc, r_bc), (v, dtdt, drdt))
    u_tgt = (v - (t_bc - r_bc) * dudt).detach()

# Measure the difference between MeanFlow target and standard FM target
correction = u_tgt - v
correction_norm = correction.norm(dim=1).mean().item()
v_norm = v.norm(dim=1).mean().item()
is_instantaneous = (h_bc.abs() < 1e-6).squeeze(-1)
n_instant = is_instantaneous.sum().item()
n_average = (~is_instantaneous).sum().item()

# For instantaneous samples, correction should be ~0
if n_instant > 0:
    inst_correction = correction[is_instantaneous].norm(dim=1).mean().item()
else:
    inst_correction = float('nan')

# For average samples, correction should be nonzero
if n_average > 0:
    avg_correction = correction[~is_instantaneous].norm(dim=1).mean().item()
else:
    avg_correction = float('nan')

print(f"  MeanFlow loss: {mf_loss.item():.6f}")
print(f"  Standard FM loss (MSE to v): {fm_loss.item():.6f}")
print(f"  Instantaneous samples (r==t): {n_instant}/{B}")
print(f"  Average velocity samples (r!=t): {n_average}/{B}")
print(f"  ||v|| mean: {v_norm:.4f}")
print(f"  ||correction|| mean (all): {correction_norm:.4f}")
print(f"  ||correction|| instantaneous: {inst_correction:.4f}")
print(f"  ||correction|| average: {avg_correction:.4f}")
print(f"  Correction/velocity ratio: {correction_norm/v_norm:.4f}")

if correction_norm / v_norm < 0.01:
    print("  WARNING: JVP correction is negligible! MeanFlow may be equivalent to standard FM.")
else:
    print(f"  OK: JVP correction is {correction_norm/v_norm*100:.1f}% of velocity norm — meaningful contribution.")

# ============================================================
# AUDIT 2: Does the model actually use h (interval width)?
# ============================================================
print("\n" + "="*60)
print("AUDIT 2: Does the model use h?")
print("="*60)

z_test = torch.randn(4, 24, device=device)
ctx_test = torch.randn(4, 24, device=device)
t_test = torch.full((4,), 0.5, device=device)

# Compare output at h=0 vs h=0.5 vs h=1.0
with torch.no_grad():
    out_h0 = net(z_test, (t_test, torch.zeros(4, device=device)), ctx_test)
    out_h05 = net(z_test, (t_test, torch.full((4,), 0.5, device=device)), ctx_test)
    out_h1 = net(z_test, (t_test, torch.ones(4, device=device)), ctx_test)

diff_h0_h05 = (out_h0 - out_h05).norm().item()
diff_h0_h1 = (out_h0 - out_h1).norm().item()
out_norm = out_h0.norm().item()

print(f"  ||out(h=0) - out(h=0.5)||: {diff_h0_h05:.6f}")
print(f"  ||out(h=0) - out(h=1.0)||: {diff_h0_h1:.6f}")
print(f"  ||out(h=0)||: {out_norm:.6f}")
print(f"  Relative change h=0→0.5: {diff_h0_h05/out_norm*100:.2f}%")
print(f"  Relative change h=0→1.0: {diff_h0_h1/out_norm*100:.2f}%")

if diff_h0_h1 / out_norm < 0.01:
    print("  WARNING: Output barely changes with h! The network may be ignoring h.")
else:
    print("  OK: Network output depends on h.")

# ============================================================
# AUDIT 3: Manual CRPS verification
# ============================================================
print("\n" + "="*60)
print("AUDIT 3: Manual CRPS verification")
print("="*60)

# Take a few test series and manually compute CRPS
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.transform import (
    AddObservedValuesIndicator, AddTimeFeatures, AsNumpyArray,
    Chain, InstanceSplitter, TestSplitSampler,
)
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify

transformation = Chain([
    AsNumpyArray(field="target", expected_ndim=1),
    AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
    AddTimeFeatures(
        start_field="start", target_field="target", output_field="time_feat",
        time_features=time_features_from_frequency_str("H"),
        pred_length=24,
    ),
])

test_splitter = InstanceSplitter(
    target_field="target",
    is_pad_field="is_pad",
    start_field="start",
    forecast_start_field="forecast_start",
    instance_sampler=TestSplitSampler(),
    past_length=24 + 672,
    future_length=24,
    time_series_fields=["time_feat", "observed_values"],
)

test_data = transformation.apply(dataset.test, is_train=False)

# Get first 10 test instances
loader = InferenceDataLoader(
    test_data,
    transform=test_splitter,
    batch_size=10,
    stack_fn=batchify,
)
batch = next(iter(loader))

past_target = batch["past_target"].to(device)
future_target = batch.get("future_target", torch.zeros(0)).to(device)
B = past_target.shape[0]

# For test data, future_target may be empty. Extract it from the raw series instead.
# The past_target includes the full history. The last 24 values of past_target ARE the context,
# and the actual future is not in the batch (held out by GluonTS).
# Let's use the training data instead where we have both past and future.
from gluonts.dataset.loader import TrainDataLoader
from gluonts.transform import ExpectedNumInstanceSampler
from gluonts.itertools import Cached

train_splitter = InstanceSplitter(
    target_field="target",
    is_pad_field="is_pad",
    start_field="start",
    forecast_start_field="forecast_start",
    instance_sampler=ExpectedNumInstanceSampler(num_instances=1, min_future=24),
    past_length=24 + 672,
    future_length=24,
    time_series_fields=["time_feat", "observed_values"],
)
train_data = transformation.apply(dataset.train, is_train=True)
train_loader = TrainDataLoader(
    Cached(train_data), batch_size=10, stack_fn=batchify,
    transform=train_splitter, num_batches_per_epoch=1,
)
batch = next(iter(train_loader))
past_target = batch["past_target"].to(device)
future_target = batch["future_target"].to(device)
B = past_target.shape[0]

# Generate predictions
context = past_target[:, -24:]
loc = context.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
scaled_ctx = context / loc

print(f"\n  Sample predictions for {B} train series:")
print(f"  Context range: [{context.min().item():.2f}, {context.max().item():.2f}]")
print(f"  Future range: [{future_target.min().item():.2f}, {future_target.max().item():.2f}]")
print(f"  Scale (loc) range: [{loc.min().item():.4f}, {loc.max().item():.4f}]")

# Generate 50 forecast samples
samples = []
with torch.no_grad():
    for _ in range(50):
        z_1 = torch.randn(B, 24, device=device)
        t = torch.ones(B, device=device)
        h = torch.ones(B, device=device)
        u = net(z_1, (t, h), scaled_ctx)
        pred = (z_1 - u) * loc
        samples.append(pred)
samples = torch.stack(samples, dim=1)  # (B, 50, 24)

# Manual CRPS for each series
print(f"\n  Per-series analysis:")
print(f"  {'Series':>6} | {'True Mean':>10} | {'Pred Mean':>10} | {'Pred Std':>10} | {'MAE':>10} | {'Manual CRPS':>12}")
print(f"  {'-'*70}")

manual_crps_list = []
for i in range(B):
    truth = future_target[i].cpu().numpy()  # (24,)
    samp = samples[i].cpu().numpy()  # (50, 24)
    pred_median = np.median(samp, axis=0)

    mae = np.mean(np.abs(pred_median - truth))

    # CRPS = E|X-y| - 0.5*E|X-X'| per timestep, then average
    crps_per_step = []
    for step in range(24):
        y = truth[step]
        x = samp[:, step]
        term1 = np.mean(np.abs(x - y))
        # E|X-X'| via all pairs
        n = len(x)
        term2 = 0
        for a in range(n):
            for b in range(a+1, n):
                term2 += np.abs(x[a] - x[b])
        term2 = 2 * term2 / (n * (n-1))
        crps_per_step.append(term1 - 0.5 * term2)
    crps_i = np.mean(crps_per_step)
    manual_crps_list.append(crps_i)

    # Normalize CRPS by |truth| for comparison with wQuantileLoss
    norm_crps = crps_i / (np.abs(truth).mean() + 1e-6)

    print(f"  {i:>6} | {truth.mean():>10.3f} | {pred_median.mean():>10.3f} | {samp.std():>10.3f} | {mae:>10.3f} | {norm_crps:>12.6f}")

print(f"\n  Average normalized CRPS: {np.mean([c / (np.abs(future_target[i].cpu().numpy()).mean() + 1e-6) for i, c in enumerate(manual_crps_list)]):.6f}")

# ============================================================
# AUDIT 4: Trivial baseline comparison
# ============================================================
print("\n" + "="*60)
print("AUDIT 4: Trivial baseline comparison")
print("="*60)

# Baseline 1: Predict last value (persistence)
persist_pred = context[:, -1:].expand_as(future_target)
persist_mae = (persist_pred - future_target).abs().mean().item()

# Baseline 2: Predict context mean
mean_pred = context.mean(dim=1, keepdim=True).expand_as(future_target)
mean_mae = (mean_pred - future_target).abs().mean().item()

# Baseline 3: Random noise
noise_pred = torch.randn_like(future_target) * context.std(dim=1, keepdim=True)
noise_mae = (noise_pred - future_target).abs().mean().item()

# Our model
our_median = samples.median(dim=1).values
our_mae = (our_median - future_target).abs().mean().item()

print(f"  Persistence (last value): MAE = {persist_mae:.4f}")
print(f"  Context mean:            MAE = {mean_mae:.4f}")
print(f"  Random noise:            MAE = {noise_mae:.4f}")
print(f"  MeanFlow (1-step):       MAE = {our_mae:.4f}")

if our_mae >= persist_mae:
    print("  WARNING: MeanFlow is worse than persistence! Model may not be learning.")
elif our_mae >= mean_mae:
    print("  WARNING: MeanFlow is worse than predicting the mean!")
else:
    print("  OK: MeanFlow beats trivial baselines.")

# ============================================================
# AUDIT 5: Does prediction vary with different noise seeds?
# ============================================================
print("\n" + "="*60)
print("AUDIT 5: Sample diversity check")
print("="*60)

with torch.no_grad():
    pred1 = []
    for _ in range(10):
        z = torch.randn(1, 24, device=device)
        u = net(z, (torch.ones(1, device=device), torch.ones(1, device=device)), scaled_ctx[:1])
        pred1.append((z - u).squeeze(0))
    pred1 = torch.stack(pred1)  # (10, 24)

spread = pred1.std(dim=0).mean().item()
mean_val = pred1.mean().abs().mean().item()
print(f"  Mean of predictions: {pred1.mean(dim=0).mean().item():.4f}")
print(f"  Std across 10 samples: {spread:.4f}")
print(f"  Std / |Mean| ratio: {spread / (mean_val + 1e-6):.4f}")

if spread < 1e-3:
    print("  WARNING: All samples are nearly identical! Model may have collapsed.")
else:
    print("  OK: Samples show diversity.")

# ============================================================
# AUDIT 6: Check that context actually influences prediction
# ============================================================
print("\n" + "="*60)
print("AUDIT 6: Context influence check")
print("="*60)

with torch.no_grad():
    z_fixed = torch.randn(1, 24, device=device)
    t_one = torch.ones(1, device=device)

    ctx_a = scaled_ctx[:1]  # real context
    ctx_b = torch.zeros(1, 24, device=device)  # zero context
    ctx_c = torch.ones(1, 24, device=device) * 5  # high context

    pred_a = z_fixed - net(z_fixed, (t_one, t_one), ctx_a)
    pred_b = z_fixed - net(z_fixed, (t_one, t_one), ctx_b)
    pred_c = z_fixed - net(z_fixed, (t_one, t_one), ctx_c)

diff_ab = (pred_a - pred_b).norm().item()
diff_ac = (pred_a - pred_c).norm().item()
pred_norm = pred_a.norm().item()

print(f"  ||pred(real_ctx) - pred(zero_ctx)||: {diff_ab:.4f}")
print(f"  ||pred(real_ctx) - pred(high_ctx)||: {diff_ac:.4f}")
print(f"  ||pred(real_ctx)||: {pred_norm:.4f}")
print(f"  Context influence (zero): {diff_ab/pred_norm*100:.1f}%")
print(f"  Context influence (high): {diff_ac/pred_norm*100:.1f}%")

if diff_ab / pred_norm < 0.05:
    print("  WARNING: Context has minimal influence on prediction!")
else:
    print("  OK: Context significantly influences prediction.")

print("\n" + "="*60)
print("AUDIT COMPLETE")
print("="*60)
