"""
Train unconditional MeanFlow for Table 1 (2-Wasserstein) and Table 2 (LPS) evaluation.

Generates full synthetic time series (no context conditioning).
Usage: python train_unconditional.py <dataset_name> --epochs 1200
"""
import os, sys, time, math, argparse, logging, tempfile
import numpy as np
import torch
from torch.optim import AdamW
from copy import deepcopy
from sklearn.preprocessing import StandardScaler

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

try:
    import pykeops
    tmp = tempfile.mkdtemp(prefix="pykeops_build_")
    pykeops.set_build_folder(tmp)
    pykeops.clean_pykeops()
except: pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model import UnconditionalMeanFlowNet, unconditional_meanflow_loss, meanflow_sample

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIGS = {
    "electricity_nips": {"freq": "H", "seq_len": 24},
    "solar_nips":       {"freq": "H", "seq_len": 24},
    "traffic_nips":     {"freq": "H", "seq_len": 24},
    "exchange_rate_nips": {"freq": "B", "seq_len": 30},
    "m4_hourly":        {"freq": "H", "seq_len": 48},
}


def load_train_windows(dataset_name, seq_len):
    """Load dataset and extract non-overlapping training windows."""
    dataset = get_dataset(dataset_name)
    windows = []
    for entry in dataset.train:
        ts = np.array(entry["target"], dtype=np.float32)
        n_windows = len(ts) // seq_len
        for i in range(n_windows):
            windows.append(ts[i * seq_len : (i + 1) * seq_len])
    windows = np.stack(windows)
    logger.info(f"Extracted {len(windows)} windows of length {seq_len}")
    return windows, dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--epochs", type=int, default=1200)
    args = parser.parse_args()

    name = args.dataset
    cfg = CONFIGS[name]
    seq_len = cfg["seq_len"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(6432)
    np.random.seed(6432)

    logger.info(f"=== Unconditional MeanFlow: {name} (seq_len={seq_len}) ===")

    # Load data
    windows, dataset = load_train_windows(name, seq_len)

    # Normalize
    scaler = StandardScaler()
    flat = windows.reshape(-1, 1)
    scaler.fit(flat)
    windows_scaled = scaler.transform(flat).reshape(windows.shape)
    train_tensor = torch.tensor(windows_scaled, dtype=torch.float32, device=device)
    logger.info(f"Data range: [{train_tensor.min():.2f}, {train_tensor.max():.2f}]")

    # Model
    net = UnconditionalMeanFlowNet(
        seq_len=seq_len, model_channels=128, num_res_blocks=4,
        time_emb_dim=64, dropout=0.1,
    ).to(device)
    net_ema = deepcopy(net).eval()
    logger.info(f"Params: {sum(p.numel() for p in net.parameters()):,}")

    optimizer = AdamW(net.parameters(), lr=6e-4)
    batch_size = 16384  # ~32GB GPU, run 1 dataset at a time
    batches_per_epoch = 8   # 8 * 16384 = 131K samples/epoch, 8 gradient steps

    for epoch in range(args.epochs):
        net.train()
        epoch_loss = 0
        for _ in range(batches_per_epoch):
            idx = torch.randint(0, len(train_tensor), (batch_size,))
            x = train_tensor[idx]

            loss = unconditional_meanflow_loss(net, x)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                for p, pe in zip(net.parameters(), net_ema.parameters()):
                    pe.data.lerp_(p.data, 1e-4)
            epoch_loss += loss.item()
        epoch_loss /= batches_per_epoch

        if (epoch + 1) % 100 == 0 or epoch == 0:
            # Generate samples and compute basic stats
            net_ema.eval()
            samples = meanflow_sample(net_ema, (512, seq_len), device)
            s_mean, s_std = samples.mean().item(), samples.std().item()
            r_mean, r_std = train_tensor.mean().item(), train_tensor.std().item()
            logger.info(f"Epoch {epoch+1:>4} | Loss: {epoch_loss:.4f} | "
                        f"Gen: mean={s_mean:.3f} std={s_std:.3f} | "
                        f"Real: mean={r_mean:.3f} std={r_std:.3f}")

        if (epoch + 1) % 300 == 0 or (epoch + 1) == args.epochs:
            # Save checkpoint
            torch.save({
                'net_ema': net_ema.state_dict(),
                'scaler_mean': scaler.mean_.tolist(),
                'scaler_scale': scaler.scale_.tolist(),
                'epoch': epoch + 1,
                'seq_len': seq_len,
            }, f'uncond_meanflow_{name}.pt')
            logger.info(f"Saved uncond_meanflow_{name}.pt")

    logger.info("Done.")


if __name__ == "__main__":
    main()
