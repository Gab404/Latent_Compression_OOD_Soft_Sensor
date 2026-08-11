"""
scripts/latent_trajectory.py
=============================
Latent trajectory plot for a β-VAE encoder (default: β=0.5).

Shows the surviving active latent dimension overlaid on the true
penicillin concentration for a single full batch, demonstrating
that the encoder has learned a "biological macro-time" axis.

Usage
-----
    PYTHONPATH=. python scripts/latent_trajectory.py
    PYTHONPATH=. python scripts/latent_trajectory.py --beta 0.5 --batch-id 5
    PYTHONPATH=. python scripts/latent_trajectory.py --beta 0.01 --batch-id 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

# ── Project root ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.autoencoder.model import BetaVAE_Raman
from src.data.dataset import RAMAN_START_COL_IDX

# ── Graphic style (mirrors the user template) ─────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 12, "figure.dpi": 300, "lines.linewidth": 2.5})

# ── Constants ──────────────────────────────────────────────────────────────────
CSV_PATH       = "./100_Batches_IndPenSim_V3.csv"
CKPT_ROOT      = Path("./outputs/beta_sweep_e2e/checkpoints")
N_RAMAN        = 2001
SG_WINDOW, SG_POLY, SG_DERIV = 15, 2, 1
PENICILLIN_COL = "Offline Penicillin concentration(P_offline:P(g L^{-1}))"
OUTPUT_DIR     = Path("./outputs/latent_trajectory")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _beta_tag(beta: float) -> str:
    if beta == 0.0:
        return "b0"
    # strip trailing zeros: 0.500 -> "b0.5", 0.050 -> "b0.05"
    return "b" + f"{beta:g}"


def load_batch(csv_path: str, batch_id: int):
    """
    Load one full batch from IndPenSim V3 CSV.

    Returns
    -------
    raman   : (T, N_RAMAN) float32 — raw Raman spectra, zero rows removed
    pen     : (T,)          float64 — interpolated penicillin concentration (g/L)
    time_h  : (T,)          float64 — fermentation time (h)
    mask    : (T,)          bool    — rows kept after zero-energy filter
    """
    import pandas as pd

    print(f"  Loading CSV: {csv_path} ...")
    df = pd.read_csv(csv_path)

    # Detect batch boundaries by time reset
    t_vals    = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid
    df["_batch_id"] = batch_ids

    batch_df = df[df["_batch_id"] == batch_id].copy().reset_index(drop=True)
    if len(batch_df) == 0:
        raise ValueError(f"Batch ID {batch_id} not found in {csv_path}")

    print(f"  Batch {batch_id}: {len(batch_df)} timesteps")

    # Raman spectra
    raman_cols = df.columns[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + N_RAMAN]
    raman_raw  = batch_df[raman_cols].values.astype(np.float32)

    # Keep only timesteps with non-zero Raman signal
    mask   = np.abs(raman_raw).sum(axis=1) > 0
    raman  = raman_raw[mask]
    time_h = batch_df["Time (h)"].values[mask]

    # Penicillin — dense-interpolate offline measurements
    pen_series = batch_df[PENICILLIN_COL].copy()
    pen_series = pen_series.interpolate(method="linear").ffill().bfill()
    pen = pen_series.values.astype(np.float64)[mask]

    print(f"  After zero-energy filter: {mask.sum()} timesteps")
    return raman, pen, time_h, mask


def preprocess(raman: np.ndarray, scaler) -> torch.Tensor:
    """SG derivative → StandardScaler → torch tensor (N, 1, L)."""
    sg = savgol_filter(raman, window_length=SG_WINDOW,
                       polyorder=SG_POLY, deriv=SG_DERIV, axis=1)
    scaled = scaler.transform(sg).astype(np.float32)
    return torch.from_numpy(scaled).unsqueeze(1)   # (N, 1, L)


def encode_batch(model: BetaVAE_Raman, x: torch.Tensor,
                 device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu, log_var) as numpy arrays, shape (T, latent_dim)."""
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        mu, log_var = model.encode(x)
    return mu.cpu().numpy(), log_var.cpu().numpy()


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_trajectory(
    time_h:       np.ndarray,
    z_trajectory: np.ndarray,
    pen:          np.ndarray,
    active_dim:   int,
    beta:         float,
    batch_id:     int,
    corr:         float,
    save_path:    str,
) -> None:
    scaler_z   = MinMaxScaler()
    scaler_pen = MinMaxScaler()
    z_norm   = scaler_z.fit_transform(z_trajectory.reshape(-1, 1)).flatten()
    pen_norm = scaler_pen.fit_transform(pen.reshape(-1, 1)).flatten()

    color_z   = "#1f77b4"   # blue
    color_pen = "#2ca02c"   # green

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Fermentation time (h)", fontweight="bold")
    ax1.set_ylabel(
        f"Latent activation $z_{{{active_dim}}}$ (normalised)",
        color=color_z,
    )
    line1 = ax1.plot(
        time_h, z_norm,
        color=color_z, linewidth=2.5,
        label=f"Latent $z_{{{active_dim}}}$  [surviving dim]",
    )
    ax1.tick_params(axis="y", labelcolor=color_z)
    ax1.set_ylim(-0.05, 1.2)

    # Overlay penicillin on same axis (both normalised to [0,1])
    line2 = ax1.plot(
        time_h, pen_norm,
        color=color_pen, linestyle="--", linewidth=2.5,
        label="Penicillin concentration (g/L)",
    )

    # Pearson annotation box
    ax1.text(
        0.03, 0.93,
        f"Pearson r = {corr:+.4f}",
        transform=ax1.transAxes,
        fontsize=11, fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="#aaaaaa", alpha=0.9),
    )

    lines  = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", fontsize=10)

    plt.title(
        f"Causal Disentanglement — β={beta}  |  batch {batch_id}\n"
        f"$z_{{{active_dim}}}$ as Biological Macro-Time",
        pad=15, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_all_dims(
    time_h:     np.ndarray,
    mu:         np.ndarray,
    active_dim: int,
    beta:       float,
    batch_id:   int,
    save_path:  str,
) -> None:
    """
    Small-multiple grid of all 64 latent dimensions coloured by variance.
    Active dim is highlighted in blue, collapsed dims in light grey.
    """
    D          = mu.shape[1]
    variances  = mu.var(axis=0)
    vmax       = variances.max()
    n_cols     = 8
    n_rows     = (D + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 2.0, n_rows * 1.5),
                              sharex=True)
    axes = axes.flatten()

    for d in range(D):
        ax  = axes[d]
        rel = variances[d] / (vmax + 1e-10)
        col = "#1f77b4" if d == active_dim else (
              "#e67e22" if rel > 0.01 else "#cccccc")
        lw  = 2.0 if d == active_dim else 0.8
        ax.plot(time_h, mu[:, d], color=col, linewidth=lw)
        ax.set_title(f"z{d:02d}", fontsize=6, pad=1)
        ax.tick_params(labelsize=5, length=2)
        ax.set_yticks([])

    for d in range(D, len(axes)):
        axes[d].set_visible(False)

    fig.suptitle(
        f"All 64 latent dimensions — β={beta}  batch {batch_id}\n"
        f"Blue = active dim z{active_dim}  |  Orange = semi-active  |  Grey = collapsed",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  All-dims plot saved → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(beta: float = 0.5, batch_id: int = 5) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[latent_trajectory]  β={beta}  batch={batch_id}  device={device}")

    tag       = _beta_tag(beta)
    ckpt_path = str(CKPT_ROOT / tag / "vae.pt")
    scaler_path = str(CKPT_ROOT / tag / "vae_scaler.joblib")

    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Available: {sorted(p.name for p in CKPT_ROOT.iterdir())}"
        )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n  Loading checkpoint: {ckpt_path}")
    state  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model  = BetaVAE_Raman(
        input_length = int(state["input_length"]),
        latent_dim   = int(state["latent_dim"]),
    )
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    print(f"  epoch={state['epoch']}  val_recon={state['val_loss']:.5f}  "
          f"β_stored={state['beta']}")

    scaler = joblib.load(scaler_path)
    print(f"  Scaler loaded from {scaler_path}")

    # ── Load batch ────────────────────────────────────────────────────────────
    print(f"\n  Loading batch {batch_id} from {CSV_PATH} ...")
    raman, pen, time_h, _ = load_batch(CSV_PATH, batch_id)

    # ── Encode ────────────────────────────────────────────────────────────────
    x_tensor      = preprocess(raman, scaler)
    mu, log_var   = encode_batch(model, x_tensor, device)
    print(f"  mu shape: {mu.shape}   log_var shape: {log_var.shape}")

    # ── Find active dimension (highest temporal variance in mu) ───────────────
    variances   = mu.var(axis=0)                  # (D,)
    active_dim  = int(variances.argmax())
    active_var  = float(variances[active_dim])
    print(f"\n  Variance per dim — top 5:")
    top5 = np.argsort(variances)[::-1][:5]
    for d in top5:
        print(f"    z{d:02d}  var={variances[d]:.6f}"
              + ("  ← ACTIVE" if d == active_dim else ""))

    z_trajectory = mu[:, active_dim]

    # ── Pearson correlation ───────────────────────────────────────────────────
    corr, pval = pearsonr(z_trajectory, pen)
    print(f"\n  Pearson r(z{active_dim}, Penicillin) = {corr:+.4f}  "
          f"p={pval:.2e}")

    # Also check the negative correlation (sign can flip arbitrarily)
    if abs(corr) < 0.5:
        print("  WARNING: correlation is weak — "
              "try a different batch or the sign may be flipped by encoder.")

    # ── Save plots ────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = f"beta{beta:g}_batch{batch_id}"

    plot_trajectory(
        time_h, z_trajectory, pen,
        active_dim=active_dim, beta=beta, batch_id=batch_id,
        corr=corr,
        save_path=str(OUTPUT_DIR / f"latent_trajectory_{slug}.png"),
    )
    plot_all_dims(
        time_h, mu,
        active_dim=active_dim, beta=beta, batch_id=batch_id,
        save_path=str(OUTPUT_DIR / f"all_dims_{slug}.png"),
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    n_active = int((variances > 0.01 * active_var).sum())
    print(f"\n  Summary")
    print(f"  ─────────────────────────────────────────")
    print(f"  β                   : {beta}")
    print(f"  Batch               : {batch_id}  ({len(time_h)} timesteps)")
    print(f"  Active dim          : z{active_dim}  (var={active_var:.5f})")
    print(f"  Dims with >1% max   : {n_active} / {mu.shape[1]}")
    print(f"  Pearson r           : {corr:+.4f}  (p={pval:.2e})")
    print(f"  Output dir          : {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Latent trajectory vs penicillin for a single batch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--beta",     type=float, default=0.5,
                        help="β value (must have a checkpoint in outputs/beta_sweep_e2e/checkpoints/)")
    parser.add_argument("--batch-id", type=int,   default=5,
                        help="0-indexed batch ID to visualise (0=batch1 … 99=batch100)")
    parser.add_argument("--csv",      default=CSV_PATH)
    args = parser.parse_args()

    CSV_PATH = args.csv
    main(beta=args.beta, batch_id=args.batch_id)
