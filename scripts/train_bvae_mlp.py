"""
scripts/train_bvae_mlp.py
=========================
β-VAE (β=0.05, frozen) + MLP head for direct Raman→Penicillin regression.

Architecture
------------
    Frozen β-VAE encoder (β=0.05, trained on R1)
        Raman spectrum (2001-dim, SG-derivative, scaled)
        → encode() → μ  (64-dim latent)

    Trainable MLP head
        64 → 128 → 64 → 1
        ReLU activations + Dropout(0.2) after each hidden layer

Pipeline
--------
    Spectrum  ──[SG deriv]──[StandardScaler]──[β-VAE enc, frozen]──  μ (64)
                                                                       │
                                                            [MLP head, trained]
                                                                       │
                                                              Penicillin (g/L)

Training
--------
    - Train regime : R1 only (batches 1–30, IDs 0–29)
    - Targets      : sparse offline Penicillin measurements only
                     (consistent with PLS/PCA-PCR and PI-LSTM evaluations)
    - Val split    : last 6 batches of R1 (IDs 24–29), train on IDs 0–23
    - Optimiser    : Adam, lr=1e-3, weight_decay=1e-4
    - Loss         : MSE
    - Epochs       : up to 500, early stopping (patience=30) on val RMSE
    - Batch size   : 256

Evaluation
----------
    Sparse offline measurement points in R1 / R2 / R3 / Fault

Usage
-----
    PYTHONPATH=. python scripts/train_bvae_mlp.py
    PYTHONPATH=. python scripts/train_bvae_mlp.py --beta 0.01
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error, r2_score

# ── Project root ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.autoencoder.model import BetaVAE_Raman
from src.data.dataset import RAMAN_START_COL_IDX, TARGET_COLS

# ── Config ─────────────────────────────────────────────────────────────────────
CSV_PATH      = "./100_Batches_IndPenSim_V3.csv"
SWEEP_DIR     = Path("./outputs/beta_sweep_e2e/checkpoints")
OUTPUT_DIR    = Path("./outputs/bvae_mlp")
CKPT_DIR      = Path("./checkpoints")

N_RAMAN       = 2001
LATENT_DIM    = 64
SG_WINDOW, SG_POLY, SG_DERIV = 15, 2, 1
MIN_BATCH_LEN = 50

REGIMES = {
    "R1":    list(range(0,   30)),
    "R2":    list(range(30,  60)),
    "R3":    list(range(60,  90)),
    "Fault": list(range(90, 100)),
}

# MLP training hyperparameters
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 256
MAX_EPOCHS   = 500
PATIENCE     = 30
DROPOUT      = 0.2
VAL_FRAC     = 0.2   # last 20% of R1 batches → val


# ── MLP head ───────────────────────────────────────────────────────────────────

class MLPHead(nn.Module):
    """64 → 128 → 64 → 1  (ReLU + Dropout)."""

    def __init__(self, latent_dim: int = LATENT_DIM, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)          # (B,)


# ── Data loading ───────────────────────────────────────────────────────────────

def _sg(X: np.ndarray) -> np.ndarray:
    return savgol_filter(X, window_length=SG_WINDOW,
                         polyorder=SG_POLY, deriv=SG_DERIV, axis=1)


def load_batches(csv_path: str) -> Dict[str, List[Dict]]:
    """Load + SG-preprocess all batches, bucketed by regime."""
    print(f"\n[DATA]  Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"[DATA]  Shape: {df.shape[0]:,} × {df.shape[1]}")

    t_vals    = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid
    df["_batch_id"] = batch_ids
    print(f"[DATA]  Detected {bid + 1} batches")

    all_cols    = df.columns.tolist()
    raman_names = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + N_RAMAN]
    pen_col     = TARGET_COLS[1]   # Penicillin_offline (sparse)

    regime_id_sets = {r: set(ids) for r, ids in REGIMES.items()}
    buckets: Dict[str, List[Dict]] = {r: [] for r in REGIMES}

    for b_id in range(bid + 1):
        b = df[df["_batch_id"] == b_id].copy().reset_index(drop=True)
        if len(b) < MIN_BATCH_LEN:
            continue

        y_sparse = b[pen_col].values.copy()

        X_raw = b[raman_names].values.astype(np.float64)
        X_raw = np.clip(np.nan_to_num(X_raw, 0.0), 0.0, None)
        mask  = np.abs(X_raw).sum(axis=1) > 0
        if mask.sum() < MIN_BATCH_LEN:
            continue

        entry = {
            "X_sg":     _sg(X_raw[mask]),         # SG-derivative spectra, raw scale
            "y_sparse": y_sparse[mask],
            "batch_id": b_id,
        }
        for regime, ids in regime_id_sets.items():
            if b_id in ids:
                buckets[regime].append(entry)
                break

    for r, bl in buckets.items():
        print(f"[DATA]  {r}: {len(bl)} batches")
    return buckets


# ── Encoder loading ────────────────────────────────────────────────────────────

def load_encoder(beta: float, device: torch.device) -> Tuple[BetaVAE_Raman, object]:
    """
    Load the frozen β-VAE encoder for the given beta value from the sweep
    checkpoint directory.  Returns (model_eval, scaler).
    """
    beta_str = str(beta)
    ckpt_dir = SWEEP_DIR / f"b{beta_str}"
    vae_path    = ckpt_dir / "vae.pt"
    scaler_path = ckpt_dir / "vae_scaler.joblib"

    if not vae_path.exists():
        raise FileNotFoundError(
            f"VAE checkpoint not found: {vae_path}\n"
            f"Run scripts/beta_sweep_e2e.py first with beta={beta}."
        )

    state  = torch.load(vae_path, map_location="cpu", weights_only=False)
    model  = BetaVAE_Raman(input_length=N_RAMAN, latent_dim=LATENT_DIM)
    model.load_state_dict(state["model_state_dict"])
    model  = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    scaler = joblib.load(scaler_path)
    print(f"[ENC]   Loaded β-VAE (β={beta}) from {vae_path}")
    print(f"[ENC]   Encoder is FROZEN — {sum(p.numel() for p in model.parameters()):,} params")
    return model, scaler


# ── Latent encoding ────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_batch(
    X_sg: np.ndarray,
    encoder: BetaVAE_Raman,
    scaler,
    device: torch.device,
) -> np.ndarray:
    """Scale + encode spectra → μ.  Input: (N, N_RAMAN)  Output: (N, LATENT_DIM)."""
    X_scaled = scaler.transform(X_sg).astype(np.float32)
    X_t      = torch.from_numpy(X_scaled).to(device)
    mu, _    = encoder.encode(X_t)
    return mu.cpu().numpy()


def build_dataset(
    batches:  List[Dict],
    encoder:  BetaVAE_Raman,
    scaler,
    device:   torch.device,
    sparse_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode all batches and collect (Z, y) pairs.
    sparse_only=True → use only offline measurement time-points.
    """
    Zs, ys = [], []
    for b in batches:
        if sparse_only:
            mask = ~np.isnan(b["y_sparse"])
            if mask.sum() == 0:
                continue
            X = b["X_sg"][mask]
            y = b["y_sparse"][mask]
        else:
            X = b["X_sg"]
            y = b["y_sparse"]

        Z = encode_batch(X, encoder, scaler, device)
        Zs.append(Z)
        ys.append(y)

    return np.vstack(Zs), np.concatenate(ys)


# ── Training ───────────────────────────────────────────────────────────────────

def train_mlp(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_val:   np.ndarray,
    y_val:   np.ndarray,
    device:  torch.device,
) -> Tuple[MLPHead, List[float], List[float]]:
    """Train the MLP head with early stopping. Returns (best_model, train_losses, val_losses)."""
    head = MLPHead().to(device)
    opt  = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    Z_tr = torch.from_numpy(Z_train.astype(np.float32)).to(device)
    y_tr = torch.from_numpy(y_train.astype(np.float32)).to(device)
    Z_vl = torch.from_numpy(Z_val.astype(np.float32)).to(device)
    y_vl = torch.from_numpy(y_val.astype(np.float32)).to(device)

    n_train = len(Z_tr)
    best_val_rmse = float("inf")
    best_state    = None
    patience_cnt  = 0

    train_losses, val_losses = [], []

    print(f"\n[MLP]   Training  n_train={n_train}  n_val={len(Z_vl)}  "
          f"max_epochs={MAX_EPOCHS}  patience={PATIENCE}")

    for epoch in range(1, MAX_EPOCHS + 1):
        head.train()
        # Shuffle
        idx = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches  = 0
        for start in range(0, n_train, BATCH_SIZE):
            b_idx  = idx[start : start + BATCH_SIZE]
            z_b    = Z_tr[b_idx]
            y_b    = y_tr[b_idx]
            pred   = head(z_b)
            loss   = loss_fn(pred, y_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches  += 1

        train_mse = epoch_loss / n_batches
        train_losses.append(float(np.sqrt(train_mse)))

        # Validation
        head.eval()
        with torch.no_grad():
            val_pred  = head(Z_vl)
            val_mse   = loss_fn(val_pred, y_vl).item()
            val_rmse  = float(np.sqrt(val_mse))
        val_losses.append(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state    = {k: v.clone() for k, v in head.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:>4d}  train_RMSE={train_losses[-1]:.4f}  "
                  f"val_RMSE={val_rmse:.4f}  best={best_val_rmse:.4f}  patience={patience_cnt}")

        if patience_cnt >= PATIENCE:
            print(f"  Early stop at epoch {epoch}  (best val RMSE={best_val_rmse:.4f})")
            break

    head.load_state_dict(best_state)
    head.eval()
    print(f"[MLP]   Best val RMSE = {best_val_rmse:.4f} g/L")
    return head, train_losses, val_losses


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(
    head:     MLPHead,
    encoder:  BetaVAE_Raman,
    scaler,
    batches:  List[Dict],
    label:    str,
    device:   torch.device,
) -> Dict:
    """Evaluate on sparse points. Returns metrics dict."""
    all_true, all_pred = [], []
    per_batch = []

    for b in batches:
        mask = ~np.isnan(b["y_sparse"])
        if mask.sum() == 0:
            continue
        y_true = b["y_sparse"][mask]
        Z = encode_batch(b["X_sg"][mask], encoder, scaler, device)
        with torch.no_grad():
            y_pred = head(torch.from_numpy(Z).float().to(device)).cpu().numpy()
        all_true.append(y_true)
        all_pred.append(y_pred)
        rmse_b = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        per_batch.append({"batch_id": b["batch_id"], "RMSE": rmse_b, "n": int(mask.sum())})

    if not all_true:
        return {}

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))
    mae  = float(np.mean(np.abs(y_true_all - y_pred_all)))
    r2   = float(r2_score(y_true_all, y_pred_all))
    print(f"  [{label:<20}]  RMSE={rmse:.4f} g/L  MAE={mae:.4f}  R²={r2:.4f}  n={len(y_true_all)}")
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "n": len(y_true_all), "per_batch": per_batch}


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_training_curve(
    train_losses: List[float],
    val_losses:   List[float],
    save_path:    str,
    beta:         float,
) -> None:
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label="Train RMSE", color="#2980b9", linewidth=1.5)
    ax.plot(epochs, val_losses,   label="Val RMSE",   color="#e74c3c",  linewidth=1.5)
    best_ep = int(np.argmin(val_losses)) + 1
    best_v  = min(val_losses)
    ax.axvline(best_ep, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(best_ep + 1, best_v + 0.01, f"best ep={best_ep}\nRMSE={best_v:.4f}",
            fontsize=8, color="#e74c3c")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("RMSE (g/L)", fontsize=10)
    ax.set_title(f"β-VAE (β={beta}) + MLP Head — Training Curve", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_per_batch_rmse(
    per_batch_results: Dict[str, List[Dict]],
    save_path: str,
    beta: float,
) -> None:
    """One panel per eval split."""
    splits = list(per_batch_results.keys())
    n = len(splits)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]
    colors = ["#2980b9", "#27ae60", "#8e44ad", "#e74c3c"]

    for ax, split, col in zip(axes, splits, colors):
        pb = per_batch_results[split]
        ids   = [r["batch_id"] + 1 for r in pb]
        rmses = [r["RMSE"] for r in pb]
        ax.bar(range(len(ids)), rmses, color=col, alpha=0.8)
        ax.axhline(np.mean(rmses), color="k", linestyle="--", linewidth=1.2,
                   label=f"mean={np.mean(rmses):.3f}")
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels([str(i) for i in ids],
                           rotation=45 if len(ids) > 10 else 0, fontsize=7)
        ax.set_title(split, fontsize=10, fontweight="bold")
        ax.set_ylabel("RMSE (g/L)" if ax is axes[0] else "", fontsize=9)
        ax.set_xlabel("Batch", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_ylim(0)

    fig.suptitle(f"β-VAE (β={beta}) + MLP Head — Per-batch Penicillin RMSE",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_summary_comparison(
    results:   Dict,          # keyed by split name → {"RMSE": ...}
    save_path: str,
    beta:      float,
) -> None:
    """Bar chart of RMSE across eval splits with PLS R1 reference line."""
    splits = list(results.keys())
    rmses  = [results[s]["RMSE"] for s in splits]

    # Reference lines from PLS R1-only run (known values from prior session)
    pls_r1_ref = {"R1 (in-sample)": 0.2463, "R2 (OOD)": 0.2683,
                  "R3 (OOD)": 0.2604, "Fault (OOD)": 0.2555}

    fig, ax = plt.subplots(figsize=(max(7, len(splits) * 2), 5))
    x = np.arange(len(splits))
    bars = ax.bar(x, rmses, color="#8e44ad", alpha=0.85, label=f"β-VAE+MLP (β={beta})")
    for bar, v in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Overlay PLS reference dots
    pls_vals = []
    for s in splits:
        key = s if s in pls_r1_ref else None
        if key:
            pls_vals.append(pls_r1_ref[key])
        else:
            pls_vals.append(None)

    for i, pv in enumerate(pls_vals):
        if pv is not None:
            ax.plot(i, pv, "D", color="#2980b9", markersize=8, zorder=5,
                    label="PLS K=64 R1" if i == 0 else "_nolegend_")

    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=10)
    ax.set_ylabel("Penicillin RMSE (g/L)", fontsize=10)
    ax.set_title(f"β-VAE (β={beta}) + MLP Head vs PLS baseline  —  sparse points",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(csv_path: str = CSV_PATH, beta: float = 0.05) -> None:
    t0 = time.time()

    out_dir = OUTPUT_DIR / f"beta{beta}"
    out_dir.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INIT]  device={device}  β={beta}")

    # ── 1. Load encoder ────────────────────────────────────────────────────────
    encoder, scaler = load_encoder(beta, device)

    # ── 2. Load data ───────────────────────────────────────────────────────────
    buckets = load_batches(csv_path)

    # ── 3. Build train / val datasets from R1 ─────────────────────────────────
    r1 = buckets["R1"]
    n_val   = max(1, int(len(r1) * VAL_FRAC))
    r1_train = r1[: len(r1) - n_val]
    r1_val   = r1[len(r1) - n_val :]
    print(f"\n[DATA]  R1 train batches: {len(r1_train)}  val batches: {len(r1_val)}")

    Z_train, y_train = build_dataset(r1_train, encoder, scaler, device, sparse_only=True)
    Z_val,   y_val   = build_dataset(r1_val,   encoder, scaler, device, sparse_only=True)
    print(f"[DATA]  Z_train={Z_train.shape}  Z_val={Z_val.shape}")

    # ── 4. Train MLP head ──────────────────────────────────────────────────────
    head, train_losses, val_losses = train_mlp(Z_train, y_train, Z_val, y_val, device)

    # ── 5. Evaluate ────────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  EVALUATION — sparse offline measurement points only")
    print(f"{'─'*55}")

    eval_splits = [
        ("R1 (in-sample)", r1),
        ("R2 (OOD)",       buckets["R2"]),
        ("R3 (OOD)",       buckets["R3"]),
        ("Fault (OOD)",    buckets["Fault"]),
    ]
    results = {}
    per_batch_all = {}
    for label, split_batches in eval_splits:
        m = evaluate(head, encoder, scaler, split_batches, label, device)
        results[label]     = m
        per_batch_all[label] = m.get("per_batch", [])

    # ── 6. Summary table ───────────────────────────────────────────────────────
    sep = "─" * 70
    print(f"\n{'='*70}")
    print(f"  β-VAE (β={beta}) + MLP Head  —  Penicillin RMSE (g/L)")
    print(f"{'='*70}")
    header = f"  {'Split':<22}" + f"  {'RMSE':>8}  {'MAE':>8}  {'R²':>8}  {'n':>6}"
    print(header)
    print(sep)
    r1_rmse = results["R1 (in-sample)"]["RMSE"]
    for label in results:
        m = results[label]
        ood = (m["RMSE"] - r1_rmse) / (r1_rmse + 1e-8) * 100
        ood_str = f"  ({ood:+.1f}%)" if label != "R1 (in-sample)" else ""
        print(f"  {label:<22}  {m['RMSE']:>8.4f}  {m['MAE']:>8.4f}  {m['R2']:>8.4f}  {m['n']:>6}{ood_str}")
    print(f"{'='*70}")

    # ── 7. Plots ───────────────────────────────────────────────────────────────
    plot_training_curve(train_losses, val_losses,
                        save_path=str(out_dir / "training_curve.png"), beta=beta)
    plot_per_batch_rmse(per_batch_all,
                        save_path=str(out_dir / "per_batch_rmse.png"), beta=beta)
    plot_summary_comparison(results,
                            save_path=str(out_dir / "summary_comparison.png"), beta=beta)

    # ── 8. Save checkpoint ─────────────────────────────────────────────────────
    mlp_ckpt = CKPT_DIR / f"bvae_mlp_beta{beta}.pt"
    torch.save({"model_state_dict": head.state_dict(),
                "beta": beta, "latent_dim": LATENT_DIM,
                "results": results},
               mlp_ckpt)
    print(f"\n  MLP checkpoint → {mlp_ckpt}")
    print(f"  Plots in {out_dir}/")
    print(f"  Total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a frozen β-VAE encoder + MLP head for Penicillin prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv",  default=CSV_PATH, help="Path to IndPenSim CSV")
    parser.add_argument("--beta", type=float, default=0.05,
                        help="β value — must have a trained checkpoint in "
                             "outputs/beta_sweep_e2e/checkpoints/b{beta}/")
    args = parser.parse_args()
    main(csv_path=args.csv, beta=args.beta)
