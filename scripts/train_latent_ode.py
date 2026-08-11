"""
scripts/train_latent_ode.py
===========================
Train a Latent ODE soft-sensor on top of a frozen β-VAE Raman encoder.

Data flow
---------
                  (offline, pre-trained, FROZEN)
   Raman spectra  ──[SG deriv]──[StandardScaler]──[β-VAE encoder]──► μ(t) ∈ ℝ^64
                                                                         │
                                                 (sliding window, seq_len steps)
                                                                         │
                                              z₀ = μ(t_window_start)    ◄──── z₀ only!
                                                                         │
                                              LatentODE.forward(z₀, t)  │
                                              (integrates dz/dt = f(z,t))│
                                                                         │
                                              ŷ(t) = W·z_ode(t) + b  ───► Penicillin (g/L)

Key differences vs PI-LSTM
--------------------------
  PI-LSTM   : processes the *full* latent sequence at every step; gates accumulate
              error over long sequences; no guarantee of smooth interpolation.
  LatentODE : only needs z₀ (the initial latent state); integrates a smooth
              continuous-time vector field dz/dt = f_θ(z, t); predictions at
              any time t are exact evaluations of the ODE solution.

ODE input dimensionality
------------------------
  --ode-dim 64  (default)  : full 64-dim latent — most expressive
  --ode-dim 2               : active dims only ([52, 27] for β=0.05)
                              Penicillin + Glucose axes, highly interpretable
  --ode-dim 1               : Penicillin axis only (z52, r=0.995 with Pen)

Usage
-----
    # Full latent (64-dim ODE)
    PYTHONPATH=. python scripts/train_latent_ode.py

    # Active dims only (2-dim ODE, β=0.05)
    PYTHONPATH=. python scripts/train_latent_ode.py --ode-dim 2 --active-dims 52 27

    # Different beta
    PYTHONPATH=. python scripts/train_latent_ode.py --beta 0.5 --ode-dim 1 --active-dims 49

    # dopri5 solver (adaptive, adjoint backprop)
    PYTHONPATH=. python scripts/train_latent_ode.py --solver dopri5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from sklearn.preprocessing import StandardScaler as SKStandardScaler

# ── Project root ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.autoencoder.model import BetaVAE_Raman
from src.data.dataset import RAMAN_START_COL_IDX, TARGET_COLS
from src.models.latent_ode import LatentODE, make_training_step

# ── Config ─────────────────────────────────────────────────────────────────────
CSV_PATH   = "./100_Batches_IndPenSim_V3.csv"
SWEEP_DIR  = Path("./outputs/beta_sweep_e2e/checkpoints")
OUTPUT_DIR = Path("./outputs/latent_ode")
CKPT_DIR   = Path("./checkpoints")

N_RAMAN        = 2001
VAE_LATENT_DIM = 64
SG_WINDOW, SG_POLY, SG_DERIV = 15, 2, 1
MIN_BATCH_LEN  = 50
DT_HOURS       = 0.2   # physical time step (h) of IndPenSim

# Training
SEQ_LEN       = 24    # sliding window length (steps = 24 × 0.2 h = 4.8 h)
BATCH_SIZE    = 128
MAX_EPOCHS    = 300
PATIENCE      = 40
LR            = 3e-3
WEIGHT_DECAY  = 1e-4
VAL_FRAC      = 0.2   # last 20% of R1 batches reserved for validation

# Regime batch ID ranges (0-indexed)
REGIMES = {
    "R1":    list(range(0,   30)),
    "R2":    list(range(30,  60)),
    "R3":    list(range(60,  90)),
    "Fault": list(range(90, 100)),
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _sg(X: np.ndarray) -> np.ndarray:
    """Savitzky-Golay 1st-derivative filter (matches VAE preprocessing)."""
    return savgol_filter(X, window_length=SG_WINDOW,
                         polyorder=SG_POLY, deriv=SG_DERIV, axis=1)


def load_batches(csv_path: str) -> Dict[str, List[Dict]]:
    """
    Load IndPenSim CSV and return batch dicts bucketed by regime.

    Each dict:
        X_sg      : (T, 2001) float64 — SG-derivative spectra
        y_sparse  : (T,)      float64 — offline Penicillin (NaN between measures)
        y_dense   : (T,)      float64 — linearly-interpolated Penicillin
        t_hours   : (T,)      float64 — physical time in hours
        batch_id  : int
    """
    print(f"\n[DATA]  Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"[DATA]  Shape: {df.shape[0]:,} × {df.shape[1]}")

    t_vals    = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid       = 0
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
        # Dense target: interpolate offline measurements for training signal
        b[pen_col] = b[pen_col].interpolate(method="linear").ffill().bfill()
        y_dense    = b[pen_col].values.copy()

        X_raw = b[raman_names].values.astype(np.float64)
        X_raw = np.clip(np.nan_to_num(X_raw, 0.0), 0.0, None)
        mask  = np.abs(X_raw).sum(axis=1) > 0
        if mask.sum() < MIN_BATCH_LEN:
            continue

        entry = {
            "X_sg":     _sg(X_raw[mask]),
            "y_sparse": y_sparse[mask],
            "y_dense":  y_dense[mask],
            "t_hours":  b["Time (h)"].values[mask],
            "batch_id": b_id,
        }
        for regime, ids in regime_id_sets.items():
            if b_id in ids:
                buckets[regime].append(entry)
                break

    for r, bl in buckets.items():
        print(f"[DATA]  {r}: {len(bl)} batches")
    return buckets


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Frozen encoder loading + pre-encoding
# ─────────────────────────────────────────────────────────────────────────────

def load_encoder(beta: float, device: torch.device) -> Tuple[BetaVAE_Raman, object]:
    """Load the frozen β-VAE encoder and its spectrum scaler."""
    beta_str    = str(beta)
    vae_path    = SWEEP_DIR / f"b{beta_str}" / "vae.pt"
    scaler_path = SWEEP_DIR / f"b{beta_str}" / "vae_scaler.joblib"
    if not vae_path.exists():
        raise FileNotFoundError(
            f"VAE checkpoint not found: {vae_path}\n"
            f"Run scripts/beta_sweep_e2e.py first with --betas {beta}."
        )
    state  = torch.load(vae_path, map_location="cpu", weights_only=False)
    model  = BetaVAE_Raman(input_length=N_RAMAN, latent_dim=VAE_LATENT_DIM)
    model.load_state_dict(state["model_state_dict"])
    model  = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    scaler = joblib.load(scaler_path)
    print(f"[ENC]   Loaded β-VAE (β={beta}) — {sum(p.numel() for p in model.parameters()):,} params FROZEN")
    return model, scaler


@torch.no_grad()
def encode_batch_spectra(
    X_sg:    np.ndarray,     # (T, N_RAMAN) SG-derivative spectra
    encoder: BetaVAE_Raman,
    scaler,
    device:  torch.device,
    chunk:   int = 1024,
) -> np.ndarray:
    """
    Scale + encode spectra → posterior mean μ.

    Returns
    -------
    mu : (T, VAE_LATENT_DIM) float32
    """
    X_scaled = scaler.transform(X_sg).astype(np.float32)
    parts    = []
    for start in range(0, len(X_scaled), chunk):
        x_t    = torch.from_numpy(X_scaled[start : start + chunk]).to(device)
        mu, _  = encoder.encode(x_t)
        parts.append(mu.cpu().numpy())
    return np.concatenate(parts, axis=0)   # (T, 64)


def pre_encode_all(
    buckets:  Dict[str, List[Dict]],
    encoder:  BetaVAE_Raman,
    scaler,
    device:   torch.device,
) -> Dict[str, List[Dict]]:
    """
    Replace X_sg with mu (VAE latent) in every batch entry.
    This is done once before training to avoid repeated encoder calls.

    Returns a new dict with the same structure but X_sg replaced by:
        mu : (T, VAE_LATENT_DIM) float32
    """
    print("\n[ENC]   Pre-encoding all batches (one-time pass) ...")
    encoded: Dict[str, List[Dict]] = {}
    for regime, batches in buckets.items():
        enc_batches = []
        for b in batches:
            mu = encode_batch_spectra(b["X_sg"], encoder, scaler, device)
            enc_batches.append({
                "mu":       mu,              # (T, 64)  — replaces X_sg
                "y_sparse": b["y_sparse"],
                "y_dense":  b["y_dense"],
                "t_hours":  b["t_hours"],
                "batch_id": b["batch_id"],
            })
        encoded[regime] = enc_batches
        print(f"  {regime}: {len(enc_batches)} batches encoded")
    return encoded


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Penicillin scaler (fitted on R1 train)
# ─────────────────────────────────────────────────────────────────────────────

def fit_pen_scaler(r1_batches: List[Dict]) -> SKStandardScaler:
    """
    Fit a StandardScaler on the Penicillin y_dense values of the R1 batches.
    We normalise the target so the ODE readout layer operates in a stable
    range regardless of the absolute Penicillin magnitude.
    """
    y_all  = np.concatenate([b["y_dense"] for b in r1_batches])
    scaler = SKStandardScaler()
    scaler.fit(y_all.reshape(-1, 1))
    print(f"[SCALE] Pen scaler fit:  mean={scaler.mean_[0]:.3f}  std={scaler.scale_[0]:.3f}")
    return scaler


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Sliding-window dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_windows(
    batches:    List[Dict],
    pen_scaler: SKStandardScaler,
    seq_len:    int = SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding windows from pre-encoded latent trajectories.

    Returns
    -------
    Z0       : (N, VAE_LATENT_DIM)  — initial latent state (μ at t=0 of window)
    Y        : (N, seq_len)         — normalised dense Penicillin targets
    Y_sparse : (N, seq_len) bool    — True where offline measurement exists
    """
    Z0_list, Y_list, Ysp_list = [], [], []

    for b in batches:
        mu      = b["mu"]                                        # (T, 64)
        y_dense = pen_scaler.transform(                          # (T,)
            b["y_dense"].reshape(-1, 1)).flatten().astype(np.float32)
        y_sp    = ~np.isnan(b["y_sparse"])                       # (T,) bool

        T = len(mu)
        for i in range(T - seq_len + 1):
            # z₀ = encoder output at the FIRST time step of this window
            # The ODE will reconstruct the rest from this single initial condition
            Z0_list.append(mu[i].astype(np.float32))            # (64,)
            Y_list.append(y_dense[i : i + seq_len])             # (seq_len,)
            Ysp_list.append(y_sp[i : i + seq_len])              # (seq_len,) bool

    Z0       = np.array(Z0_list,  dtype=np.float32)    # (N, 64)
    Y        = np.array(Y_list,   dtype=np.float32)    # (N, seq_len)
    Y_sparse = np.array(Ysp_list, dtype=bool)          # (N, seq_len)
    print(f"  Windows built: {len(Z0):,}  (seq_len={seq_len})")
    return Z0, Y, Y_sparse


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model:        LatentODE,
    Z0_tr:        np.ndarray,    # (N_tr, 64)
    Y_tr:         np.ndarray,    # (N_tr, seq_len)
    Ysp_tr:       np.ndarray,    # (N_tr, seq_len) bool
    Z0_vl:        np.ndarray,    # (N_vl, 64)
    Y_vl:         np.ndarray,
    Ysp_vl:       np.ndarray,
    device:       torch.device,
    max_epochs:   int   = MAX_EPOCHS,
    patience:     int   = PATIENCE,
    lr:           float = LR,
    weight_decay: float = WEIGHT_DECAY,
    batch_size:   int   = BATCH_SIZE,
    seq_len:      int   = SEQ_LEN,
) -> Tuple[List[float], List[float]]:
    """
    Train the LatentODE with Adam + early stopping.

    Time tensor construction
    ------------------------
    t is built as torch.linspace(0, 1, seq_len) — normalised to [0, 1].
    The ODE dynamics are learned in this normalised time, making the model
    invariant to the absolute window duration.

    Loss strategy
    -------------
    Primary loss : MSE on the dense interpolated Penicillin (every time step).
    Auxiliary    : no physics penalty here; the ODE structure already
                   provides implicit regularisation via its inductive bias.

    Returns train / val RMSE curves.
    """
    t = torch.linspace(0, 1, seq_len, device=device)   # (T,) normalised time

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-5
    )
    step_fn = make_training_step(model, optimizer)

    # Convert to tensors
    Z0_tr_t  = torch.from_numpy(Z0_tr).to(device)
    Y_tr_t   = torch.from_numpy(Y_tr).to(device)
    Ysp_tr_t = torch.from_numpy(Ysp_tr).to(device)
    Z0_vl_t  = torch.from_numpy(Z0_vl).to(device)
    Y_vl_t   = torch.from_numpy(Y_vl).to(device)
    Ysp_vl_t = torch.from_numpy(Ysp_vl).to(device)

    n_tr = len(Z0_tr_t)
    train_rmses, val_rmses = [], []
    best_val_rmse  = float("inf")
    best_state     = None
    patience_count = 0

    print(f"\n[TRAIN] n_train={n_tr:,}  n_val={len(Z0_vl_t):,}  "
          f"max_epochs={max_epochs}  patience={patience}")
    print(f"[TRAIN] solver={model.solver}  ode_dim={model.ode_dim}  "
          f"active_dims={model.active_dims}  params={model.n_params():,}")

    for epoch in range(1, max_epochs + 1):
        model.train()
        # Shuffle training indices each epoch
        idx       = torch.randperm(n_tr, device=device)
        ep_losses = []

        for start in range(0, n_tr, batch_size):
            b_idx  = idx[start : start + batch_size]
            z0_b   = Z0_tr_t[b_idx]      # (B, latent_dim)
            y_b    = Y_tr_t[b_idx]       # (B, T)
            # Dense loss: supervise at every time step
            # This gives the ODE more signal to learn the penicillin trajectory shape
            loss = step_fn(z0_b, t, y_b, sparse_mask=None)
            ep_losses.append(loss)

        # Validation: evaluate at SPARSE measurement points for fair comparison
        model.eval()
        with torch.no_grad():
            val_pred   = model(Z0_vl_t, t)            # (N_vl, T)
            sp_mask_vl = Ysp_vl_t                     # (N_vl, T) bool
            if sp_mask_vl.any():
                val_mse  = nn.functional.mse_loss(
                    val_pred[sp_mask_vl], Y_vl_t[sp_mask_vl]
                ).item()
            else:
                val_mse  = nn.functional.mse_loss(val_pred, Y_vl_t).item()
            val_rmse = float(np.sqrt(val_mse))

        tr_rmse = float(np.sqrt(np.mean(ep_losses)))
        train_rmses.append(tr_rmse)
        val_rmses.append(val_rmse)
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 25 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch:>4d}  train={tr_rmse:.4f}  val={val_rmse:.4f}  "
                  f"best={best_val_rmse:.4f}  patience={patience_count}  lr={lr_now:.2e}")

        if patience_count >= patience:
            print(f"  Early stop at epoch {epoch} — best val RMSE = {best_val_rmse:.4f}")
            break

    model.load_state_dict(best_state)
    model.eval()
    print(f"\n[TRAIN] Done. Best val RMSE = {best_val_rmse:.4f} (normalised)")
    return train_rmses, val_rmses


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Evaluation at sparse offline measurement points
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_regime(
    model:      LatentODE,
    batches:    List[Dict],
    pen_scaler: SKStandardScaler,
    device:     torch.device,
    label:      str,
    seq_len:    int = SEQ_LEN,
) -> Dict:
    """
    Evaluate the LatentODE on a list of batches.

    Strategy: for each batch, slide a window of length seq_len across the
    latent trajectory.  The prediction for each time step is the output of
    the window whose START coincides with that step (i.e. z₀ = μ(t_step)
    and the ODE is asked to predict pen(t_step … t_step + seq_len - 1)).

    We collect predictions only at sparse offline measurement points.

    Returns
    -------
    dict with RMSE, MAE, R2, n, per_batch
    """
    model.eval()
    t = torch.linspace(0, 1, seq_len, device=device)

    all_true, all_pred = [], []
    per_batch = []

    with torch.no_grad():
        for b in batches:
            mu      = b["mu"]                         # (T, 64)
            y_sp    = b["y_sparse"]                   # (T,)
            T       = len(mu)

            if T < seq_len:
                continue

            # Build all windows starting at each time step
            # z₀ for step i = mu[i] ; predict pen at [i, i+1, ..., i+seq_len-1]
            # We only need the FIRST prediction from each window (pos 0) to
            # get one penicillin estimate per time step.
            # This avoids averaging over multiple overlapping window predictions.
            N_win   = T - seq_len + 1
            z0_all  = torch.from_numpy(
                mu[:N_win].astype(np.float32)
            ).to(device)                              # (N_win, 64)

            # Run the ODE for all windows in one batched call
            # y_hat: (N_win, seq_len) — normalised penicillin predictions
            CHUNK   = 256   # process in chunks to avoid OOM on long batches
            y_hat_norm = np.full((T,), np.nan, dtype=np.float32)

            for start in range(0, N_win, CHUNK):
                z0_ch    = z0_all[start : start + CHUNK]   # (ch, 64)
                y_ch     = model(z0_ch, t).cpu().numpy()   # (ch, seq_len)
                # Take only the first time-step prediction from each window
                # (position 0 in the seq_len dimension = prediction at t_start)
                y_hat_norm[start : start + len(z0_ch)] = y_ch[:, 0]

            # Inverse-transform to g/L
            valid_mask    = ~np.isnan(y_hat_norm)
            pred_pen_full = np.full(T, np.nan, dtype=np.float32)
            if valid_mask.any():
                pred_pen_full[valid_mask] = pen_scaler.inverse_transform(
                    y_hat_norm[valid_mask].reshape(-1, 1)
                ).flatten()

            # Collect sparse-point pairs
            sp_mask = ~np.isnan(y_sp) & ~np.isnan(pred_pen_full)
            if sp_mask.sum() == 0:
                continue

            all_true.append(y_sp[sp_mask])
            all_pred.append(pred_pen_full[sp_mask])

            pb_rmse = float(np.sqrt(mean_squared_error(y_sp[sp_mask], pred_pen_full[sp_mask])))
            per_batch.append({"batch_id": b["batch_id"], "RMSE": pb_rmse, "n": int(sp_mask.sum())})

    if not all_true:
        return {"RMSE": float("nan"), "MAE": float("nan"), "R2": float("nan"), "n": 0}

    y_true_cat = np.concatenate(all_true)
    y_pred_cat = np.concatenate(all_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true_cat, y_pred_cat)))
    mae  = float(np.mean(np.abs(y_true_cat - y_pred_cat)))
    r2   = float(r2_score(y_true_cat, y_pred_cat))
    n    = len(y_true_cat)

    print(f"  [{label:<20}]  RMSE={rmse:.4f} g/L  MAE={mae:.4f}  R²={r2:.4f}  n={n}")
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "n": n, "per_batch": per_batch}


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curve(
    train_rmses: List[float],
    val_rmses:   List[float],
    out_dir:     Path,
    cfg_tag:     str,
) -> None:
    epochs = list(range(1, len(train_rmses) + 1))
    best_ep = int(np.argmin(val_rmses)) + 1
    best_v  = min(val_rmses)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, train_rmses, label="Train RMSE (dense)", color="#2980b9", lw=1.5)
    ax.plot(epochs, val_rmses,   label="Val RMSE (sparse)",  color="#e74c3c", lw=1.5)
    ax.axvline(best_ep, color="#e74c3c", linestyle="--", lw=1, alpha=0.7)
    ax.text(best_ep + 1, best_v + 0.01,
            f"best ep={best_ep}\nval={best_v:.4f}", fontsize=8, color="#e74c3c")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("RMSE (normalised)", fontsize=10)
    ax.set_title(f"Latent ODE — Training curve  [{cfg_tag}]",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = str(out_dir / "training_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {path}")


def plot_summary_bar(
    results:  Dict[str, Dict],
    out_dir:  Path,
    cfg_tag:  str,
    beta:     float,
) -> None:
    """RMSE per split with reference lines for PLS and β-VAE+MLP."""
    pls_ref  = {"R1 (in-sample)": 0.2463, "R2 (OOD)": 0.2683,
                "R3 (OOD)": 0.2604, "Fault (OOD)": 0.2555}
    mlp_ref  = {"R1 (in-sample)": 0.4454, "R2 (OOD)": 0.5238,
                "R3 (OOD)": 0.4926, "Fault (OOD)": 0.5486}

    splits = list(results.keys())
    rmses  = [results[s]["RMSE"] for s in splits]
    x      = np.arange(len(splits))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, rmses, color="#1abc9c", alpha=0.85,
                  label=f"Latent ODE (β={beta}, {cfg_tag})")
    for bar, v in zip(bars, rmses):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    for i, s in enumerate(splits):
        if s in pls_ref:
            ax.plot(i, pls_ref[s], "D", color="#2980b9", markersize=9, zorder=5,
                    label="PLS K=64 (R1)" if i == 0 else "_nolegend_")
        if s in mlp_ref:
            ax.plot(i, mlp_ref[s], "^", color="#8e44ad", markersize=9, zorder=5,
                    label="β-VAE+MLP (R1)" if i == 0 else "_nolegend_")

    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=10)
    ax.set_ylabel("Penicillin RMSE (g/L)", fontsize=10)
    ax.set_title(f"Latent ODE vs baselines  —  β={beta}  [{cfg_tag}]",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0)
    plt.tight_layout()
    path = str(out_dir / "summary_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {path}")


def plot_trajectory(
    model:      LatentODE,
    batch:      Dict,
    pen_scaler: SKStandardScaler,
    device:     torch.device,
    out_path:   str,
    seq_len:    int = SEQ_LEN,
) -> None:
    """Overlay ODE prediction vs offline measurements for one representative batch."""
    model.eval()
    t   = torch.linspace(0, 1, seq_len, device=device)
    mu  = batch["mu"]
    y_sp = batch["y_sparse"]
    T   = len(mu)

    N_win      = T - seq_len + 1
    pred_norm  = np.full(T, np.nan, dtype=np.float32)
    CHUNK      = 256

    with torch.no_grad():
        for start in range(0, N_win, CHUNK):
            end   = min(start + CHUNK, N_win)
            z0_ch = torch.from_numpy(
                mu[start:end].astype(np.float32)
            ).to(device)
            y_ch  = model(z0_ch, t).cpu().numpy()   # (ch, seq_len)
            pred_norm[start:end] = y_ch[:, 0]

    valid = ~np.isnan(pred_norm)
    pred_gL = np.full(T, np.nan, dtype=np.float32)
    if valid.any():
        pred_gL[valid] = pen_scaler.inverse_transform(
            pred_norm[valid].reshape(-1, 1)
        ).flatten()

    t_axis   = batch["t_hours"]
    sp_mask  = ~np.isnan(y_sp)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_axis, pred_gL, color="#1abc9c", lw=2, label="Latent ODE prediction")
    ax.scatter(t_axis[sp_mask], y_sp[sp_mask], color="#2c3e50", s=30,
               zorder=5, label="Offline measurement")
    ax.set_xlabel("Time (h)", fontsize=10)
    ax.set_ylabel("Penicillin (g/L)", fontsize=10)
    ax.set_title(f"Latent ODE — Batch {batch['batch_id']+1}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    csv_path:    str        = CSV_PATH,
    beta:        float      = 0.05,
    ode_dim:     int        = VAE_LATENT_DIM,
    active_dims: Optional[List[int]] = None,
    hidden_size: int        = 32,
    n_layers:    int        = 2,
    activation:  str        = "swish",
    solver:      str        = "rk4",
    seq_len:     int        = SEQ_LEN,
    max_epochs:  int        = MAX_EPOCHS,
) -> None:
    t0 = time.time()

    # ── Build config tag for output naming ────────────────────────────────────
    dims_tag  = f"z{'_'.join(str(d) for d in active_dims)}" if active_dims else f"dim{ode_dim}"
    cfg_tag   = f"{dims_tag}_{activation}_h{hidden_size}l{n_layers}_{solver}"
    out_dir   = OUTPUT_DIR / f"beta{beta}" / cfg_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INIT]  device={device}  β={beta}  ode_dim={ode_dim}  "
          f"active_dims={active_dims}  solver={solver}")

    # ── 1. Load frozen encoder ────────────────────────────────────────────────
    encoder, vae_scaler = load_encoder(beta, device)

    # ── 2. Load & pre-encode all batches ──────────────────────────────────────
    buckets = load_batches(csv_path)
    encoded = pre_encode_all(buckets, encoder, vae_scaler, device)

    # ── 3. R1 train / val split ───────────────────────────────────────────────
    r1 = encoded["R1"]
    n_val     = max(1, int(len(r1) * VAL_FRAC))
    r1_train  = r1[: len(r1) - n_val]
    r1_val    = r1[len(r1) - n_val :]
    print(f"\n[DATA]  R1 train batches: {len(r1_train)}  val batches: {len(r1_val)}")

    # ── 4. Fit Penicillin scaler on R1 train ──────────────────────────────────
    pen_scaler = fit_pen_scaler(r1_train)

    # ── 5. Build sliding-window datasets ──────────────────────────────────────
    print("\n[WIN]   Building sliding windows ...")
    Z0_tr, Y_tr, Ysp_tr = build_windows(r1_train, pen_scaler, seq_len)
    Z0_vl, Y_vl, Ysp_vl = build_windows(r1_val,   pen_scaler, seq_len)

    # ── 6. Build LatentODE model ──────────────────────────────────────────────
    model = LatentODE(
        latent_dim=VAE_LATENT_DIM,
        ode_dim=ode_dim,
        active_dims=active_dims,
        hidden_size=hidden_size,
        n_layers=n_layers,
        activation=activation,
        solver=solver,
    ).to(device)

    print(f"\n{model.describe()}\n")

    # ── 7. Train ──────────────────────────────────────────────────────────────
    train_rmses, val_rmses = train(
        model, Z0_tr, Y_tr, Ysp_tr, Z0_vl, Y_vl, Ysp_vl,
        device=device, max_epochs=max_epochs, seq_len=seq_len,
    )
    plot_training_curve(train_rmses, val_rmses, out_dir, cfg_tag)

    # ── 8. Evaluate on all regimes ────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  EVALUATION — sparse offline measurement points only")
    print(f"{'─'*60}")

    eval_map = [
        ("R1 (in-sample)", encoded["R1"]),
        ("R2 (OOD)",       encoded["R2"]),
        ("R3 (OOD)",       encoded["R3"]),
        ("Fault (OOD)",    encoded["Fault"]),
    ]
    results = {}
    for label, batches in eval_map:
        m = evaluate_regime(model, batches, pen_scaler, device, label, seq_len)
        results[label] = m

    # ── 9. Summary table ──────────────────────────────────────────────────────
    r1_rmse = results["R1 (in-sample)"]["RMSE"]
    sep = "─" * 72
    print(f"\n{'='*72}")
    print(f"  Latent ODE (β={beta}, {cfg_tag})  —  Penicillin RMSE (g/L)")
    print(f"{'='*72}")
    print(f"  {'Split':<22}  {'RMSE':>8}  {'MAE':>8}  {'R²':>8}  {'OOD Δ':>8}")
    print(sep)
    for label in results:
        m   = results[label]
        ood = (m["RMSE"] - r1_rmse) / (r1_rmse + 1e-8) * 100
        ood_s = f"{ood:+.1f}%" if label != "R1 (in-sample)" else "—"
        print(f"  {label:<22}  {m['RMSE']:>8.4f}  {m['MAE']:>8.4f}  "
              f"{m['R2']:>8.4f}  {ood_s:>8}")
    print(f"{'='*72}")

    # ── 10. Trajectory plots (one representative batch per regime) ────────────
    for regime, enc_batches in [
        ("R1",    encoded["R1"]),
        ("Fault", encoded["Fault"]),
    ]:
        if enc_batches:
            rep = enc_batches[len(enc_batches) // 2]
            plot_trajectory(
                model, rep, pen_scaler, device,
                out_path=str(out_dir / f"trajectory_{regime.lower()}_batch{rep['batch_id']+1}.png"),
                seq_len=seq_len,
            )

    plot_summary_bar(results, out_dir, cfg_tag, beta)

    # ── 11. Save checkpoint ───────────────────────────────────────────────────
    ckpt_name = f"latent_ode_beta{beta}_{cfg_tag}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "beta":        beta,
        "ode_dim":     ode_dim,
        "active_dims": active_dims,
        "hidden_size": hidden_size,
        "n_layers":    n_layers,
        "activation":  activation,
        "solver":      solver,
        "seq_len":     seq_len,
        "pen_scaler_mean":  pen_scaler.mean_[0],
        "pen_scaler_scale": pen_scaler.scale_[0],
        "results":     results,
    }, CKPT_DIR / ckpt_name)
    print(f"\n  Checkpoint → {CKPT_DIR / ckpt_name}")
    print(f"  Outputs    → {out_dir}/")
    print(f"  Total time : {(time.time() - t0) / 60:.1f} min")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Latent ODE on frozen β-VAE latents for Penicillin prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv",         default=CSV_PATH)
    parser.add_argument("--beta",        type=float,  default=0.05,
                        help="β value (must have a trained sweep checkpoint)")
    parser.add_argument("--ode-dim",     type=int,    default=VAE_LATENT_DIM,
                        help="ODE state dimensionality. Set to len(--active-dims) "
                             "when using index-based dim selection.")
    parser.add_argument("--active-dims", type=int,    nargs="+", default=None,
                        help="Indices of VAE latent dims to feed the ODE. "
                             "E.g. --active-dims 52 27  (β=0.05 active dims). "
                             "Must satisfy len == --ode-dim.")
    parser.add_argument("--hidden",      type=int,    default=32,
                        help="ODEFunc hidden layer width.")
    parser.add_argument("--n-layers",    type=int,    default=2,
                        help="ODEFunc number of hidden layers.")
    parser.add_argument("--activation",  default="swish", choices=["swish", "tanh", "relu"])
    parser.add_argument("--solver",      default="rk4",   choices=["rk4", "dopri5", "euler"])
    parser.add_argument("--seq-len",     type=int,    default=SEQ_LEN,
                        help="Sliding window length (time steps).")
    parser.add_argument("--max-epochs",  type=int,    default=MAX_EPOCHS)
    args = parser.parse_args()

    # Validate active-dims vs ode-dim
    if args.active_dims is not None and len(args.active_dims) != args.ode_dim:
        parser.error(f"--active-dims has {len(args.active_dims)} elements but "
                     f"--ode-dim={args.ode_dim}. They must match.")

    main(
        csv_path=args.csv,
        beta=args.beta,
        ode_dim=args.ode_dim,
        active_dims=args.active_dims,
        hidden_size=args.hidden,
        n_layers=args.n_layers,
        activation=args.activation,
        solver=args.solver,
        seq_len=args.seq_len,
        max_epochs=args.max_epochs,
    )
