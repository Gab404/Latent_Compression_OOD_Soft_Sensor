#!/usr/bin/env python
"""
train_autoencoder.py
====================
Train a Convolutional Denoising AE (CDAE) or Convolutional VAE (CVAE) on
IndPenSim V3 Raman spectra.

Both models share the identical preprocessing pipeline and training loop;
only the model class and loss function differ:

  CDAE : MSE reconstruction loss (denoising)
  CVAE : ELBO = MSE reconstruction + β · KL divergence

Preprocessing pipeline (identical to CDAERamanEncoderV2 / CVAERamanEncoderV2)
------------------------------------------------------------------------------
  raw Raman (2001 channels)
      → clamp NaN / negatives to 0
      → Savitzky-Golay 1st-derivative  (window=15, poly=2, deriv=1)
      → StandardScaler (fit on training batches only)
  → scaled spectra  (N, 2001)

Data split (identical to downstream LSTM training)
--------------------------------------------------
  seed=42, random permutation of fault-free batches
  Train : 80 %  (72 batches)
  Val   : 20 %  (18 batches)
  Test  : all faulty batches  (10 batches, not used during AE training)

Outputs
-------
  --model cdae:  checkpoints/cdae_best.pt   + checkpoints/cdae_scaler.joblib
  --model cvae:  checkpoints/cvae_best.pt   + checkpoints/cvae_scaler.joblib

Checkpoint format (read by CDAERamanEncoderV2 / CVAERamanEncoderV2):
  {
      "epoch":            <int>,
      "model_state_dict": <OrderedDict>,
      "val_loss":         <float>,
      "input_length":     <int>,
      "latent_dim":       <int>,
  }

Usage
-----
  python train_autoencoder.py --model cdae
  python train_autoencoder.py --model cvae
  python train_autoencoder.py --model cvae --epochs 300 --beta 1.0
  python train_autoencoder.py --model cdae --device cpu
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.autoencoder.model import CDAE_Raman, CVAE_Raman
from src.data.dataset import RAMAN_START_COL_IDX, FAULT_COL

# ── Constants ─────────────────────────────────────────────────────────────────
N_RAMAN_COLS  = 2001          # wavenumber channels used (400–2400 cm⁻¹)
LATENT_DIM    = 64
SG_WINDOW     = 15
SG_POLY       = 2
SG_DERIV      = 1
NOISE_STD     = 0.1           # CDAE denoising noise std
BATCH_SIZE    = 512
N_EPOCHS      = 300
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
LR_FACTOR     = 0.5
LR_PATIENCE   = 15            # ReduceLROnPlateau patience
ES_PATIENCE   = 30            # early stopping patience
GRAD_CLIP     = 5.0
BETA_VAE      = 1.0           # KL weight for CVAE
TRAIN_FRAC    = 0.80
SEED          = 42
MIN_BATCH_LEN = 50


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_sg(X: np.ndarray) -> np.ndarray:
    """Savitzky-Golay first-derivative filter, row-wise."""
    return savgol_filter(
        X.astype(np.float64),
        window_length=SG_WINDOW,
        polyorder=SG_POLY,
        deriv=SG_DERIV,
        axis=1,
    ).astype(np.float32)


def load_raman_splits(
    csv_path: str,
    seed: int = SEED,
    train_frac: float = TRAIN_FRAC,
    min_batch_len: int = MIN_BATCH_LEN,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the CSV, extract Raman spectra, apply the same batch split as the
    downstream LSTM training (seed=42, random, 80/20 fault-free).

    Returns
    -------
    X_train : (N_train, N_RAMAN_COLS)  float32  raw (un-preprocessed)
    X_val   : (N_val,   N_RAMAN_COLS)  float32  raw
    """
    print(f"Reading {csv_path} ...")
    # Only load process cols + fault flag + Raman block to keep memory manageable
    df = pd.read_csv(csv_path)
    print(f"  Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Extract Raman block ───────────────────────────────────────────────────
    all_cols   = df.columns.tolist()
    avail      = len(all_cols) - RAMAN_START_COL_IDX
    n_use      = min(N_RAMAN_COLS, avail)
    raman_cols = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + n_use]
    X_raman    = df[raman_cols].values.astype(np.float32)     # (N_total, 2001)

    fault_col_values = df[FAULT_COL].values if FAULT_COL in df.columns else np.zeros(len(df))
    time_vals        = df["Time (h)"].values

    # ── Detect batch boundaries ───────────────────────────────────────────────
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if time_vals[i] < time_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid
    n_batches = int(bid + 1)
    print(f"  Detected {n_batches} batches")

    # ── Separate fault-free / faulty batches ──────────────────────────────────
    clean_batch_spectra: List[np.ndarray] = []
    # faulty_batch_spectra: List[np.ndarray] = []

    for b_id in range(n_batches):
        mask  = batch_ids == b_id
        x_b   = X_raman[mask]
        fault = fault_col_values[mask]

        if len(x_b) < min_batch_len:
            continue

        is_faulty = bool(np.max(fault) > 0)

        # Filter to valid (non-zero energy) rows
        energy = np.abs(x_b).sum(axis=1)
        valid  = energy > 0
        x_valid = x_b[valid]
        if len(x_valid) < 10:
            continue

        if not is_faulty:
            clean_batch_spectra.append(x_valid)

    n_clean = len(clean_batch_spectra)
    print(f"  Fault-free batches with valid Raman: {n_clean}")

    # ── Same random 80/20 split as downstream LSTM (seed=42) ─────────────────
    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(n_clean)
    n_tr  = int(n_clean * train_frac)
    train_idx = perm[:n_tr]
    val_idx   = perm[n_tr:]

    X_train = np.concatenate([clean_batch_spectra[i] for i in train_idx], axis=0)
    X_val   = np.concatenate([clean_batch_spectra[i] for i in val_idx],   axis=0)

    print(f"  Train spectra: {len(X_train):,}  |  Val spectra: {len(X_val):,}")
    return X_train, X_val


def preprocess(
    X_train_raw: np.ndarray,
    X_val_raw:   np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Clamp → SG first-derivative → StandardScaler.
    Scaler is fit on training data only.
    """
    def _clean(X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(X, 0.0, None)

    print("Preprocessing Raman spectra ...")

    X_tr = _apply_sg(_clean(X_train_raw))
    X_va = _apply_sg(_clean(X_val_raw))

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr).astype(np.float32)
    X_va   = scaler.transform(X_va).astype(np.float32)

    print(f"  Scaled  — train {X_tr.shape}, val {X_va.shape}")
    return X_tr, X_va, scaler


def make_dataloaders(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    ds_train = TensorDataset(torch.from_numpy(X_train))
    ds_val   = TensorDataset(torch.from_numpy(X_val))
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=False)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=True, drop_last=False)
    return dl_train, dl_val


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def cdae_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared reconstruction error."""
    return nn.functional.mse_loss(recon, target)


def cvae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = BETA_VAE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ELBO = reconstruction MSE + β · KL divergence.

    KL = -0.5 · mean( 1 + log_var - mu² - exp(log_var) )
    """
    recon_loss = nn.functional.mse_loss(recon, target)
    kl_loss    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    total      = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


# ─────────────────────────────────────────────────────────────────────────────
# Training functions
# ─────────────────────────────────────────────────────────────────────────────

def train_cdae(
    csv_path:    str,
    ckpt_out:    str,
    scaler_out:  str,
    latent_dim:  int   = LATENT_DIM,
    noise_std:   float = NOISE_STD,
    n_epochs:    int   = N_EPOCHS,
    batch_size:  int   = BATCH_SIZE,
    lr:          float = LR,
    weight_decay: float = WEIGHT_DECAY,
    seed:        int   = SEED,
    device:      str   = "cuda",
) -> None:
    """Train CDAE_Raman and save checkpoint + scaler."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Training CDAE  |  latent={latent_dim}  noise={noise_std}  device={dev}")
    print(f"{'='*60}")

    X_train_raw, X_val_raw = load_raman_splits(csv_path, seed=seed)
    X_train, X_val, scaler = preprocess(X_train_raw, X_val_raw)
    dl_train, dl_val       = make_dataloaders(X_train, X_val, batch_size)

    model = CDAE_Raman(
        input_length=N_RAMAN_COLS,
        latent_dim=latent_dim,
        noise_std=noise_std,
    ).to(dev)

    optim   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                  optim, mode="min", factor=LR_FACTOR,
                  patience=LR_PATIENCE, verbose=True)

    best_val  = float("inf")
    no_improv = 0

    for epoch in range(1, n_epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for (x,) in dl_train:
            x = x.to(dev)
            optim.zero_grad()
            recon = model(x)          # adds noise internally in train mode
            loss  = cdae_loss(recon, x)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
            tr_loss += loss.item() * len(x)
        tr_loss /= len(dl_train.dataset)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for (x,) in dl_val:
                x = x.to(dev)
                recon = model(x)      # no noise in eval mode
                va_loss += cdae_loss(recon, x).item() * len(x)
        va_loss /= len(dl_val.dataset)

        sched.step(va_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{n_epochs}  "
                  f"train={tr_loss:.6f}  val={va_loss:.6f}  "
                  f"lr={optim.param_groups[0]['lr']:.2e}")

        # ── Early stopping / checkpoint ───────────────────────────────────────
        if va_loss < best_val - 1e-7:
            best_val  = va_loss
            no_improv = 0
            Path(ckpt_out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss":         best_val,
                    "input_length":     N_RAMAN_COLS,
                    "latent_dim":       latent_dim,
                },
                ckpt_out,
            )
        else:
            no_improv += 1
            if no_improv >= ES_PATIENCE:
                print(f"  Early stopping at epoch {epoch} (no improvement for {ES_PATIENCE} epochs)")
                break

    joblib.dump(scaler, scaler_out)
    print(f"\n  CDAE best val loss : {best_val:.6f}")
    print(f"  Checkpoint saved   : {ckpt_out}")
    print(f"  Scaler saved       : {scaler_out}")


def train_cvae(
    csv_path:    str,
    ckpt_out:    str,
    scaler_out:  str,
    latent_dim:  int   = LATENT_DIM,
    beta:        float = BETA_VAE,
    n_epochs:    int   = N_EPOCHS,
    batch_size:  int   = BATCH_SIZE,
    lr:          float = LR,
    weight_decay: float = WEIGHT_DECAY,
    seed:        int   = SEED,
    device:      str   = "cuda",
) -> None:
    """Train CVAE_Raman and save checkpoint + scaler."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Training CVAE  |  latent={latent_dim}  beta={beta}  device={dev}")
    print(f"{'='*60}")

    X_train_raw, X_val_raw = load_raman_splits(csv_path, seed=seed)
    X_train, X_val, scaler = preprocess(X_train_raw, X_val_raw)
    dl_train, dl_val       = make_dataloaders(X_train, X_val, batch_size)

    model = CVAE_Raman(
        input_length=N_RAMAN_COLS,
        latent_dim=latent_dim,
    ).to(dev)

    optim   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                  optim, mode="min", factor=LR_FACTOR,
                  patience=LR_PATIENCE, verbose=True)

    best_val  = float("inf")
    no_improv = 0

    for epoch in range(1, n_epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_total = tr_recon = tr_kl = 0.0
        for (x,) in dl_train:
            x = x.to(dev)
            optim.zero_grad()
            recon, mu, log_var = model(x)    # reparameterised in train mode
            loss, rec, kl = cvae_loss(recon, x, mu, log_var, beta=beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
            n = len(x)
            tr_total += loss.item() * n
            tr_recon += rec.item()  * n
            tr_kl    += kl.item()   * n
        N = len(dl_train.dataset)
        tr_total /= N;  tr_recon /= N;  tr_kl /= N

        # ── Validate (use mu only, no sampling) ───────────────────────────────
        model.eval()
        va_total = va_recon = va_kl = 0.0
        with torch.no_grad():
            for (x,) in dl_val:
                x = x.to(dev)
                recon, mu, log_var = model(x)   # eval: uses mu (no sampling)
                loss, rec, kl = cvae_loss(recon, x, mu, log_var, beta=beta)
                n = len(x)
                va_total += loss.item() * n
                va_recon += rec.item()  * n
                va_kl    += kl.item()   * n
        Nv = len(dl_val.dataset)
        va_total /= Nv;  va_recon /= Nv;  va_kl /= Nv

        sched.step(va_total)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{n_epochs}  "
                  f"train={tr_total:.5f} (rec={tr_recon:.5f} kl={tr_kl:.5f})  "
                  f"val={va_total:.5f} (rec={va_recon:.5f} kl={va_kl:.5f})  "
                  f"lr={optim.param_groups[0]['lr']:.2e}")

        # ── Early stopping / checkpoint ───────────────────────────────────────
        if va_total < best_val - 1e-7:
            best_val  = va_total
            no_improv = 0
            Path(ckpt_out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss":         best_val,
                    "input_length":     N_RAMAN_COLS,
                    "latent_dim":       latent_dim,
                },
                ckpt_out,
            )
        else:
            no_improv += 1
            if no_improv >= ES_PATIENCE:
                print(f"  Early stopping at epoch {epoch} (no improvement for {ES_PATIENCE} epochs)")
                break

    joblib.dump(scaler, scaler_out)
    print(f"\n  CVAE best val loss : {best_val:.6f}")
    print(f"  Checkpoint saved   : {ckpt_out}")
    print(f"  Scaler saved       : {scaler_out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CDAE or CVAE Raman autoencoder on IndPenSim V3"
    )
    parser.add_argument("--model",       choices=["cdae", "cvae"], default="cdae",
                        help="Which autoencoder to train (default: cdae)")
    parser.add_argument("--csv",         default="./100_Batches_IndPenSim_V3.csv",
                        help="Path to IndPenSim V3 CSV")
    parser.add_argument("--latent-dim",  type=int,   default=LATENT_DIM)
    parser.add_argument("--epochs",      type=int,   default=N_EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=LR)
    parser.add_argument("--weight-decay",type=float, default=WEIGHT_DECAY)
    parser.add_argument("--noise-std",   type=float, default=NOISE_STD,
                        help="CDAE denoising noise std (ignored for CVAE)")
    parser.add_argument("--beta",        type=float, default=BETA_VAE,
                        help="KL weight β for CVAE ELBO (ignored for CDAE)")
    parser.add_argument("--seed",        type=int,   default=SEED)
    parser.add_argument("--device",      default="cuda",
                        help="Training device: cuda | cpu")
    parser.add_argument("--ckpt",        default=None,
                        help="Override output checkpoint path")
    parser.add_argument("--scaler",      default=None,
                        help="Override output scaler path")
    args = parser.parse_args()

    if args.model == "cdae":
        ckpt_out   = args.ckpt   or "./checkpoints/cdae_best.pt"
        scaler_out = args.scaler or "./checkpoints/cdae_scaler.joblib"
        train_cdae(
            csv_path    = args.csv,
            ckpt_out    = ckpt_out,
            scaler_out  = scaler_out,
            latent_dim  = args.latent_dim,
            noise_std   = args.noise_std,
            n_epochs    = args.epochs,
            batch_size  = args.batch_size,
            lr          = args.lr,
            weight_decay= args.weight_decay,
            seed        = args.seed,
            device      = args.device,
        )
    else:  # cvae
        ckpt_out   = args.ckpt   or "./checkpoints/cvae_best.pt"
        scaler_out = args.scaler or "./checkpoints/cvae_scaler.joblib"
        train_cvae(
            csv_path    = args.csv,
            ckpt_out    = ckpt_out,
            scaler_out  = scaler_out,
            latent_dim  = args.latent_dim,
            beta        = args.beta,
            n_epochs    = args.epochs,
            batch_size  = args.batch_size,
            lr          = args.lr,
            weight_decay= args.weight_decay,
            seed        = args.seed,
            device      = args.device,
        )


if __name__ == "__main__":
    main()
