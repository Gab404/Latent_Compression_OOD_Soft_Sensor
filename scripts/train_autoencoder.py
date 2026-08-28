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
from torch.amp import autocast, GradScaler
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ── Project root on sys.path ──────────────────────────────────────────────────
# __file__ is scripts/train_autoencoder.py → parent is scripts/ → parent.parent is project root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.autoencoder.model import CDAE_Raman, CVAE_Raman, BetaVAE_Raman
from src.autoencoder.beta_schedule import LinearBetaSchedule, CyclicalBetaSchedule
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
    regime_ablation: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the CSV, extract Raman spectra, apply the same batch split as the
    downstream LSTM training (seed=42, random, 80/20 fault-free).

    regime_ablation=True  →  use only R1 (batches 0-29, ids < 30) for training,
                             mirroring Protocol 2 (--regime-ablation) in train.py.

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
    # Protocol 2 (regime_ablation): use only R1, i.e. batch ids 0–29
    r1_only_ids = set(range(30)) if regime_ablation else None

    clean_batch_spectra: List[np.ndarray] = []

    for b_id in range(n_batches):
        mask  = batch_ids == b_id
        x_b   = X_raman[mask]
        fault = fault_col_values[mask]

        if len(x_b) < min_batch_len:
            continue

        is_faulty = bool(np.max(fault) > 0)
        if is_faulty:
            continue

        # Protocol 2: skip batches outside R1
        if r1_only_ids is not None and b_id not in r1_only_ids:
            continue

        # Filter to valid (non-zero energy) rows
        energy = np.abs(x_b).sum(axis=1)
        valid  = energy > 0
        x_valid = x_b[valid]
        if len(x_valid) < 10:
            continue

        clean_batch_spectra.append(x_valid)

    n_clean = len(clean_batch_spectra)
    proto_tag = "R1 only (regime-ablation)" if regime_ablation else "all fault-free"
    print(f"  Fault-free batches with valid Raman ({proto_tag}): {n_clean}")

    # ── Same random 80/20 split as downstream LSTM ────────────────────────────
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
    X_train:    np.ndarray,
    X_val:      np.ndarray,
    batch_size: int,
    device:     torch.device = torch.device("cpu"),
) -> Tuple[DataLoader, DataLoader]:
    """
    Build DataLoaders.  When ``device`` is CUDA the full dataset is pre-moved
    to GPU VRAM so per-batch H2D transfers are eliminated.
    """
    t_train = torch.from_numpy(X_train)
    t_val   = torch.from_numpy(X_val)

    if device.type == "cuda":
        # Pre-move to GPU once — no H2D overhead per batch
        t_train = t_train.to(device, non_blocking=True)
        t_val   = t_val.to(device,   non_blocking=True)
        loader_kw: dict = {"num_workers": 0, "pin_memory": False}
    else:
        loader_kw = {"num_workers": 0, "pin_memory": True}

    ds_train = TensorDataset(t_train)
    ds_val   = TensorDataset(t_val)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          drop_last=False, **loader_kw)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                          drop_last=False, **loader_kw)
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


def beta_vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    β-VAE ELBO with KL summed over latent dims and averaged over the batch.

    Differs from ``cvae_loss`` in the KL estimator:
        cvae_loss   : ``torch.mean(…)``   — mean over *both* batch and latent dims
        beta_vae_loss: ``sum_D / mean_B`` — sum over latent dim, mean over batch

    This matches the standard β-VAE formulation (Higgins et al., 2017) and
    makes the KL term scale with latent_dim, which matters when sweeping β.

    Returns
    -------
    total      : recon_loss + β · kl_loss
    recon_loss : MSE reconstruction term
    kl_loss    : KL divergence (per-sample average, summed across latent dims)
    """
    recon_loss = nn.functional.mse_loss(recon, target)          # scalar mean
    # sum over latent dim D, mean over batch B  → shape ()
    kl_loss    = -0.5 * torch.mean(
        torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    )
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
    regime_ablation: bool = False,
) -> None:
    """Train CDAE_Raman and save checkpoint + scaler."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Training CDAE  |  latent={latent_dim}  noise={noise_std}  device={dev}"
          f"{'  [R1 only]' if regime_ablation else ''}")
    print(f"{'='*60}")

    X_train_raw, X_val_raw = load_raman_splits(csv_path, seed=seed,
                                               regime_ablation=regime_ablation)
    X_train, X_val, scaler = preprocess(X_train_raw, X_val_raw)
    dl_train, dl_val       = make_dataloaders(X_train, X_val, batch_size, dev)

    model = CDAE_Raman(
        input_length=N_RAMAN_COLS,
        latent_dim=latent_dim,
        noise_std=noise_std,
    ).to(dev)

    optim   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                  optim, mode="min", factor=LR_FACTOR,
                  patience=LR_PATIENCE, verbose=False)

    use_amp = (dev.type == "cuda")
    best_val  = float("inf")
    no_improv = 0

    for epoch in range(1, n_epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_loss = torch.zeros(1, device=dev)
        for (x,) in dl_train:
            if x.device != dev:
                x = x.to(dev, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                enabled=use_amp):
                recon = model(x)
            loss  = cdae_loss(recon.float(), x.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
            tr_loss += loss.detach()
        tr_loss_val = (tr_loss / len(dl_train)).item()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        va_loss = torch.zeros(1, device=dev)
        with torch.no_grad():
            for (x,) in dl_val:
                if x.device != dev:
                    x = x.to(dev, non_blocking=True)
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                    enabled=use_amp):
                    recon = model(x)
                va_loss += cdae_loss(recon.float(), x.float()).detach()
        va_loss_val = (va_loss / len(dl_val)).item()

        sched.step(va_loss_val)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{n_epochs}  "
                  f"train={tr_loss_val:.6f}  val={va_loss_val:.6f}  "
                  f"lr={optim.param_groups[0]['lr']:.2e}", flush=True)

        # ── Early stopping / checkpoint ───────────────────────────────────────
        if va_loss_val < best_val - 1e-7:
            best_val  = va_loss_val
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
    regime_ablation: bool = False,
) -> None:
    """Train CVAE_Raman and save checkpoint + scaler."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Training CVAE  |  latent={latent_dim}  beta={beta}  device={dev}"
          f"{'  [R1 only]' if regime_ablation else ''}")
    print(f"{'='*60}")

    X_train_raw, X_val_raw = load_raman_splits(csv_path, seed=seed,
                                               regime_ablation=regime_ablation)
    X_train, X_val, scaler = preprocess(X_train_raw, X_val_raw)
    dl_train, dl_val       = make_dataloaders(X_train, X_val, batch_size, dev)

    model = CVAE_Raman(
        input_length=N_RAMAN_COLS,
        latent_dim=latent_dim,
    ).to(dev)

    optim   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                  optim, mode="min", factor=LR_FACTOR,
                  patience=LR_PATIENCE, verbose=False)

    use_amp   = (dev.type == "cuda")
    best_val  = float("inf")
    no_improv = 0

    for epoch in range(1, n_epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_total = torch.zeros(1, device=dev)
        tr_recon = torch.zeros(1, device=dev)
        tr_kl    = torch.zeros(1, device=dev)
        for (x,) in dl_train:
            if x.device != dev:
                x = x.to(dev, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                enabled=use_amp):
                recon, mu, log_var = model(x)
            loss, rec, kl = cvae_loss(recon.float(), x.float(),
                                      mu.float(), log_var.float(), beta=beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
            tr_total += loss.detach()
            tr_recon += rec.detach()
            tr_kl    += kl.detach()
        N = len(dl_train)
        tr_total_v = (tr_total / N).item()
        tr_recon_v = (tr_recon / N).item()
        tr_kl_v    = (tr_kl    / N).item()

        # ── Validate (use mu only, no sampling) ───────────────────────────────
        model.eval()
        va_total = torch.zeros(1, device=dev)
        va_recon = torch.zeros(1, device=dev)
        va_kl    = torch.zeros(1, device=dev)
        with torch.no_grad():
            for (x,) in dl_val:
                if x.device != dev:
                    x = x.to(dev, non_blocking=True)
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                    enabled=use_amp):
                    recon, mu, log_var = model(x)
                loss, rec, kl = cvae_loss(recon.float(), x.float(),
                                          mu.float(), log_var.float(), beta=beta)
                va_total += loss.detach()
                va_recon += rec.detach()
                va_kl    += kl.detach()
        Nv = len(dl_val)
        va_total_v = (va_total / Nv).item()
        va_recon_v = (va_recon / Nv).item()
        va_kl_v    = (va_kl    / Nv).item()

        sched.step(va_total_v)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{n_epochs}  "
                  f"train={tr_total_v:.5f} (rec={tr_recon_v:.5f} kl={tr_kl_v:.5f})  "
                  f"val={va_total_v:.5f} (rec={va_recon_v:.5f} kl={va_kl_v:.5f})  "
                  f"lr={optim.param_groups[0]['lr']:.2e}", flush=True)

        # ── Early stopping / checkpoint ───────────────────────────────────────
        if va_total_v < best_val - 1e-7:
            best_val  = va_total_v
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


def train_beta_vae(
    csv_path:        str,
    ckpt_out:        str,
    scaler_out:      str,
    latent_dim:      int   = LATENT_DIM,
    max_beta:        float = 4.0,
    anneal_schedule: str   = "linear",
    anneal_epochs:   int   = 100,
    anneal_delay:    int   = 0,
    n_cycles:        int   = 4,
    n_epochs:        int   = N_EPOCHS,
    batch_size:      int   = BATCH_SIZE,
    lr:              float = LR,
    weight_decay:    float = WEIGHT_DECAY,
    seed:            int   = SEED,
    device:          str   = "cuda",
    use_amp:         bool  = False,
    regime_ablation: bool  = False,
) -> None:
    """Train BetaVAE_Raman with β annealing and save checkpoint + scaler."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    # AMP: auto-enable on CUDA unless caller explicitly disables via use_amp=False
    amp_enabled = use_amp and dev.type == "cuda"
    # Default: always use AMP on CUDA (consistent with CDAE/CVAE)
    if not use_amp and dev.type == "cuda":
        amp_enabled = True
    print(f"\n{'='*60}")
    print(f"  Training β-VAE  |  latent={latent_dim}  max_β={max_beta}  AMP={amp_enabled}"
          f"{'  [R1 only]' if regime_ablation else ''}")
    print(f"  schedule={anneal_schedule}  anneal_epochs={anneal_epochs}  "
          f"delay={anneal_delay}  n_cycles={n_cycles}  device={dev}")
    print(f"{'='*60}")

    # ── Build β schedule ──────────────────────────────────────────────────────
    if anneal_schedule == "linear":
        schedule = LinearBetaSchedule(
            max_beta=max_beta,
            anneal_epochs=anneal_epochs,
            delay_epochs=anneal_delay,
        )
    else:  # cyclical
        schedule = CyclicalBetaSchedule(
            max_beta=max_beta,
            total_epochs=n_epochs,
            n_cycles=n_cycles,
            ratio=0.5,
        )
    print(f"  β schedule: {schedule}")

    X_train_raw, X_val_raw = load_raman_splits(csv_path, seed=seed,
                                               regime_ablation=regime_ablation)
    X_train, X_val, scaler = preprocess(X_train_raw, X_val_raw)
    dl_train, dl_val       = make_dataloaders(X_train, X_val, batch_size, dev)

    model = BetaVAE_Raman(
        input_length=N_RAMAN_COLS,
        latent_dim=latent_dim,
    ).to(dev)

    optim   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                  optim, mode="min", factor=LR_FACTOR,
                  patience=LR_PATIENCE, verbose=False)
    grad_scaler = GradScaler("cuda", enabled=amp_enabled)

    best_val        = float("inf")
    no_improv       = 0
    beta_fully_on   = False   # set True once β first reaches max_beta

    for epoch in range(1, n_epochs + 1):
        beta = schedule.get_beta(epoch)

        if not beta_fully_on and beta >= max_beta - 1e-6:
            beta_fully_on = True
            best_val      = float("inf")
            no_improv     = 0
            print(f"  Epoch {epoch}: β reached max ({beta:.4f}) — "
                  f"early stopping and checkpoint tracking now active", flush=True)

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_total = torch.zeros(1, device=dev)
        tr_recon = torch.zeros(1, device=dev)
        tr_kl    = torch.zeros(1, device=dev)
        for (x,) in dl_train:
            if x.device != dev:
                x = x.to(dev, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=amp_enabled):
                recon, mu, log_var = model(x)
                loss, rec, kl = beta_vae_loss(recon, x, mu, log_var, beta=beta)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            grad_scaler.step(optim)
            grad_scaler.update()
            tr_total += loss.detach()
            tr_recon += rec.detach()
            tr_kl    += kl.detach()
        N = len(dl_train)
        tr_total_v = (tr_total / N).item()
        tr_recon_v = (tr_recon / N).item()
        tr_kl_v    = (tr_kl    / N).item()

        # ── Validate (use mu only, no sampling) ───────────────────────────────
        model.eval()
        va_total = torch.zeros(1, device=dev)
        va_recon = torch.zeros(1, device=dev)
        va_kl    = torch.zeros(1, device=dev)
        with torch.no_grad():
            for (x,) in dl_val:
                if x.device != dev:
                    x = x.to(dev, non_blocking=True)
                with autocast("cuda", enabled=amp_enabled):
                    recon, mu, log_var = model(x)
                loss, rec, kl = beta_vae_loss(recon.float(), x.float(),
                                              mu.float(), log_var.float(), beta=beta)
                va_total += loss.detach()
                va_recon += rec.detach()
                va_kl    += kl.detach()
        Nv = len(dl_val)
        va_total_v = (va_total / Nv).item()
        va_recon_v = (va_recon / Nv).item()
        va_kl_v    = (va_kl    / Nv).item()

        # ReduceLROnPlateau tracks reconstruction loss to avoid β noise
        sched.step(va_recon_v)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{n_epochs}  β={beta:.4f}  "
                  f"train={tr_total_v:.5f} (rec={tr_recon_v:.5f} kl={tr_kl_v:.5f})  "
                  f"val={va_total_v:.5f} (rec={va_recon_v:.5f} kl={va_kl_v:.5f})  "
                  f"lr={optim.param_groups[0]['lr']:.2e}", flush=True)

        # ── Early stopping / checkpoint ───────────────────────────────────────
        if beta_fully_on and va_recon_v < best_val - 1e-7:
            best_val  = va_recon_v
            no_improv = 0
            Path(ckpt_out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss":         best_val,
                    "input_length":     N_RAMAN_COLS,
                    "latent_dim":       latent_dim,
                    "beta_schedule": {
                        "type":          anneal_schedule,
                        "max_beta":      max_beta,
                        "anneal_epochs": anneal_epochs,
                        "anneal_delay":  anneal_delay,
                        "n_cycles":      n_cycles,
                        "beta_at_save":  beta,
                    },
                },
                ckpt_out,
            )
        else:
            if beta_fully_on:
                no_improv += 1
                if no_improv >= ES_PATIENCE:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(no improvement for {ES_PATIENCE} epochs)")
                    break

    joblib.dump(scaler, scaler_out)
    print(f"\n  β-VAE best val recon loss : {best_val:.6f}")
    print(f"  Checkpoint saved    : {ckpt_out}")
    print(f"  Scaler saved        : {scaler_out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CDAE, CVAE, or β-VAE Raman autoencoder on IndPenSim V3"
    )
    parser.add_argument("--model",       choices=["cdae", "cvae", "beta_vae"],
                        default="cdae",
                        help="Which autoencoder to train (default: cdae)")
    parser.add_argument("--csv",         default="./100_Batches_IndPenSim_V3.csv",
                        help="Path to IndPenSim V3 CSV")
    parser.add_argument("--latent-dim",  type=int,   default=LATENT_DIM)
    parser.add_argument("--epochs",      type=int,   default=N_EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=LR)
    parser.add_argument("--weight-decay",type=float, default=WEIGHT_DECAY)
    parser.add_argument("--noise-std",   type=float, default=NOISE_STD,
                        help="CDAE denoising noise std (ignored for CVAE/β-VAE)")
    parser.add_argument("--beta",        type=float, default=BETA_VAE,
                        help="Fixed KL weight β for CVAE ELBO (ignored for CDAE/β-VAE)")
    # β-VAE specific arguments
    parser.add_argument("--max-beta",    type=float, default=4.0,
                        help="Maximum β for β-VAE annealing (default: 4.0)")
    parser.add_argument("--anneal-schedule", choices=["linear", "cyclical"],
                        default="linear",
                        help="β annealing schedule for β-VAE (default: linear)")
    parser.add_argument("--anneal-epochs",   type=int, default=100,
                        help="Epochs to ramp β from 0 → max_beta (linear schedule, default: 100)")
    parser.add_argument("--anneal-delay",    type=int, default=0,
                        help="Epochs to keep β=0 before annealing (default: 0)")
    parser.add_argument("--n-cycles",        type=int, default=4,
                        help="Number of annealing cycles (cyclical schedule, default: 4)")
    parser.add_argument("--amp",             action="store_true",
                        help="Enable automatic mixed precision (float16, CUDA only)")
    parser.add_argument("--seed",        type=int,   default=SEED)
    parser.add_argument("--device",      default="cuda",
                        help="Training device: cuda | cpu")
    parser.add_argument("--ckpt",        default=None,
                        help="Override output checkpoint path")
    parser.add_argument("--scaler",      default=None,
                        help="Override output scaler path")
    parser.add_argument("--regime-ablation", action="store_true",
                        help="Train on R1 only (batches 0-29), mirroring Protocol 2")
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
            regime_ablation = args.regime_ablation,
        )
    elif args.model == "cvae":
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
            regime_ablation = args.regime_ablation,
        )
    else:  # beta_vae
        ckpt_out   = args.ckpt   or "./checkpoints/beta_vae_best.pt"
        scaler_out = args.scaler or "./checkpoints/beta_vae_scaler.joblib"
        train_beta_vae(
            csv_path        = args.csv,
            ckpt_out        = ckpt_out,
            scaler_out      = scaler_out,
            latent_dim      = args.latent_dim,
            max_beta        = args.max_beta,
            anneal_schedule = args.anneal_schedule,
            anneal_epochs   = args.anneal_epochs,
            anneal_delay    = args.anneal_delay,
            n_cycles        = args.n_cycles,
            n_epochs        = args.epochs,
            batch_size      = args.batch_size,
            lr              = args.lr,
            weight_decay    = args.weight_decay,
            seed            = args.seed,
            device          = args.device,
            use_amp         = args.amp,
            regime_ablation = args.regime_ablation,
        )


if __name__ == "__main__":
    main()
