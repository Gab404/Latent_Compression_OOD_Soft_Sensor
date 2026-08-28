#!/usr/bin/env python
"""
beta_sweep_e2e.py
=================
End-to-end β-VAE sweep for the β-VAE-PI-LSTM soft-sensor pipeline.

For each β in a user-defined grid:
  1. Train a BetaVAE_Raman encoder on Regime 1 spectra only  (fixed β, no anneal).
  2. Freeze the encoder; pre-encode all batches into _raman_0..63 columns.
  3. Train a BetaVAEPILSTMAlgorithm on Regime 1 (physics-informed LSTM).
  4. Evaluate downstream Penicillin RMSE on Regime 1 val and Regime 4 (Fault).

Advanced metrics computed per β
---------------------------------
  • VAE diagnostics   : raw reconstruction MSE, total KL divergence.
  • Per-dimension KL  : KL_d = E_x[0.5(μ_d² + exp(σ_d²) - 1 - σ_d²)],
                        active-dim count (threshold KL > 0.01).
  • Partial correlation: Pearson r between z_d and each physical target
                        (Biomass, Penicillin), controlling for Time via
                        the OLS residuals method.
  • DCI score         : Disentanglement, Completeness, Informativeness
                        (Eastwood & Williams, 2018) using GradientBoosting
                        importance matrices over 6 physical factors.

Three publication-ready plots
-------------------------------
  1. Rate-distortion Pareto front (KL vs recon MSE, β annotated).
  2. Downstream robustness dual-axis (RMSE R1/Fault + active dims vs β).
  3. Posterior collapse barplot (per-dim KL for β=0, sweet-spot, over-reg).

Usage
-----
  # Full sweep (each β trains fresh encoders + LSTMs):
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/beta_sweep_e2e.py

  # Skip already-trained β values (resume after interruption):
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/beta_sweep_e2e.py --skip-trained

  # Only regenerate plots from saved results CSV:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/beta_sweep_e2e.py --plots-only

  # Custom β grid:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/beta_sweep_e2e.py \\
      --betas 0.0 0.001 0.005 0.01 0.05 0.1 0.5 1.0

Data layout expected
---------------------
  ./100_Batches_IndPenSim_V3.csv   (IndPenSim V3, 100 batches)
  Regime 1 = batch IDs 0–29  (recipe-driven)   → train/val for encoder + LSTM
  Regime 4 = batch IDs 90–99 (fault injection) → OOD test set
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.algorithms  # noqa: F401  — populates REGISTRY
from src.algorithms.base import REGISTRY
from src.autoencoder.model import BetaVAE_Raman
from src.config import Config
from src.data.beta_vae_encoder import BetaVAERamanEncoderV2
from src.data.dataset import FAULT_COL, RAMAN_START_COL_IDX, TARGET_COLS
from src.evaluation.metrics import compute_metrics

# ── Constants ─────────────────────────────────────────────────────────────────
N_RAMAN_COLS   = 2001
LATENT_DIM     = 64
SG_WINDOW, SG_POLY, SG_DERIV = 15, 2, 1
VAE_BATCH_SIZE = 512
VAE_N_EPOCHS   = 300
VAE_LR         = 1e-3
VAE_WD         = 1e-4
VAE_LR_FACTOR  = 0.5
VAE_LR_PAT     = 15
VAE_ES_PAT     = 30
VAE_GRAD_CLIP  = 5.0

LSTM_N_EPOCHS  = 200
SEED           = 42
ACTIVE_KL_THR  = 0.01      # per-dim KL threshold for "active" dimension

REGIME_IDS = {
    "R1":    set(range(0,  30)),
    "R2":    set(range(30, 60)),
    "R3":    set(range(60, 90)),
    "Fault": set(range(90, 100)),
}

# Physical variable patterns used for DCI and partial correlation
_FACTOR_PATTERNS: Dict[str, str] = {
    "Biomass":      r"Biomass|X_offline|X\(g",
    "Penicillin":   r"Penicillin|P_offline|P\(g",
    "Glucose":      r"Glucose|Sugar feed|Fs:",
    "DO2":          r"[Dd]issolved.oxygen|DO2|DO\(",
    "pH":           r"^pH\b|:pH\b",
    "Temperature":  r"Temperature|Temp|:T\(K\)",
}

# PI-LSTM config template (regime-ablation style, physics-informed)
_PILSTM_CFG_DICT = {
    "name": "beta_vae_pi_lstm_sweep",
    "data": {
        "csv_path":           "./100_Batches_IndPenSim_V3.csv",
        "use_raman":          True,
        "raman_encoder_type": "beta_vae",
        "raman_ckpt":         "",      # overridden per β
        "raman_scaler":       "",      # overridden per β
        "raman_latent_dim":   LATENT_DIM,
        "split_strategy":     "random",
        "train_frac":         0.80,
        "seed":               SEED,
        "min_len":            50,
    },
    "model": {
        "hidden_size":     64,
        "num_lstm_layers": 2,
        "fc_hidden":       32,
        "lstm_dropout":    0.2,
        "k_prod_init":     1.0,
    },
    "training": {
        "seq_len":               24,
        "dt":                    0.2,
        "batch_size":            64,
        "n_epochs":              LSTM_N_EPOCHS,
        "lr":                    1e-3,
        "lr_physics":            1e-2,
        "weight_decay":          1e-4,
        "lr_factor":             0.5,
        "lr_patience":           15,
        "es_patience":           30,
        "grad_clip":             5.0,
        "vanilla_mode":          False,
        "lambda_physics_start":  0.0,
        "lambda_physics_end":    0.1,
        "lambda_anneal_delay":   30,
        "lambda_anneal_epochs":  80,
    },
    "output": {
        "checkpoint": "",   # overridden per β
        "plots_dir":  "",   # overridden per β
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 1  —  Data loading & regime split
# ─────────────────────────────────────────────────────────────────────────────

def load_regime_batches(
    csv_path: str,
    min_batch_len: int = 50,
) -> Dict[str, List[pd.DataFrame]]:
    """
    Load IndPenSim V3 CSV and split into regime buckets.

    Adds helper columns expected by the downstream LSTM:
      _batch_id          : integer batch index (0-based)
      _penicillin_sparse : offline Penicillin measurement (NaN between samples)
      _biomass_sparse    : offline Biomass measurement (NaN between samples)

    Target columns are then dense-interpolated (fills NaN for LSTM targets).

    Returns
    -------
    {"R1": [...], "R2": [...], "R3": [...], "Fault": [...]}
    """
    print(f"\nLoading CSV: {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    t_vals    = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid
    df["_batch_id"] = batch_ids

    # Preserve sparse measurement grid before interpolation
    df["_penicillin_sparse"] = df[TARGET_COLS[1]].copy()
    df["_biomass_sparse"]    = df[TARGET_COLS[0]].copy()

    # Dense interpolation for LSTM training targets
    for tc in TARGET_COLS:
        df[tc] = df[tc].interpolate(method="linear").ffill().bfill()

    buckets: Dict[str, List[pd.DataFrame]] = {k: [] for k in REGIME_IDS}
    for b_id in range(int(df["_batch_id"].max()) + 1):
        b = df[df["_batch_id"] == b_id].copy().reset_index(drop=True)
        if len(b) < min_batch_len:
            continue
        for regime, ids in REGIME_IDS.items():
            if b_id in ids:
                buckets[regime].append(b)
                break

    for regime, blist in buckets.items():
        print(f"  {regime}: {len(blist)} batches")
    return buckets


def split_r1_train_val(
    r1_batches: List[pd.DataFrame],
    train_frac: float = 0.80,
    seed: int = SEED,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """Reproducible 80/20 random split of R1 batches (matches train.py logic)."""
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(len(r1_batches))
    n_tr = int(len(r1_batches) * train_frac)
    train = [r1_batches[i] for i in perm[:n_tr]]
    val   = [r1_batches[i] for i in perm[n_tr:]]
    return train, val


def _find_factor_cols(df: pd.DataFrame) -> Dict[str, str]:
    """Return {factor_name: col_name} by regex-matching _FACTOR_PATTERNS."""
    import re
    found = {}
    for name, pattern in _FACTOR_PATTERNS.items():
        for col in df.columns:
            if re.search(pattern, col):
                found[name] = col
                break
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Section 2  —  β-VAE encoder training on R1 spectra
# ─────────────────────────────────────────────────────────────────────────────

def _apply_sg(X: np.ndarray) -> np.ndarray:
    return savgol_filter(
        X.astype(np.float64), window_length=SG_WINDOW,
        polyorder=SG_POLY, deriv=SG_DERIV, axis=1,
    ).astype(np.float32)


def _extract_raman(
    batches: List[pd.DataFrame],
) -> np.ndarray:
    """Extract valid (non-zero energy) Raman rows from a list of batches."""
    parts = []
    for b in batches:
        all_cols   = b.columns.tolist()
        n_use      = min(N_RAMAN_COLS, len(all_cols) - RAMAN_START_COL_IDX)
        raman_cols = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + n_use]
        X          = b[raman_cols].values.astype(np.float32)
        energy     = np.abs(X).sum(axis=1)
        valid      = energy > 0
        if valid.any():
            parts.append(X[valid])
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, N_RAMAN_COLS), np.float32)


def _beta_vae_loss(
    recon: torch.Tensor, target: torch.Tensor,
    mu: torch.Tensor, log_var: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    β-VAE ELBO: recon_MSE + β · KL  (KL summed over D, meaned over B).

    This matches Higgins et al. 2017 — KL scales with latent_dim, making
    β comparable across different D values.
    """
    recon_loss = nn.functional.mse_loss(recon, target)
    kl_loss    = -0.5 * torch.mean(
        torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    )
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_beta_vae_r1(
    train_batches: List[pd.DataFrame],
    val_batches:   List[pd.DataFrame],
    ckpt_out:      str,
    scaler_out:    str,
    beta:          float,
    device:        torch.device,
    n_epochs:      int = VAE_N_EPOCHS,
    seed:          int = SEED,
    anneal_delay:  int = 0,
    anneal_epochs: int = 1,
) -> Dict[str, float]:
    """
    Train BetaVAE_Raman on Regime 1 spectra.

    anneal_delay=0, anneal_epochs=1  →  fixed β from epoch 1 (original behaviour)
    anneal_delay=20, anneal_epochs=100 →  β ramps 0→target over epochs 20-120
        (same schedule as run_seeds.py / train_autoencoder.py, strongly reduces
        the probability of dimensional collapse at low β values such as 0.05).

    Early stopping tracks val reconstruction MSE.
    When annealing, ES is frozen during the β=0 warmup and ramp phases
    (only becomes active after β has reached its target value).

    Returns final best {val_recon_mse, val_kl, epoch}.
    """
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    use_anneal = (anneal_delay > 0 or anneal_epochs > 1)
    anneal_str = (f"anneal delay={anneal_delay} ramp={anneal_epochs}"
                  if use_anneal else "fixed")
    print(f"\n{'─'*60}")
    print(f"  β-VAE encoder  β={beta:.4f}  device={device}  schedule={anneal_str}")
    print(f"{'─'*60}")

    # ── Prepare spectra ────────────────────────────────────────────────────
    X_tr_raw = _extract_raman(train_batches)
    X_va_raw = _extract_raman(val_batches)
    print(f"  R1 spectra — train: {len(X_tr_raw):,}  val: {len(X_va_raw):,}")

    def _clean(X): return np.clip(np.nan_to_num(X, 0.0), 0.0, None)
    X_tr = _apply_sg(_clean(X_tr_raw))
    X_va = _apply_sg(_clean(X_va_raw))
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr).astype(np.float32)
    X_va   = scaler.transform(X_va).astype(np.float32)

    dl_tr = DataLoader(TensorDataset(torch.from_numpy(X_tr)),
                       batch_size=VAE_BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    dl_va = DataLoader(TensorDataset(torch.from_numpy(X_va)),
                       batch_size=VAE_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model  = BetaVAE_Raman(input_length=N_RAMAN_COLS, latent_dim=LATENT_DIM).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=VAE_LR, weight_decay=VAE_WD)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                 optim, mode="min", factor=VAE_LR_FACTOR,
                 patience=VAE_LR_PAT, verbose=False)

    best_recon    = float("inf")
    no_improve    = 0
    beta_fully_on = False   # ES frozen until β has reached its target
    Path(ckpt_out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, n_epochs + 1):
        # ── β schedule ────────────────────────────────────────────────────────
        if not use_anneal:
            beta_now = beta
            beta_fully_on = True
        else:
            epoch_after_delay = epoch - anneal_delay
            if epoch_after_delay <= 0:
                beta_now = 0.0
            elif anneal_epochs <= 1:
                beta_now = beta
            else:
                frac = min((epoch_after_delay - 1) / (anneal_epochs - 1), 1.0)
                beta_now = beta * frac
            if not beta_fully_on and beta_now >= beta - 1e-9:
                beta_fully_on = True
                best_recon    = float("inf")   # reset ES when β is fully ramped
                no_improve    = 0
                print(f"  ep {epoch}: β reached target ({beta:.4f}) — "
                      f"ES and checkpointing now active", flush=True)

        # ── Train epoch ───────────────────────────────────────────────────
        model.train()
        tr_recon = tr_kl = 0.0
        for (x,) in dl_tr:
            x = x.to(device)
            optim.zero_grad()
            recon, mu, lv = model(x)
            _, rec, kl    = _beta_vae_loss(recon, x, mu, lv, beta_now)
            loss = rec + beta_now * kl
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), VAE_GRAD_CLIP)
            optim.step()
            tr_recon += rec.item() * len(x)
            tr_kl    += kl.item()  * len(x)
        N = len(dl_tr.dataset)
        tr_recon /= N; tr_kl /= N

        # ── Val epoch ─────────────────────────────────────────────────────
        model.eval()
        va_recon = va_kl = 0.0
        with torch.no_grad():
            for (x,) in dl_va:
                x = x.to(device)
                recon, mu, lv = model(x)
                _, rec, kl    = _beta_vae_loss(recon, x, mu, lv, beta_now)
                va_recon += rec.item() * len(x)
                va_kl    += kl.item()  * len(x)
        Nv = len(dl_va.dataset)
        va_recon /= Nv; va_kl /= Nv

        sched.step(va_recon)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  ep {epoch:4d}/{n_epochs}  β_now={beta_now:.4f}  "
                  f"tr_rec={tr_recon:.5f} tr_kl={tr_kl:.4f}  "
                  f"va_rec={va_recon:.5f} va_kl={va_kl:.4f}  "
                  f"lr={optim.param_groups[0]['lr']:.1e}", flush=True)

        # ── Checkpoint on reconstruction improvement (only after β is fully on)
        if beta_fully_on and va_recon < best_recon - 1e-7:
            best_recon = va_recon
            no_improve = 0
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":         best_recon,
                "val_kl":           va_kl,
                "input_length":     N_RAMAN_COLS,
                "latent_dim":       LATENT_DIM,
                "beta":             beta,
            }, ckpt_out)
        elif beta_fully_on:
            no_improve += 1
            if no_improve >= VAE_ES_PAT:
                print(f"  Early stop at epoch {epoch}.", flush=True)
                break

    joblib.dump(scaler, scaler_out)
    saved = torch.load(ckpt_out, map_location="cpu", weights_only=False)
    print(f"  Best val recon={saved['val_loss']:.6f}  kl={saved.get('val_kl', float('nan')):.4f}  "
          f"epoch={saved['epoch']}")
    print(f"  Saved → {ckpt_out}")
    return {"val_recon_mse": float(saved["val_loss"]),
            "val_kl":        float(saved.get("val_kl", np.nan)),
            "epoch":         int(saved["epoch"])}


# ─────────────────────────────────────────────────────────────────────────────
# Section 3  —  Pre-encode batches + PI-LSTM training
# ─────────────────────────────────────────────────────────────────────────────

def encode_batches(
    batches:  List[pd.DataFrame],
    encoder:  BetaVAERamanEncoderV2,
) -> List[pd.DataFrame]:
    """
    Apply the frozen encoder to every batch, adding _raman_0 … _raman_63
    columns in-place (copy).  These columns are consumed by
    CVAEPILSTMAlgorithm._build_dataset() during alg.fit().
    """
    encoded = []
    for b in batches:
        b2     = b.copy()
        latents = encoder.encode_dataframe(b2)        # (N, LATENT_DIM)
        for d in range(latents.shape[1]):
            b2[f"_raman_{d}"] = latents[:, d]
        encoded.append(b2)
    return encoded


def train_pilstm_r1(
    train_enc:    List[pd.DataFrame],
    val_enc:      List[pd.DataFrame],
    ckpt_out:     str,
    device:       torch.device,
    lstm_epochs:  Optional[int] = None,
    seq_len:      Optional[int] = None,
) -> object:
    """
    Train BetaVAEPILSTMAlgorithm on the pre-encoded R1 batches.

    The config is built programmatically — no YAML read required.
    The raman_ckpt / raman_scaler fields are left blank because the batches
    already carry _raman_X columns; the algorithm reads those directly.

    Returns the fitted algorithm object.
    """
    cfg_dict = copy.deepcopy(_PILSTM_CFG_DICT)
    cfg_dict["output"]["checkpoint"] = ckpt_out
    cfg_dict["output"]["plots_dir"]  = str(Path(ckpt_out).parent / "plots")
    if lstm_epochs is not None:
        cfg_dict["training"]["n_epochs"] = lstm_epochs
    if seq_len is not None:
        cfg_dict["training"]["seq_len"] = seq_len
    cfg = Config.from_dict(cfg_dict)
    Path(ckpt_out).parent.mkdir(parents=True, exist_ok=True)

    AlgoCls = REGISTRY["beta_vae_pi_lstm"]
    alg     = AlgoCls(cfg, device)

    print(f"\n  Training PI-LSTM  →  {Path(ckpt_out).name}")
    alg.fit(train_enc, val_enc, verbose=False)
    alg.save(ckpt_out)
    print(f"  PI-LSTM saved → {ckpt_out}")
    return alg


# ─────────────────────────────────────────────────────────────────────────────
# Section 4  —  Downstream RMSE evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pilstm(
    alg,
    batches_enc: List[pd.DataFrame],
    label: str = "",
) -> Dict[str, float]:
    """Run alg.evaluate() and return Penicillin metrics."""
    metrics = alg.evaluate(batches_enc)
    rmse    = float(metrics["pen"]["RMSE"])
    mae     = float(metrics["pen"]["MAE"])
    r2      = float(metrics["pen"]["R2"])
    print(f"  [{label}] Pen RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


# ─────────────────────────────────────────────────────────────────────────────
# Section 5  —  VAE diagnostic metrics (recon MSE, per-dim KL)
# ─────────────────────────────────────────────────────────────────────────────

def compute_vae_metrics(
    vae_ckpt:   str,
    vae_scaler: str,
    batches:    List[pd.DataFrame],
    device:     torch.device,
) -> Dict:
    """
    Compute reconstruction MSE, total KL, and per-dimension KL divergence
    by forwarding the R1 val batches through the VAE.

    Per-dim KL (analytical):
        KL_d = E_x [ 0.5 * (μ_d² + exp(σ_d²) - 1 - σ_d²) ]

    Active dims = number of dimensions with KL_d > ACTIVE_KL_THR.

    Returns
    -------
    {
        "recon_mse":  float,
        "total_kl":   float,            # mean over samples, summed over D
        "per_dim_kl": List[float],      # length D
        "active_dims": int,
        "n_samples":  int,
    }
    """
    state = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
    model = BetaVAE_Raman(input_length=N_RAMAN_COLS, latent_dim=LATENT_DIM)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    model.eval()
    scaler = joblib.load(vae_scaler)

    recon_sum, kl_sum, n_total = 0.0, 0.0, 0
    mu_accum     = []   # collect mu and log_var for per-dim KL
    logvar_accum = []

    def _clean(X): return np.clip(np.nan_to_num(X, 0.0), 0.0, None)

    with torch.no_grad():
        for b in batches:
            all_cols   = b.columns.tolist()
            n_use      = min(N_RAMAN_COLS, len(all_cols) - RAMAN_START_COL_IDX)
            raman_cols = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + n_use]
            X_raw      = b[raman_cols].values.astype(np.float32)
            energy     = np.abs(X_raw).sum(axis=1) > 0
            if not energy.any():
                continue
            X_valid  = _clean(X_raw[energy])
            X_sg     = _apply_sg(X_valid)
            X_scaled = scaler.transform(X_sg).astype(np.float32)

            X_t   = torch.tensor(X_scaled, device=device)
            mu, lv = model.encode(X_t)
            recon  = model.decode(mu)     # use mean (no sampling) for clean recon

            rec_mse = nn.functional.mse_loss(recon, X_t, reduction="sum").item()
            kl_sum  += (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(dim=1)).sum().item()
            recon_sum += rec_mse

            mu_accum.append(mu.cpu().numpy())
            logvar_accum.append(lv.cpu().numpy())
            n_total += len(X_valid)

    if n_total == 0:
        return {"recon_mse": np.nan, "total_kl": np.nan,
                "per_dim_kl": [np.nan]*LATENT_DIM, "active_dims": 0, "n_samples": 0}

    MU  = np.concatenate(mu_accum,     axis=0)    # (N, D)
    LV  = np.concatenate(logvar_accum, axis=0)    # (N, D)

    # Analytical per-dim KL:  E_x[0.5*(mu_d^2 + exp(sigma_d^2) - 1 - log_var_d)]
    per_dim_kl = 0.5 * (MU**2 + np.exp(LV) - 1.0 - LV)   # (N, D)
    per_dim_kl = per_dim_kl.mean(axis=0)                   # (D,)

    active_dims = int((per_dim_kl > ACTIVE_KL_THR).sum())

    return {
        "recon_mse":   recon_sum / n_total,
        "total_kl":    kl_sum    / n_total,
        "per_dim_kl":  per_dim_kl.tolist(),
        "active_dims": active_dims,
        "n_samples":   n_total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 6  —  Partial correlation (controlling for Time)
# ─────────────────────────────────────────────────────────────────────────────

def _ols_residuals(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Return residuals of regressing Z (1-D) on X (1-D)."""
    X = X.reshape(-1, 1)
    coef = np.linalg.lstsq(
        np.column_stack([X, np.ones(len(X))]), Z, rcond=None
    )[0]
    return Z - (X[:, 0] * coef[0] + coef[1])


def compute_partial_corr(
    batches:       List[pd.DataFrame],
    encoder:       BetaVAERamanEncoderV2,
    target_cols:   List[str],     # ['Penicillin_col_name', 'Biomass_col_name']
    factor_names:  List[str],     # ['Penicillin', 'Biomass']
) -> Dict[str, np.ndarray]:
    """
    For each active latent dimension d and each physical target t:
        partial_r(z_d, t | Time) = pearsonr(
            residuals(z_d  ~ Time),
            residuals(t    ~ Time)
        )

    Manual residuals method — equivalent to the matrix formula for
    partial correlation with a single control variable.

    Returns
    -------
    {factor_name: np.ndarray shape (D,)}  — partial r per dimension
    """
    time_vals  = []
    z_vals     = []
    target_mat = {fn: [] for fn in factor_names}

    def _clean(X): return np.clip(np.nan_to_num(X, 0.0), 0.0, None)

    for b in batches:
        all_cols   = b.columns.tolist()
        n_use      = min(N_RAMAN_COLS, len(all_cols) - RAMAN_START_COL_IDX)
        raman_cols = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + n_use]
        X_raw      = b[raman_cols].values.astype(np.float32)
        energy     = np.abs(X_raw).sum(axis=1) > 0
        if not energy.any():
            continue

        scaler    = encoder._scaler
        X_valid   = _clean(X_raw[energy])
        X_sg      = _apply_sg(X_valid)
        X_scaled  = scaler.transform(X_sg).astype(np.float32)
        mu        = encoder.encode(X_scaled)          # (n_valid, D)

        t_all = b["Time (h)"].values
        z_vals.append(mu)
        time_vals.append(t_all[energy])

        for fn, col in zip(factor_names, target_cols):
            target_vals = b[col].values.astype(np.float64)
            target_mat[fn].append(target_vals[energy])

    if not z_vals:
        return {fn: np.full(LATENT_DIM, np.nan) for fn in factor_names}

    Z    = np.concatenate(z_vals,   axis=0).astype(np.float64)  # (N, D)
    time = np.concatenate(time_vals, axis=0).astype(np.float64)  # (N,)
    targets = {fn: np.concatenate(v, axis=0).astype(np.float64)
               for fn, v in target_mat.items()}

    # Time residuals of latent dims
    Z_resid = np.column_stack([_ols_residuals(time, Z[:, d]) for d in range(Z.shape[1])])

    result = {}
    for fn in factor_names:
        y       = targets[fn]
        valid   = np.isfinite(y) & np.isfinite(Z_resid).all(axis=1)
        y_resid = _ols_residuals(time[valid], y[valid])
        r_vals  = np.array([
            pearsonr(Z_resid[valid, d], y_resid)[0]
            for d in range(Z_resid.shape[1])
        ])
        result[fn] = r_vals

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Section 7  —  DCI score (Eastwood & Williams 2018)
# ─────────────────────────────────────────────────────────────────────────────

def compute_dci(
    batches:      List[pd.DataFrame],
    encoder:      BetaVAERamanEncoderV2,
    factor_cols:  List[str],
    factor_names: List[str],
    seed:         int = SEED,
) -> Dict:
    """
    Disentanglement, Completeness, Informativeness (DCI).

    For each factor v_k, fit a GradientBoostingRegressor f_k: Z → v_k.
    The feature importances give the importance matrix R ∈ R^{K × D}.

    Disentanglement D_d = (1 - H_K(R_norm_col[:,d])) weighted by total
    column importance.  Completeness C_k = 1 - H_D(R_norm_row[k,:]).
    Informativeness I_k = R² of f_k on held-out 20 % split.

    D, C ∈ [0, 1] (1 = perfectly disentangled / complete).
    I ∈ [0, 1]   (1 = perfectly predictive latent space).
    """
    def _clean(X): return np.clip(np.nan_to_num(X, 0.0), 0.0, None)

    z_parts  = []
    tgt_parts = {fn: [] for fn in factor_names}

    for b in batches:
        all_cols   = b.columns.tolist()
        n_use      = min(N_RAMAN_COLS, len(all_cols) - RAMAN_START_COL_IDX)
        raman_cols = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + n_use]
        X_raw      = b[raman_cols].values.astype(np.float32)
        energy     = np.abs(X_raw).sum(axis=1) > 0
        if not energy.any():
            continue
        X_valid  = _clean(X_raw[energy])
        X_sg     = _apply_sg(X_valid)
        X_scaled = encoder._scaler.transform(X_sg).astype(np.float32)
        mu       = encoder.encode(X_scaled)
        z_parts.append(mu)
        for fn, col in zip(factor_names, factor_cols):
            tgt_parts[fn].append(b[col].values.astype(np.float64)[energy])

    if not z_parts:
        return {"D": np.nan, "C": np.nan, "I": np.nan}

    Z = np.concatenate(z_parts, axis=0).astype(np.float64)
    K, D = len(factor_names), Z.shape[1]

    rng       = np.random.default_rng(seed)
    perm      = rng.permutation(len(Z))
    n_tr      = int(0.8 * len(Z))
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]

    R          = np.zeros((K, D))
    info_scores = np.zeros(K)

    for ki, fn in enumerate(factor_names):
        v = np.concatenate(tgt_parts[fn], axis=0).astype(np.float64)
        valid = np.isfinite(v) & np.isfinite(Z).all(axis=1)
        if valid.sum() < 50:
            print(f"    DCI: factor '{fn}' skipped (only {valid.sum()} valid samples)")
            continue
        v_tr = v[valid][tr_idx[tr_idx < valid.sum()]]
        z_tr = Z[valid][tr_idx[tr_idx < valid.sum()]]
        v_te = v[valid][te_idx[te_idx < valid.sum()]]
        z_te = Z[valid][te_idx[te_idx < valid.sum()]]

        gbr = GradientBoostingRegressor(
            n_estimators=100, max_depth=2, min_samples_leaf=5,
            random_state=seed, subsample=0.8,
        )
        gbr.fit(z_tr, v_tr)
        R[ki, :]       = gbr.feature_importances_
        info_scores[ki] = max(0.0, gbr.score(z_te, v_te))

    # ── Disentanglement D ──────────────────────────────────────────────────
    # For each latent dim d, how concentrated is its importance across factors?
    col_totals = R.sum(axis=0)                                  # (D,)
    R_norm_col = R / (col_totals[None, :] + 1e-10)             # (K, D)  sum over K == 1

    def entropy_nats(p: np.ndarray) -> float:
        p = np.clip(p, 1e-10, None); return -float((p * np.log(p)).sum())

    max_H_K = np.log(K) if K > 1 else 1.0
    D_d = np.array([1.0 - entropy_nats(R_norm_col[:, d]) / max_H_K for d in range(D)])
    D_score = float((D_d * col_totals).sum() / (col_totals.sum() + 1e-10))

    # ── Completeness C ─────────────────────────────────────────────────────
    # For each factor k, how concentrated is it on one latent dim?
    max_H_D = np.log(D) if D > 1 else 1.0
    R_norm_row = R / (R.sum(axis=1, keepdims=True) + 1e-10)    # (K, D) sum over D == 1
    C_k = np.array([1.0 - entropy_nats(R_norm_row[ki, :]) / max_H_D for ki in range(K)])
    C_score = float(C_k.mean())

    # ── Informativeness I ──────────────────────────────────────────────────
    valid_info = info_scores[np.isfinite(info_scores) & (info_scores >= 0)]
    I_score = float(valid_info.mean()) if len(valid_info) > 0 else np.nan

    return {
        "D":              D_score,
        "C":              C_score,
        "I":              I_score,
        "D_per_dim":      D_d.tolist(),
        "C_per_factor":   C_k.tolist(),
        "I_per_factor":   info_scores.tolist(),
        "importance_matrix": R.tolist(),
        "factor_names":   factor_names,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 8  —  Plots
# ─────────────────────────────────────────────────────────────────────────────

def _beta_label(b: float) -> str:
    if b == 0.0:   return "0"
    if b < 0.01:   return f"{b:.3f}"
    if b < 1.0:    return f"{b:.2f}"
    return f"{b:.1f}"


def plot_rate_distortion(results: List[Dict], out_dir: Path) -> None:
    """
    Rate-distortion Pareto front: KL (x) vs reconstruction MSE (y).
    Points annotated with β.  Visualises the fundamental reconstruction–
    regularisation trade-off and highlights the Pareto knee.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    kl_vals    = [r["total_kl"]    for r in results]
    recon_vals = [r["recon_mse"]   for r in results]
    betas      = [r["beta"]        for r in results]

    # Sort by KL for a connected line
    order = np.argsort(kl_vals)
    kl_s  = [kl_vals[i]    for i in order]
    rc_s  = [recon_vals[i] for i in order]
    be_s  = [betas[i]      for i in order]

    ax.plot(kl_s, rc_s, "-o", color="#2176ae", linewidth=1.5, markersize=7,
            markerfacecolor="white", markeredgewidth=1.8)

    for kl, rc, b in zip(kl_s, rc_s, be_s):
        ax.annotate(
            f"β={_beta_label(b)}",
            xy=(kl, rc), xytext=(5, 5), textcoords="offset points",
            fontsize=8, color="#333333",
        )

    ax.set_xlabel("Total KL Divergence  (nats per sample)", fontsize=11)
    ax.set_ylabel("Reconstruction MSE", fontsize=11)
    ax.set_title("Rate-Distortion Pareto Front\n(Raman β-VAE, trained on Regime 1)",
                 fontsize=11)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"rate_distortion.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir}/rate_distortion.[pdf/png]")


def plot_robustness(results: List[Dict], out_dir: Path) -> None:
    """
    Dual-axis downstream robustness plot.
    Left  axis  (Y1) : PI-LSTM RMSE for Regime 1 val (blue) and Fault (red).
    Right axis  (Y2) : Number of active latent dimensions (dashed green).
    X-axis: β (log scale).
    """
    betas       = [r["beta"]            for r in results]
    rmse_r1     = [r["lstm_rmse_r1val"] for r in results]
    rmse_fault  = [r["lstm_rmse_fault"] for r in results]
    active_dims = [r["active_dims"]     for r in results]

    # Use 1e-4 as x-value for β=0 so log scale works
    x_vals = [b if b > 0 else 1e-5 for b in betas]
    x_labels = [_beta_label(b) for b in betas]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(x_vals, rmse_r1,    "o-",  color="#2176ae", linewidth=1.8,
             markersize=7, label="RMSE — R1 val")
    ax1.plot(x_vals, rmse_fault, "s-",  color="#c0392b", linewidth=1.8,
             markersize=7, label="RMSE — Fault (OOD)")

    ax2.plot(x_vals, active_dims, "^--", color="#27ae60", linewidth=1.5,
             markersize=7, label="Active latent dims")

    ax1.set_xscale("log")
    ax1.set_xticks(x_vals)
    ax1.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
    ax1.set_xlabel("β  (log scale)", fontsize=11)
    ax1.set_ylabel("Downstream Penicillin RMSE (g/L)", fontsize=11)
    ax2.set_ylabel("Active Latent Dimensions  (KL > 0.01)", fontsize=11, color="#27ae60")
    ax2.tick_params(axis="y", labelcolor="#27ae60")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.set_title("Downstream Robustness vs β\n(PI-LSTM trained on Regime 1, OOD test = Regime 4)",
                  fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=9)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"robustness.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir}/robustness.[pdf/png]")


def plot_per_dim_kl(
    results:        List[Dict],
    highlight_betas: Tuple[float, float, float],
    out_dir:        Path,
) -> None:
    """
    Grouped barplot showing per-latent-dimension KL divergence for three
    selected β values: β=0 (no regularisation / CDAE-like),
    β=sweet-spot, β=over-regularised.

    Dimensions are sorted by descending KL of the sweet-spot model so that
    the "most informative" dimensions appear on the left.
    """
    # Find the three requested rows
    def _find(b: float) -> Optional[Dict]:
        best = min(results, key=lambda r: abs(r["beta"] - b))
        if abs(best["beta"] - b) < 1e-9:
            return best
        return None

    b_none, b_sweet, b_over = highlight_betas
    rows = {}
    for label, b in [("β=0 (CDAE-like)", b_none),
                     (f"β={_beta_label(b_sweet)} (sweet spot)", b_sweet),
                     (f"β={_beta_label(b_over)} (over-reg.)", b_over)]:
        r = _find(b)
        if r is None:
            print(f"  plot_per_dim_kl: β={b} not found in results; skipping.")
            return
        rows[label] = np.array(r["per_dim_kl"])

    # Sort dims by sweet-spot KL descending
    sweet_kl  = rows[f"β={_beta_label(b_sweet)} (sweet spot)"]
    sort_order = np.argsort(-sweet_kl)

    D = len(sweet_kl)
    x = np.arange(D)

    colors = ["#95a5a6", "#2176ae", "#c0392b"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    for ax, (label, kl_vals), color in zip(axes, rows.items(), colors):
        kl_sorted = kl_vals[sort_order]
        bar_colors = [color if kl > ACTIVE_KL_THR else "#dfe6e9" for kl in kl_sorted]
        ax.bar(x, kl_sorted, color=bar_colors, edgecolor="none", width=0.8)
        ax.axhline(ACTIVE_KL_THR, color="#e74c3c", linewidth=0.8, linestyle="--",
                   label=f"Active threshold ({ACTIVE_KL_THR})")
        n_active = int((kl_sorted > ACTIVE_KL_THR).sum())
        ax.set_title(f"{label}  —  {n_active} active dimensions", fontsize=10)
        ax.set_ylabel("KL per dim", fontsize=9)
        ax.set_yscale("symlog", linthresh=ACTIVE_KL_THR)
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Latent Dimension Index  (sorted by sweet-spot KL desc.)", fontsize=10)
    axes[-1].set_xticks(x[::4])
    axes[-1].set_xticklabels(sort_order[::4], rotation=45, ha="right", fontsize=7)

    fig.suptitle("Posterior Collapse Analysis: Per-Dimension KL Divergence", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"per_dim_kl.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir}/per_dim_kl.[pdf/png]")


# ─────────────────────────────────────────────────────────────────────────────
# Section 9  —  Sweep orchestration
# ─────────────────────────────────────────────────────────────────────────────

def _tag(beta: float) -> str:
    """File-system-safe β tag, e.g. 'b0.005'."""
    return f"b{beta:.5f}".rstrip("0").rstrip(".")


def run_sweep(
    betas:        List[float],
    csv_path:     str,
    out_root:     Path,
    device:       torch.device,
    skip_trained: bool = False,
    plots_only:   bool = False,
    vae_epochs:   int  = VAE_N_EPOCHS,
    lstm_epochs:  Optional[int] = None,
    seq_len:      Optional[int] = None,
    vae_src_root: Optional[Path] = None,
    seed:         int  = SEED,
    vae_anneal_delay:  int = 0,
    vae_anneal_epochs: int = 1,
) -> None:
    """Main sweep loop."""
    out_root.mkdir(parents=True, exist_ok=True)
    results_csv = out_root / "sweep_results.csv"
    results_json = out_root / "sweep_results.json"

    # ── Load & split data (once) ───────────────────────────────────────────
    regime_batches = load_regime_batches(csv_path)
    r1_all         = regime_batches["R1"]
    fault_batches  = regime_batches["Fault"]
    r1_train, r1_val = split_r1_train_val(r1_all, seed=seed)

    print(f"\n  R1 split → train: {len(r1_train)} batches  val: {len(r1_val)} batches")
    print(f"  Fault set: {len(fault_batches)} batches")

    # ── Identify physical factor columns ──────────────────────────────────
    example_df   = r1_val[0]
    factor_map   = _find_factor_cols(example_df)
    factor_names = list(factor_map.keys())
    factor_cols  = [factor_map[fn] for fn in factor_names]
    print(f"  Physical factors found: {factor_names}")

    all_results: List[Dict] = []

    # Load existing results if available
    if results_json.exists():
        existing = json.loads(results_json.read_text())
        all_results = existing
        print(f"\n  Loaded {len(all_results)} existing result(s) from {results_json}")

    if plots_only:
        _generate_all_plots(all_results, betas, out_root)
        return

    already_done = {r["beta"] for r in all_results}

    for beta in betas:
        if skip_trained and beta in already_done:
            print(f"\n  β={beta} — found in existing results, skipping.")
            continue

        tag    = _tag(beta)
        ckpt_dir   = out_root / "checkpoints" / tag
        # When vae_src_root is provided, load the frozen VAE from there
        # (allows reusing existing encoders with a different LSTM seq_len)
        vae_src_dir = (vae_src_root / "checkpoints" / tag) if vae_src_root else ckpt_dir
        vae_ckpt   = str(vae_src_dir / "vae.pt")
        vae_scaler = str(vae_src_dir / "vae_scaler.joblib")
        lstm_ckpt  = str(ckpt_dir / "lstm.pt")
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  β = {beta}   tag = {tag}")
        print(f"{'='*70}")
        t_start = time.time()

        row: Dict = {"beta": beta, "tag": tag}

        # ── 1. Train β-VAE encoder ─────────────────────────────────────────
        vae_complete = Path(vae_ckpt).exists() and Path(vae_scaler).exists()
        if vae_src_root:
            # VAE source dir specified: always skip VAE training, load from there
            if not vae_complete:
                raise FileNotFoundError(
                    f"vae-src-dir specified but checkpoint not found: {vae_ckpt}\n"
                    f"Run the full sweep first without --vae-src-dir."
                )
            print(f"  [vae-src] Loading VAE from {vae_ckpt}")
            state = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
            row.update({"val_recon_mse": float(state["val_loss"]),
                        "val_kl":        float(state.get("val_kl", np.nan)),
                        "vae_epoch":     int(state["epoch"])})
        elif skip_trained and vae_complete:
            print(f"  [skip] VAE checkpoint exists: {vae_ckpt}")
            state = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
            row.update({"val_recon_mse": float(state["val_loss"]),
                        "val_kl":        float(state.get("val_kl", np.nan)),
                        "vae_epoch":     int(state["epoch"])})
        else:
            vae_metrics = train_beta_vae_r1(
                r1_train, r1_val, vae_ckpt, vae_scaler, beta, device,
                n_epochs=vae_epochs, seed=seed,
                anneal_delay=vae_anneal_delay,
                anneal_epochs=vae_anneal_epochs,
            )
            row.update({"val_recon_mse": vae_metrics["val_recon_mse"],
                        "val_kl":        vae_metrics["val_kl"],
                        "vae_epoch":     vae_metrics["epoch"]})

        # ── 2. Load frozen encoder, pre-encode batches ─────────────────────
        torch.cuda.empty_cache()
        encoder = BetaVAERamanEncoderV2(vae_ckpt, vae_scaler, device)

        print("  Pre-encoding batches ...")
        r1_train_enc = encode_batches(r1_train, encoder)
        r1_val_enc   = encode_batches(r1_val,   encoder)
        fault_enc    = encode_batches(fault_batches, encoder)

        # ── 3. Train PI-LSTM ───────────────────────────────────────────────
        if skip_trained and Path(lstm_ckpt).exists():
            print(f"  [skip] LSTM checkpoint exists: {lstm_ckpt}")
            from src.algorithms.base import BaseAlgorithm
            alg = BaseAlgorithm.load(lstm_ckpt, device)
        else:
            alg = train_pilstm_r1(r1_train_enc, r1_val_enc, lstm_ckpt, device,
                                   lstm_epochs=lstm_epochs, seq_len=seq_len)

        # ── 4. Evaluate downstream RMSE ────────────────────────────────────
        torch.cuda.empty_cache()
        m_val   = evaluate_pilstm(alg, r1_val_enc,  label="R1 val")
        m_fault = evaluate_pilstm(alg, fault_enc,   label="Fault")
        row.update({
            "lstm_rmse_r1val":  m_val["RMSE"],
            "lstm_mae_r1val":   m_val["MAE"],
            "lstm_r2_r1val":    m_val["R2"],
            "lstm_rmse_fault":  m_fault["RMSE"],
            "lstm_mae_fault":   m_fault["MAE"],
            "lstm_r2_fault":    m_fault["R2"],
            "ood_degradation":  (m_fault["RMSE"] - m_val["RMSE"]) / (m_val["RMSE"] + 1e-8),
        })

        # ── 5. VAE diagnostics (recon MSE, per-dim KL) ────────────────────
        print("  Computing VAE diagnostics ...")
        torch.cuda.empty_cache()
        vae_diag = compute_vae_metrics(vae_ckpt, vae_scaler, r1_val, device)
        row.update({
            "recon_mse":   vae_diag["recon_mse"],
            "total_kl":    vae_diag["total_kl"],
            "per_dim_kl":  vae_diag["per_dim_kl"],
            "active_dims": vae_diag["active_dims"],
            "n_vae_samples": vae_diag["n_samples"],
        })
        print(f"  recon_MSE={vae_diag['recon_mse']:.5f}  "
              f"total_KL={vae_diag['total_kl']:.3f}  "
              f"active_dims={vae_diag['active_dims']}")

        # ── 6. Partial correlation ─────────────────────────────────────────
        print("  Computing partial correlations ...")
        # Use Biomass and Penicillin as targets (dense interpolated cols)
        bio_col = factor_map.get("Biomass",   TARGET_COLS[0])
        pen_col = factor_map.get("Penicillin", TARGET_COLS[1])
        pcorr = compute_partial_corr(
            r1_val, encoder,
            target_cols  = [pen_col, bio_col],
            factor_names = ["Penicillin", "Biomass"],
        )
        row["partial_corr_pen"]  = pcorr["Penicillin"].tolist()
        row["partial_corr_bio"]  = pcorr["Biomass"].tolist()
        row["max_pcorr_pen"]     = float(np.abs(pcorr["Penicillin"]).max())
        row["max_pcorr_bio"]     = float(np.abs(pcorr["Biomass"]).max())
        print(f"  max |partial r| — Penicillin: {row['max_pcorr_pen']:.3f}  "
              f"Biomass: {row['max_pcorr_bio']:.3f}")

        # ── 7. DCI score ───────────────────────────────────────────────────
        print("  Computing DCI (GradientBoosting) ...")
        dci = compute_dci(
            r1_all, encoder,         # use ALL R1 for more stable GBT fits
            factor_cols  = factor_cols,
            factor_names = factor_names,
            seed         = seed,
        )
        row["dci_D"] = dci["D"]
        row["dci_C"] = dci["C"]
        row["dci_I"] = dci["I"]
        print(f"  DCI — D={dci['D']:.3f}  C={dci['C']:.3f}  I={dci['I']:.3f}")

        elapsed = time.time() - t_start
        row["elapsed_s"] = elapsed
        print(f"\n  β={beta} done in {elapsed/60:.1f} min")

        # ── Persist results ────────────────────────────────────────────────
        # Update or append in all_results
        all_results = [r for r in all_results if r["beta"] != beta]
        all_results.append(row)
        all_results.sort(key=lambda r: r["beta"])

        # JSON (full — includes per_dim_kl lists, importance matrices, etc.)
        results_json.write_text(
            json.dumps(all_results, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
        )

        # CSV (scalar metrics only — for quick inspection)
        scalar_keys = [k for k in row if not isinstance(row.get(k), list)]
        with open(results_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=scalar_keys)
            writer.writeheader()
            for r in all_results:
                writer.writerow({k: r.get(k, "") for k in scalar_keys})

        print(f"  Results saved → {results_json.name}")

        # Free GPU memory before next β
        del alg, encoder
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"  Sweep complete — {len(all_results)} β values evaluated")
    print(f"{'='*70}")

    _generate_all_plots(all_results, betas, out_root)
    _print_summary_table(all_results)


def _generate_all_plots(
    results:  List[Dict],
    betas:    List[float],
    out_root: Path,
) -> None:
    plots_dir = out_root / "plots"

    # Only plot β values that have complete scalar metrics
    complete = [r for r in results
                if all(k in r and r[k] is not None
                       for k in ("total_kl", "recon_mse", "lstm_rmse_r1val",
                                 "lstm_rmse_fault", "active_dims"))]
    if len(complete) < 2:
        print("  Not enough complete results to generate plots yet.")
        return

    print("\n  Generating publication plots ...")
    plot_rate_distortion(complete, plots_dir)
    plot_robustness(complete, plots_dir)

    # For the per-dim KL barplot, pick β=0, sweet-spot (~0.005), over-reg (0.1)
    avail_betas = sorted(r["beta"] for r in complete)
    b_none  = min(avail_betas, key=lambda b: abs(b - 0.0))
    b_sweet = min(avail_betas, key=lambda b: abs(b - 0.005))
    b_over  = min(avail_betas, key=lambda b: abs(b - 0.1))

    # Only barplot if all 3 are present AND have per_dim_kl
    for_barplot = [r for r in complete
                   if r["beta"] in {b_none, b_sweet, b_over} and "per_dim_kl" in r]
    if len(for_barplot) >= 3:
        plot_per_dim_kl(complete, (b_none, b_sweet, b_over), plots_dir)
    else:
        print(f"  per_dim_kl barplot skipped (need 3 β values with per_dim_kl; "
              f"found {len(for_barplot)})")


def _print_summary_table(results: List[Dict]) -> None:
    print(f"\n{'='*100}")
    print(f"  {'β':>8}  {'Recon MSE':>10}  {'Total KL':>9}  "
          f"{'Active':>6}  {'RMSE R1':>8}  {'RMSE Fault':>10}  "
          f"{'OOD Δ%':>7}  {'DCI-D':>6}  {'DCI-C':>6}  {'DCI-I':>6}  "
          f"{'maxR_Pen':>8}  {'maxR_Bio':>8}")
    print(f"  {'─'*96}")
    for r in sorted(results, key=lambda x: x["beta"]):
        print(
            f"  {r['beta']:>8.4f}  "
            f"{r.get('recon_mse', float('nan')):>10.5f}  "
            f"{r.get('total_kl', float('nan')):>9.3f}  "
            f"{r.get('active_dims', '?'):>6}  "
            f"{r.get('lstm_rmse_r1val', float('nan')):>8.4f}  "
            f"{r.get('lstm_rmse_fault', float('nan')):>10.4f}  "
            f"{r.get('ood_degradation', float('nan'))*100:>6.1f}%  "
            f"{r.get('dci_D', float('nan')):>6.3f}  "
            f"{r.get('dci_C', float('nan')):>6.3f}  "
            f"{r.get('dci_I', float('nan')):>6.3f}  "
            f"{r.get('max_pcorr_pen', float('nan')):>8.3f}  "
            f"{r.get('max_pcorr_bio', float('nan')):>8.3f}"
        )
    print(f"{'='*100}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Section 10  —  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end β-VAE sweep with regime-ablation "
                    "(train R1, test R4/Fault)"
    )
    parser.add_argument(
        "--betas", nargs="+", type=float,
        default=[0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        help="β values to sweep (default: 8-point log grid 0…1)",
    )
    parser.add_argument(
        "--csv",  default="./100_Batches_IndPenSim_V3.csv",
        help="Path to IndPenSim V3 CSV",
    )
    parser.add_argument(
        "--out-dir", default="./outputs/beta_sweep_e2e",
        help="Root output directory for checkpoints, results, and plots",
    )
    parser.add_argument(
        "--vae-epochs", type=int, default=VAE_N_EPOCHS,
        help=f"Max training epochs for each β-VAE encoder (default: {VAE_N_EPOCHS})",
    )
    parser.add_argument(
        "--lstm-epochs", type=int, default=None,
        help="Override LSTM training epochs (default: uses config value)",
    )
    parser.add_argument(
        "--seq-len", type=int, default=None,
        help="Override LSTM sequence length / context window (default: 24). "
             "Use with --vae-src-dir to reuse a trained encoder.",
    )
    parser.add_argument(
        "--vae-src-dir", default=None,
        help="Load frozen VAE checkpoints from this directory instead of "
             "training a new one. Useful for reusing an existing encoder "
             "with a different --seq-len or --out-dir. "
             "E.g. --vae-src-dir ./outputs/beta_sweep_e2e",
    )
    parser.add_argument(
        "--skip-trained", action="store_true",
        help="Skip β values whose checkpoints already exist (useful for resuming)",
    )
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Skip all training; reload saved sweep_results.json and regenerate plots",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed",   type=int, default=SEED)
    parser.add_argument(
        "--vae-anneal-delay", type=int, default=0,
        help="Epochs to keep β=0 before annealing (default 0 = fixed β). "
             "Use 20 to match run_seeds.py protocol.",
    )
    parser.add_argument(
        "--vae-anneal-epochs", type=int, default=1,
        help="Epochs to ramp β from 0 to target (default 1 = fixed β). "
             "Use 100 to match run_seeds.py protocol.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nβ-sweep e2e  |  device={device}  |  β values: {args.betas}")

    run_sweep(
        betas        = args.betas,
        csv_path     = args.csv,
        out_root     = Path(args.out_dir),
        device       = device,
        skip_trained = args.skip_trained,
        plots_only   = args.plots_only,
        vae_epochs   = args.vae_epochs,
        lstm_epochs  = args.lstm_epochs,
        seq_len      = args.seq_len,
        vae_src_root = Path(args.vae_src_dir) if args.vae_src_dir else None,
        seed         = args.seed,
        vae_anneal_delay  = args.vae_anneal_delay,
        vae_anneal_epochs = args.vae_anneal_epochs,
    )


if __name__ == "__main__":
    main()
