#!/usr/bin/env python
"""
shap_analysis.py
================
SHAP-based explainability for the CVAE-PI-LSTM / β-VAE-PI-LSTM penicillin
soft-sensor.

Maps SHAP values back to the original 2001 Raman wavenumber axis so that
chemically interpretable spectral regions can be identified.

Two outputs are produced in the output directory:
  1. global_importance.png  — mean |SHAP| per wavenumber (bar chart, top-50)
  2. temporal_heatmap.png   — SHAP(T=24, top-K wavenumbers) for one test sample

End-to-end differentiable chain for SHAP gradient backpropagation
-----------------------------------------------------------------
  raw Raman  (B, T, 2001)
  → clamp(min=0)                  # ReLU-like; approximate gradient used
  → FrozenSGFilter                # SG d=1, w=15, p=2  →  frozen nn.Conv1d
  → FrozenStandardScaler          # (x - mu) / sigma, frozen buffers
  → <encoder>.encode() → mu       # NO reparameterisation; mu only
  → FrozenMinMaxScaler            # (x - min) / range, frozen buffers
  → CDAEPILSTMModel.forward()     # LSTM → (B, T, 2)
  → pred[:, -1, 0]                # scalar pen_norm at the last time step

Usage
-----
    python shap_analysis.py [--model cvae|beta_vae]
                            [--n-bg N] [--n-test N] [--top-k K] [--sample IDX]
                            [--device cuda|cpu] [--csv PATH] [--seed N]

Defaults: model=cvae, n-bg=100, n-test=50, top-k=20, sample=0,
          device=auto, csv=./100_Batches_IndPenSim_V3.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import shap

warnings.filterwarnings("ignore")

# ── Project root on path ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.autoencoder.model import CVAE_Raman, BetaVAE_Raman
from src.models.cdae_pi_lstm import CDAEPILSTMModel
from src.algorithms.base import ScalerBundle
import src.algorithms.cvae_pi_lstm  # noqa: F401 — registers CVAEPILSTMAlgorithm
import src.algorithms.beta_vae_pi_lstm  # noqa: F401 — registers BetaVAEPILSTMAlgorithm
from src.data.dataset import RAMAN_START_COL_IDX

# Human-readable labels for each supported model type
MODEL_LABELS = {
    "cdae":     "CDAE-PI-LSTM",
    "cvae":     "CVAE-PI-LSTM",
    "beta_vae": "\u03b2-VAE-PI-LSTM",
}

MODEL_DEFAULTS = {
    "cdae": {
        "encoder_ckpt":   "/home/gabriel/ReaKt/outputs/seed_sweep/cdae_pi_lstm/p2/encoders/cdae_s42.pt",
        "encoder_scaler": "/home/gabriel/ReaKt/outputs/seed_sweep/cdae_pi_lstm/p2/encoders/cdae_scaler_s42.joblib",
        "lstm_ckpt":     "/home/gabriel/ReaKt/outputs/seed_sweep/cdae_pi_lstm/p2/checkpoints/cdae_pi_lstm_s42.pt",
        "output_dir":    "./shap_cdae",
    },
    "cvae": {
        "encoder_ckpt":   "/home/gabriel/ReaKt/outputs/seed_sweep/cvae_pi_lstm/p2/encoders/cvae_s42.pt",
        "encoder_scaler": "/home/gabriel/ReaKt/outputs/seed_sweep/cvae_pi_lstm/p2/encoders/cvae_scaler_s42.joblib",
        "lstm_ckpt":     "/home/gabriel/ReaKt/outputs/seed_sweep/cvae_pi_lstm/p2/checkpoints/cvae_pi_lstm_s42.pt",
        "output_dir":    "./shap_cvae",
    },
    "beta_vae": {
        "encoder_ckpt":   "/home/gabriel/ReaKt/outputs/seed_sweep/beta_vae_pi_lstm/p2/encoders/beta_vae_s42.pt",
        "encoder_scaler": "/home/gabriel/ReaKt/outputs/seed_sweep/beta_vae_pi_lstm/p2/encoders/beta_vae_scaler_s42.joblib",
        "lstm_ckpt":     "/home/gabriel/ReaKt/outputs/seed_sweep/beta_vae_pi_lstm/p2/checkpoints/beta_vae_pi_lstm_s42.pt",
        "output_dir":    "./shap_beta_vae",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Frozen differentiable preprocessing layers
# ─────────────────────────────────────────────────────────────────────────────

class FrozenSGFilter(nn.Module):
    """
    Savitzky-Golay first-derivative filter as a frozen, differentiable module.

    Exactly reproduces scipy.signal.savgol_filter(x, window_length=15,
    polyorder=2, deriv=1, axis=-1, mode='interp') — scipy's DEFAULT edge mode.

    Interior points (indices half_w .. L-1-half_w) are computed via a valid
    (unpadded) 1-D convolution. The half_w=7 boundary points on each side use
    scipy's 'interp' convention: fit a single polyorder-order polynomial to
    the first/last `window` samples and evaluate its derivative at each edge
    position — this is NOT equivalent to zero-padding and must be handled
    with dedicated coefficients (see savgol_coeffs(..., pos=...)).

    Both the interior kernel and the 2*half_w edge coefficient vectors are
    verified to match scipy to float64 machine precision (~1e-12) before use.
    """

    def __init__(
        self,
        n_wavenumbers: int = 2001,
        window:        int = 15,
        polyorder:     int = 2,
    ) -> None:
        super().__init__()
        self.n_wavenumbers = n_wavenumbers
        self.window        = window
        self.half_w        = window // 2

        from scipy.signal import savgol_coeffs

        # ── Interior kernel ──────────────────────────────────────────────────
        # savgol_coeffs(..., use="conv") [default] returns coefficients meant
        # for TRUE convolution (kernel flipped relative to correlation).
        # torch.nn.functional.conv1d performs CROSS-correlation (no flip), so
        # the kernel must be reversed here to match — otherwise the filter
        # response is a completely different (wrong) kernel, not merely
        # sign-inverted.
        interior_coeffs = savgol_coeffs(window, polyorder, deriv=1).astype(np.float64)
        interior_coeffs = interior_coeffs[::-1].copy()
        self.register_buffer(
            "kernel",
            torch.tensor(interior_coeffs, dtype=torch.float32).view(1, 1, window),
        )

        # ── Edge coefficients (scipy 'interp' mode, exact) ──────────────────
        # Left edge: output[p] = dot(coeffs(pos=p), x[0:window]),  p=0..half_w-1
        left = np.stack([
            savgol_coeffs(window, polyorder, deriv=1, pos=p, use="dot")
            for p in range(self.half_w)
        ]).astype(np.float64)                                  # (half_w, window)

        # Right edge: output[L-half_w+q] = dot(coeffs(pos=window-half_w+q), x[L-window:L]),
        # for q=0..half_w-1, in increasing output-index order (so it concatenates
        # directly after `interior` with no further reordering needed).
        right = np.stack([
            savgol_coeffs(window, polyorder, deriv=1, pos=window - self.half_w + q, use="dot")
            for q in range(self.half_w)
        ]).astype(np.float64)                                   # (half_w, window)

        self.register_buffer("left_coeffs",  torch.tensor(left,  dtype=torch.float32))
        self.register_buffer("right_coeffs", torch.tensor(right, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, n_wavenumbers) float32

        Returns
        -------
        x_sg : (B, n_wavenumbers) float32  — SG first-derivative filtered,
               numerically matching scipy's mode='interp' at all positions.
        """
        hw = self.half_w

        # Interior: valid (unpadded) convolution → (B, L - window + 1)
        interior = nn.functional.conv1d(
            x.unsqueeze(1),      # (B, 1, L)
            self.kernel,         # (1, 1, window)
        ).squeeze(1)             # (B, L - window + 1)

        # Edges: linear combination of the first/last `window` input samples
        left_out  = x[:, : self.window]  @ self.left_coeffs.T   # (B, half_w)
        right_out = x[:, -self.window:] @ self.right_coeffs.T  # (B, half_w)

        return torch.cat([left_out, interior, right_out], dim=1)  # (B, L)


class FrozenStandardScaler(nn.Module):
    """
    Frozen sklearn StandardScaler as a differentiable nn.Module.

    Applies  x_scaled = (x - mean) / std  element-wise using registered buffers.

    Parameters
    ----------
    scaler : sklearn.preprocessing.StandardScaler  — already fitted
    """

    def __init__(self, scaler) -> None:
        super().__init__()
        mean = torch.tensor(scaler.mean_,    dtype=torch.float32)
        std  = torch.tensor(scaler.scale_,   dtype=torch.float32)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, n_features) float32

        Returns
        -------
        x_scaled : (B, n_features) float32
        """
        return (x - self.mean) / self.std


class FrozenMinMaxScaler(nn.Module):
    """
    Frozen sklearn MinMaxScaler as a differentiable nn.Module.

    Applies  x_scaled = (x - data_min) / data_range  element-wise.

    Parameters
    ----------
    scaler : sklearn.preprocessing.MinMaxScaler  — already fitted
    """

    def __init__(self, scaler) -> None:
        super().__init__()
        data_min   = torch.tensor(scaler.data_min_,   dtype=torch.float32)
        data_range = torch.tensor(scaler.data_range_, dtype=torch.float32)
        # Avoid division by zero for constant features
        data_range = data_range.clamp(min=1e-8)
        self.register_buffer("data_min",   data_min)
        self.register_buffer("data_range", data_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, n_features) float32

        Returns
        -------
        x_scaled : (B, n_features) float32
        """
        return (x - self.data_min) / self.data_range


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end differentiable wrapper
# ─────────────────────────────────────────────────────────────────────────────

class EndToEndCVAEPILSTM(nn.Module):
    """
    Full differentiable inference chain from raw Raman sequences to pen_norm.

    Supports both CVAE_Raman and BetaVAE_Raman encoders via the same interface
    (both expose .encode() → (mu, log_var)).

    This module is ONLY used for SHAP attribution — it is never trained.
    All sub-modules have their parameters frozen; only gradient flow is used.

    Parameters
    ----------
    encoder       : CVAE_Raman or BetaVAE_Raman — loaded from the relevant checkpoint
    std_scaler    : FrozenStandardScaler — wraps the encoder's scaler.joblib
    mm_scaler     : FrozenMinMaxScaler   — wraps scalers.feat from the LSTM checkpoint
    lstm_model    : CDAEPILSTMModel      — loaded from the LSTM checkpoint
    sg_filter     : FrozenSGFilter       — SG d=1, w=15, p=2
    n_wavenumbers : int                  — 2001
    seq_len       : int                  — 24
    """

    def __init__(
        self,
        encoder:       nn.Module,
        std_scaler:    FrozenStandardScaler,
        mm_scaler:     FrozenMinMaxScaler,
        lstm_model:    CDAEPILSTMModel,
        sg_filter:     FrozenSGFilter,
        n_wavenumbers: int = 2001,
        seq_len:       int = 24,
    ) -> None:
        super().__init__()
        self.cvae          = encoder   # keep attribute name for backward compat
        self.std_scaler    = std_scaler
        self.mm_scaler     = mm_scaler
        self.lstm_model    = lstm_model
        self.sg_filter     = sg_filter
        self.n_wavenumbers = n_wavenumbers
        self.seq_len       = seq_len

        # Freeze everything — this module is attribution-only
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, 2001) float32  — raw (un-preprocessed) Raman sequences

        Returns
        -------
        pen_norm : (B,) float32  — normalised penicillin prediction at t=T
        """
        B, T, L = x.shape

        # ── 1. Flatten time into batch dimension ────────────────────────────
        x_flat = x.reshape(B * T, L)                      # (B*T, 2001)

        # ── 2. Clamp negatives (instrument artefacts) ───────────────────────
        x_flat = x_flat.clamp(min=0.0)                    # ReLU; ∂/∂x = 1 for x>0

        # ── 3. Savitzky-Golay first-derivative filter ───────────────────────
        x_sg = self.sg_filter(x_flat)                     # (B*T, 2001)

        # ── 4. StandardScaler (fitted on SG-filtered training spectra) ──────
        x_scaled = self.std_scaler(x_sg)                  # (B*T, 2001)

        # ── 5. CVAE encode → mu only (bypass reparameterisation) ────────────
        # CVAE_Raman.encode() accepts (B, L) and auto-unsqueezes the channel dim
        mu, _ = self.cvae.encode(x_scaled)                # (B*T, 64)

        # ── 6. Reshape latents back to sequence ─────────────────────────────
        mu_seq = mu.reshape(B, T, -1)                     # (B, T, 64)

        # ── 7. MinMaxScaler on CVAE latents ─────────────────────────────────
        # Apply per-feature scaling; reshape to (B*T, 64) and back
        mu_flat  = mu_seq.reshape(B * T, -1)              # (B*T, 64)
        mu_mm    = self.mm_scaler(mu_flat)                 # (B*T, 64)
        mu_mm    = mu_mm.reshape(B, T, -1)                 # (B, T, 64)

        # ── 8. PI-LSTM forward ───────────────────────────────────────────────
        out = self.lstm_model(mu_mm)                      # (B, T, 2)

        # ── 9. Scalar prediction: penicillin head at last time step ─────────
        return out[:, -1, 0]                              # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_end_to_end_model(
    cvae_ckpt:  str,
    cvae_scaler: str,
    lstm_ckpt:   str,
    device:      torch.device,
    seq_len:     int = 24,
    n_wavenumbers: int = 2001,
    model_type:  str = "cvae",
) -> Tuple[EndToEndCVAEPILSTM, ScalerBundle]:
    """
    Load all model components and assemble the end-to-end differentiable module.

    Parameters
    ----------
    cvae_ckpt    : path to the encoder checkpoint (cvae_best.pt or beta_vae_best.pt)
    cvae_scaler  : path to the encoder StandardScaler (joblib)
    lstm_ckpt    : path to the LSTM algorithm checkpoint
    device       : torch device
    seq_len      : sliding-window length (default 24)
    n_wavenumbers: Raman channels (default 2001)
    model_type   : "cvae" or "beta_vae"  (selects which encoder class to instantiate)

    Returns
    -------
    model   : EndToEndCVAEPILSTM in eval() mode on device
    scalers : ScalerBundle from the LSTM checkpoint (for pen inverse-scaling)
    """
    label = MODEL_LABELS.get(model_type, model_type)

    # ── Load encoder ──────────────────────────────────────────────────────────
    print(f"  Loading {label} encoder from {Path(cvae_ckpt).name} ...")
    enc_state     = torch.load(cvae_ckpt, map_location="cpu", weights_only=False)
    input_length  = int(enc_state.get("input_length", n_wavenumbers))
    latent_dim    = int(enc_state.get("latent_dim",   64))

    if model_type == "beta_vae":
        encoder = BetaVAE_Raman(input_length=input_length, latent_dim=latent_dim)
    else:
        encoder = CVAE_Raman(input_length=input_length, latent_dim=latent_dim)

    encoder.load_state_dict(enc_state["model_state_dict"])
    encoder.eval()
    print(f"    input_length={input_length}  latent_dim={latent_dim}")

    # ── Load StandardScaler for Raman spectra ─────────────────────────────
    print(f"  Loading StandardScaler from {Path(cvae_scaler).name} ...")
    sklearn_std_scaler = joblib.load(cvae_scaler)
    std_scaler = FrozenStandardScaler(sklearn_std_scaler)

    # ── Load LSTM checkpoint ───────────────────────────────────────────────
    print(f"  Loading {label} LSTM from {Path(lstm_ckpt).name} ...")
    import pickle
    lstm_ckpt_data = torch.load(lstm_ckpt, map_location="cpu", weights_only=False)
    scalers        = ScalerBundle.from_dict(lstm_ckpt_data["scalers"])

    # Reconstruct CDAEPILSTMModel from saved cfg
    from src.config import Config
    cfg = Config.from_dict(lstm_ckpt_data["cfg"])
    m   = cfg.model
    lstm_model = CDAEPILSTMModel(
        input_size      = latent_dim,
        hidden_size     = int(m.hidden_size),
        num_lstm_layers = int(m.num_lstm_layers),
        fc_hidden       = int(m.fc_hidden),
        lstm_dropout    = float(m.lstm_dropout),
        k_prod_init     = float(m.get("k_prod_init", 1.0)),
    )
    lstm_model.load_state_dict(lstm_ckpt_data["model_state"])
    lstm_model.eval()
    k_prod = float(lstm_model.k_prod.item())
    print(f"    k_prod={k_prod:.4f}")

    # ── MinMaxScaler for encoder latents ──────────────────────────────────
    mm_scaler = FrozenMinMaxScaler(scalers.feat)

    # ── SG Filter ─────────────────────────────────────────────────────────
    sg_filter = FrozenSGFilter(n_wavenumbers=n_wavenumbers, window=15, polyorder=2)

    # ── Assemble end-to-end model ──────────────────────────────────────────
    model = EndToEndCVAEPILSTM(
        encoder       = encoder,
        std_scaler    = std_scaler,
        mm_scaler     = mm_scaler,
        lstm_model    = lstm_model,
        sg_filter     = sg_filter,
        n_wavenumbers = n_wavenumbers,
        seq_len       = seq_len,
    ).to(device).eval()

    # cuDNN's LSTM backward pass requires the LSTM to be in training mode at
    # the time of the forward pass (it caches activations needed for backprop).
    # BN layers in the encoder must stay in eval() mode to use running statistics.
    # Since lstm_dropout=0.0, train() mode on the LSTM has no other side effects.
    model.lstm_model.lstm.train()

    print(f"  {label} end-to-end model assembled on {device}")
    print(f"  LSTM set to train() for cuDNN backward compatibility.")
    return model, scalers


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def _detect_batches(df) -> List:
    """Split a flat DataFrame into per-batch DataFrames using time-reset detection."""
    import pandas as pd
    t_vals    = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid
    df["_batch_id"] = batch_ids

    batches = []
    for b_id in range(int(bid + 1)):
        b = df[df["_batch_id"] == b_id].copy().reset_index(drop=True)
        if len(b) >= 50:
            batches.append(b)
    return batches


def _extract_raman_columns(df_cols: List[str], n_raman: int = 2001) -> List[str]:
    """Return the list of 2001 Raman column names starting at RAMAN_START_COL_IDX."""
    avail      = len(df_cols) - RAMAN_START_COL_IDX
    n_use      = min(n_raman, avail)
    raman_cols = df_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + n_use]
    assert len(raman_cols) == n_raman, (
        f"Expected {n_raman} Raman columns, found {len(raman_cols)}. "
        f"Check RAMAN_START_COL_IDX={RAMAN_START_COL_IDX}."
    )
    return raman_cols


def _build_raman_windows(
    batch_df,
    raman_cols: List[str],
    seq_len:    int = 24,
    at_sparse:  bool = False,
) -> np.ndarray:
    """
    Extract sliding Raman windows from a single batch DataFrame.

    Parameters
    ----------
    batch_df   : DataFrame for one batch (rows = time steps)
    raman_cols : list of 2001 Raman column names
    seq_len    : window length (default 24)
    at_sparse  : if True, only return windows ending at sparse measurement rows

    Returns
    -------
    windows : (N, seq_len, 2001) float32  — raw Raman sequences
              or empty array with shape (0, seq_len, 2001) if not enough rows
    """
    X_raw = batch_df[raman_cols].values.astype(np.float32)   # (T_batch, 2001)
    T     = len(X_raw)

    if T < seq_len:
        return np.empty((0, seq_len, len(raman_cols)), dtype=np.float32)

    if at_sparse:
        # Only windows ending at a sparse measurement point
        has_sparse = "_penicillin_sparse" in batch_df.columns
        if has_sparse:
            sparse_mask = ~np.isnan(batch_df["_penicillin_sparse"].values)
        else:
            sparse_mask = np.ones(T, dtype=bool)

        windows = []
        for i in range(seq_len - 1, T):
            if sparse_mask[i]:
                windows.append(X_raw[i - seq_len + 1 : i + 1])
        return np.array(windows, dtype=np.float32) if windows else \
               np.empty((0, seq_len, len(raman_cols)), dtype=np.float32)
    else:
        # All sliding windows
        windows = [X_raw[i : i + seq_len] for i in range(T - seq_len + 1)]
        return np.array(windows, dtype=np.float32)


def prepare_data(
    csv_path:    str,
    n_raman:     int = 2001,
    seq_len:     int = 24,
    train_frac:  float = 0.80,
    seed:        int = 42,
    n_bg:        int = 100,
    n_test:      int = 50,
    rng_seed:    int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load IndPenSim V3 CSV, split into train/test, and build Raman windows.

    Parameters
    ----------
    csv_path   : path to 100_Batches_IndPenSim_V3.csv
    n_raman    : number of Raman channels (default 2001)
    seq_len    : LSTM window length (default 24)
    train_frac : fraction of fault-free batches for training (default 0.80)
    seed       : random seed for train/val/test split (default 42, matches training)
    n_bg       : number of background windows for SHAP (sampled from train)
    n_test     : number of test windows for SHAP (sampled from fault test)
    rng_seed   : random seed for background / test subsampling (default 0)

    Returns
    -------
    X_background : (n_bg,  seq_len, 2001) float32
    X_test       : (n_test, seq_len, 2001) float32
    wavenumber_labels : list of 2001 strings (column names = wavenumber axis)
    """
    import pandas as pd

    print(f"\n  Reading {csv_path} ...")
    # Only load Raman + Time + Fault + sparse target columns to save memory
    all_cols      = pd.read_csv(csv_path, nrows=0).columns.tolist()
    raman_cols    = _extract_raman_columns(all_cols, n_raman)
    needed_extras = ["Time (h)",
                     "Fault reference(Fault_ref:Fault ref)",
                     "Offline Penicillin concentration(P_offline:P(g L^{-1}))"]
    load_cols     = needed_extras + raman_cols

    df = pd.read_csv(csv_path, usecols=load_cols)
    print(f"  Loaded {len(df):,} rows, {len(raman_cols)} Raman channels.")

    # Preserve sparse penicillin before batch processing
    SPARSE_COL = "Offline Penicillin concentration(P_offline:P(g L^{-1}))"
    df["_penicillin_sparse"] = df[SPARSE_COL].copy()   # keep NaN gaps
    df[SPARSE_COL] = df[SPARSE_COL].interpolate(method="linear").ffill().bfill()

    # Detect batches
    batches_all = _detect_batches(df)
    FAULT_COL   = "Fault reference(Fault_ref:Fault ref)"

    clean  = [b for b in batches_all if b[FAULT_COL].max() == 0]
    faulty = [b for b in batches_all if b[FAULT_COL].max() >  0]
    print(f"  Fault-free: {len(clean)}  |  Faulty: {len(faulty)}")

    # Replicate the same random split used at training time (seed=42)
    n_clean = len(clean)
    n_train = int(n_clean * train_frac)
    rng_split = np.random.default_rng(seed)
    perm      = rng_split.permutation(n_clean)
    train_b   = [clean[i] for i in perm[:n_train]]

    # ── Build background windows from training batches (all sliding windows) ──
    print(f"  Building training Raman windows (background pool) ...")
    bg_pool = [
        _build_raman_windows(b, raman_cols, seq_len, at_sparse=False)
        for b in train_b
    ]
    bg_pool = np.concatenate([w for w in bg_pool if len(w) > 0], axis=0)
    print(f"  Background pool: {len(bg_pool):,} windows")

    # ── Build test windows from faulty batches at sparse measurement points ──
    print(f"  Building test Raman windows at sparse measurement points ...")
    test_pool = [
        _build_raman_windows(b, raman_cols, seq_len, at_sparse=True)
        for b in faulty
    ]
    test_pool = np.concatenate([w for w in test_pool if len(w) > 0], axis=0)
    print(f"  Test pool: {len(test_pool):,} windows")

    # ── Subsample ─────────────────────────────────────────────────────────────
    rng = np.random.default_rng(rng_seed)

    n_bg   = min(n_bg,   len(bg_pool))
    n_test = min(n_test, len(test_pool))

    bg_idx   = rng.choice(len(bg_pool),   n_bg,   replace=False)
    test_idx = rng.choice(len(test_pool), n_test, replace=False)

    X_background = bg_pool[bg_idx]
    X_test       = test_pool[test_idx]

    print(f"  X_background : {X_background.shape}  (dtype={X_background.dtype})")
    print(f"  X_test       : {X_test.shape}  (dtype={X_test.dtype})")

    return X_background, X_test, raman_cols


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test (verify forward pass)
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test(
    model:        EndToEndCVAEPILSTM,
    X_background: np.ndarray,
    device:       torch.device,
) -> None:
    """
    Verify the end-to-end forward pass produces finite, non-constant outputs.

    Also checks that gradients flow back to the input tensor.
    """
    print("\n  Smoke-testing forward pass ...")
    x = torch.tensor(X_background[:4], dtype=torch.float32, device=device,
                     requires_grad=True)

    with torch.enable_grad():
        out = model(x)

    assert out.shape == (4,), f"Expected (4,), got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"
    assert out.std() > 1e-6, "Output is constant — likely a preprocessing bug"

    # Check gradient flow
    out.sum().backward()
    assert x.grad is not None, "No gradient at input — chain is broken"
    assert torch.isfinite(x.grad).all(), "Gradient contains NaN or Inf"
    nz_frac = (x.grad.abs() > 1e-10).float().mean().item()
    assert nz_frac > 0.0, "All gradients are zero — chain is broken"

    print(f"    Output range : [{out.min().item():.4f}, {out.max().item():.4f}]")
    print(f"    Grad non-zero: {100 * nz_frac:.1f}% of input elements")
    print("  Forward pass OK.\n")


# ─────────────────────────────────────────────────────────────────────────────
# SHAP computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_values(
    model:        EndToEndCVAEPILSTM,
    X_background: np.ndarray,
    X_test:       np.ndarray,
    device:       torch.device,
    batch_size:   int = 10,
) -> np.ndarray:
    """
    Compute SHAP values using GradientExplainer.

    Parameters
    ----------
    model        : EndToEndCVAEPILSTM in eval() mode
    X_background : (N_bg,  T, 2001) float32  — baseline / reference samples
    X_test       : (N_test, T, 2001) float32  — samples to explain
    device       : torch device
    batch_size   : number of test samples per SHAP call (reduce if OOM)

    Returns
    -------
    shap_values : (N_test, T, 2001) float32
        Positive value → input feature pushes prediction UP (more penicillin)
        Negative value → input feature pushes prediction DOWN
    """
    # GradientExplainer requires output shape (B, n_outputs).
    # Our model returns (B,); wrap it to return (B, 1).
    #
    # IMPORTANT: GradientExplainer.__init__ calls model.eval() internally,
    # which would reset the LSTM to eval mode and break cuDNN backward.
    # We override train()/eval() on this wrapper so the LSTM is always kept
    # in training mode (required by cuDNN for backpropagation through RNN).
    # Since lstm_dropout=0.0 this has no effect on output statistics.
    class _ShapWrapper(nn.Module):
        def __init__(self, m: EndToEndCVAEPILSTM):
            super().__init__()
            self.m     = m
            self._lstm = m.lstm_model.lstm   # direct ref for mode override

        def train(self, mode: bool = True):
            super().train(mode)
            self._lstm.train(True)           # always keep LSTM in train mode
            return self

        def eval(self):                      # redirect so self.train() override fires
            return self.train(mode=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.m(x).unsqueeze(-1)   # (B,) → (B, 1)

    shap_model = _ShapWrapper(model)

    bg_tensor = torch.tensor(X_background, dtype=torch.float32).to(device)

    explainer = shap.GradientExplainer(shap_model, bg_tensor)
    print(f"  GradientExplainer initialised  (background: {X_background.shape[0]} samples)")

    all_shap = []
    n_test = len(X_test)
    for start in range(0, n_test, batch_size):
        end   = min(start + batch_size, n_test)
        chunk = torch.tensor(
            X_test[start:end], dtype=torch.float32
        ).to(device)
        sv = explainer.shap_values(chunk)   # list of (batch, T, 2001) or ndarray
        # GradientExplainer returns a list when output is scalar (single-output model)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.array(sv, dtype=np.float32)
        # Our wrapper returns (B, 1); GradientExplainer produces an extra trailing dim
        if sv.ndim == 4 and sv.shape[-1] == 1:
            sv = sv.squeeze(-1)   # (batch, T, 2001, 1) → (batch, T, 2001)
        all_shap.append(sv)
        print(f"    SHAP batch {start}:{end}  shape={all_shap[-1].shape}")

    shap_values = np.concatenate(all_shap, axis=0)
    print(f"  SHAP values shape: {shap_values.shape}")
    return shap_values


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_global_importance(
    shap_values:       np.ndarray,
    wavenumber_labels: List[str],
    output_dir:        Path,
    top_k:             int = 50,
    model_label:       str = "CVAE-PI-LSTM",
) -> None:
    """
    Bar chart: mean |SHAP| aggregated over samples (N) and time steps (T).

    Shows the top-k most important Raman wavenumbers globally.

    Parameters
    ----------
    shap_values        : (N, T, 2001) float32
    wavenumber_labels  : list of 2001 wavenumber strings
    output_dir         : directory to save the figure
    top_k              : number of top wavenumbers to show (default 50)
    model_label        : human-readable model name for plot title
    """
    # Mean absolute SHAP over N and T → (2001,)
    mean_abs = np.abs(shap_values).mean(axis=(0, 1))

    # Sort descending
    order    = np.argsort(mean_abs)[::-1]
    top_idx  = order[:top_k]
    top_vals = mean_abs[top_idx]
    top_lbls = [wavenumber_labels[int(i)] for i in top_idx]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(range(top_k), top_vals[::-1], color="#2980b9", edgecolor="none")
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_lbls[::-1], fontsize=7)
    ax.set_xlabel("Mean |SHAP value|  (pen_norm units)", fontsize=11)
    ax.set_title(
        f"Global Raman Feature Importance  ({model_label})\n"
        f"Top-{top_k} wavenumbers  |  averaged over {shap_values.shape[0]} "
        f"samples \u00d7 T={shap_values.shape[1]} time steps",
        fontsize=12,
    )
    ax.invert_yaxis()
    plt.tight_layout()

    save_path = output_dir / "global_importance.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved global importance plot → {save_path}")

    # Also save a CSV for the paper
    import csv as csv_mod
    csv_path = output_dir / "global_importance.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow(["rank", "wavenumber_label", "wavenumber_idx", "mean_abs_shap"])
        for rank, (idx, val, lbl) in enumerate(
            zip(top_idx, top_vals, top_lbls), start=1
        ):
            w.writerow([rank, lbl, int(idx), float(val)])
    print(f"  Saved global importance CSV  → {csv_path}")


def plot_temporal_heatmap(
    shap_values:       np.ndarray,
    wavenumber_labels: List[str],
    output_dir:        Path,
    sample_idx:        int = 0,
    top_k:             int = 20,
) -> None:
    """
    Temporal heatmap: SHAP(T, top-K wavenumbers) for a single test sample.

    Rows = time steps (t-23 to t), Columns = top-K globally important wavenumbers.

    Parameters
    ----------
    shap_values        : (N, T, 2001) float32
    wavenumber_labels  : list of 2001 wavenumber strings
    output_dir         : directory to save the figure
    sample_idx         : which test sample to visualise (default 0)
    top_k              : number of top wavenumbers in the heatmap (default 20)
    """
    import matplotlib.colors as mcolors

    # Identify globally top-K wavenumbers
    global_importance = np.abs(shap_values).mean(axis=(0, 1))   # (2001,)
    top_idx           = np.argsort(global_importance)[::-1][:top_k]

    # Extract the single sample SHAP slice: (T, top_k)
    sv_sample = shap_values[sample_idx]                          # (T, 2001)
    sv_heatmap = sv_sample[:, top_idx]                           # (T, top_k)

    T = sv_heatmap.shape[0]
    time_labels = [f"t−{T - 1 - i}" if i < T - 1 else "t" for i in range(T)]
    wave_labels = [wavenumber_labels[int(i)] for i in top_idx]

    # Symmetric colour scale centred on zero
    vmax = float(np.abs(sv_heatmap).max()) or 1e-6
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(8, top_k * 0.5), max(6, T * 0.3)))
    im = ax.imshow(sv_heatmap, aspect="auto", cmap="RdBu_r", norm=norm)

    ax.set_xticks(range(top_k))
    ax.set_xticklabels(wave_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(T))
    ax.set_yticklabels(time_labels, fontsize=8)
    ax.set_xlabel("Raman wavenumber (top-K global importance)", fontsize=10)
    ax.set_ylabel("Time step in sequence window", fontsize=10)
    ax.set_title(
        f"Temporal SHAP Heatmap  (sample={sample_idx})\n"
        f"Red = increases pen prediction  |  Blue = decreases pen prediction",
        fontsize=11,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("SHAP value (pen_norm)", fontsize=9)
    plt.tight_layout()

    save_path = output_dir / "temporal_heatmap.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved temporal heatmap        → {save_path}")


def plot_shap_spectrum_overlay(
    shap_values:       np.ndarray,
    X_test:            np.ndarray,
    wavenumber_labels: List[str],
    output_dir:        Path,
    sample_idx:        int = 0,
    model_label:       str = "CVAE-PI-LSTM",
) -> None:
    """
    Overlay the mean Raman spectrum with the global |SHAP| profile.

    Useful for interpreting which spectral bands drive the model.

    Parameters
    ----------
    shap_values        : (N, T, 2001) float32
    X_test             : (N, T, 2001) float32 raw Raman sequences
    wavenumber_labels  : list of 2001 wavenumber strings
    output_dir         : directory to save the figure
    sample_idx         : sample to overlay SHAP from (default: mean over all)
    model_label        : human-readable model name for plot title
    """
    # Try to parse numeric wavenumbers from column names
    try:
        wn = np.array([float(lbl) for lbl in wavenumber_labels])
    except (ValueError, TypeError):
        wn = np.arange(len(wavenumber_labels), dtype=float)

    # Mean absolute SHAP over N and T
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 1))         # (2001,)
    # Mean raw Raman spectrum (last time step of each window)
    mean_spectrum = X_test[:, -1, :].mean(axis=0)                 # (2001,)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    color1, color2 = "#2c3e50", "#e74c3c"

    ax1.plot(wn, mean_spectrum / max(mean_spectrum.max(), 1e-6),
             color=color1, linewidth=0.8, alpha=0.7, label="Mean Raman spectrum (norm.)")
    ax1.set_xlabel("Raman wavenumber (cm\u207b\u00b9)" if wn[-1] > 100 else "Wavenumber index",
                   fontsize=11)
    ax1.set_ylabel("Normalised intensity", color=color1, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.fill_between(wn, mean_abs_shap, alpha=0.4, color=color2,
                     label="Mean |SHAP|")
    ax2.set_ylabel("Mean |SHAP value|", color=color2, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    ax1.set_title(
        f"Raman Spectrum vs. Global SHAP Importance  ({model_label})",
        fontsize=12,
    )
    plt.tight_layout()

    save_path = output_dir / "spectrum_shap_overlay.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved spectrum/SHAP overlay   → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SHAP explainability for CVAE-PI-LSTM / β-VAE-PI-LSTM Raman soft-sensor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="cvae",
                   choices=["cvae", "beta_vae", "cdae"],
                   help="Which encoder model to explain (cvae or beta_vae)")
    p.add_argument("--csv",       default="./100_Batches_IndPenSim_V3.csv",
                   help="Path to IndPenSim V3 CSV")
    p.add_argument("--cvae-ckpt", default=None,
                   help="Encoder checkpoint (defaults to model-specific path)")
    p.add_argument("--cvae-scaler", default=None,
                   help="StandardScaler for SG-filtered Raman (defaults to model-specific path)")
    p.add_argument("--lstm-ckpt", default=None,
                   help="LSTM algorithm checkpoint (defaults to model-specific path)")
    p.add_argument("--n-bg",    type=int, default=100,
                   help="Number of background samples for SHAP")
    p.add_argument("--n-test",  type=int, default=50,
                   help="Number of test samples to explain")
    p.add_argument("--top-k",   type=int, default=20,
                   help="Top-K wavenumbers in temporal heatmap")
    p.add_argument("--sample",  type=int, default=0,
                   help="Test sample index for temporal heatmap")
    p.add_argument("--device",  default="auto",
                   choices=["auto", "cuda", "cpu"],
                   help="Compute device")
    p.add_argument("--output-dir", default=None,
                   help="Directory to save output figures (defaults to model-specific path)")
    p.add_argument("--seed",    type=int, default=42,
                   help="Train/val/test split seed (must match training)")
    p.add_argument("--shap-batch", type=int, default=10,
                   help="Test samples per SHAP call (reduce if OOM)")
    p.add_argument("--smoke-only", action="store_true",
                   help="Only run the forward-pass smoke test, skip SHAP")
    p.add_argument("--no-heatmap", action="store_true",
                   help="Skip generating temporal_heatmap.png (single-sample "
                        "time x wavenumber heatmap)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Resolve model-specific defaults ──────────────────────────────────────
    defaults    = MODEL_DEFAULTS[args.model]
    model_label = MODEL_LABELS[args.model]
    enc_ckpt    = args.cvae_ckpt   or defaults["encoder_ckpt"]
    enc_scaler  = args.cvae_scaler or defaults["encoder_scaler"]
    lstm_ckpt   = args.lstm_ckpt   or defaults["lstm_ckpt"]
    out_dir_str = args.output_dir  or defaults["output_dir"]

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n  Model  : {model_label}")
    print(f"  Device : {device}")

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print("\n── Loading model ────────────────────────────────────────────────")
    model, scalers = load_end_to_end_model(
        cvae_ckpt   = enc_ckpt,
        cvae_scaler = enc_scaler,
        lstm_ckpt   = lstm_ckpt,
        device      = device,
        model_type  = args.model,
    )

    # ── Prepare data ──────────────────────────────────────────────────────────
    print("\n── Preparing data ───────────────────────────────────────────────")
    X_background, X_test, wavenumber_labels = prepare_data(
        csv_path   = args.csv,
        n_bg       = args.n_bg,
        n_test     = args.n_test,
        seed       = args.seed,
    )

    # ── Smoke test ────────────────────────────────────────────────────────────
    print("\n── Smoke test ───────────────────────────────────────────────────")
    smoke_test(model, X_background, device)

    if args.smoke_only:
        print("  --smoke-only flag set, exiting.")
        return

    # ── SHAP analysis ─────────────────────────────────────────────────────────
    print("\n── Computing SHAP values ────────────────────────────────────────")
    shap_values = compute_shap_values(
        model        = model,
        X_background = X_background,
        X_test       = X_test,
        device       = device,
        batch_size   = args.shap_batch,
    )

    # Save raw SHAP values for later analysis
    shap_npy = out_dir / "shap_values.npy"
    np.save(shap_npy, shap_values)
    print(f"  Saved raw SHAP values         → {shap_npy}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n── Generating plots ─────────────────────────────────────────────")

    plot_global_importance(
        shap_values        = shap_values,
        wavenumber_labels  = wavenumber_labels,
        output_dir         = out_dir,
        top_k              = 50,
        model_label        = model_label,
    )

    sample_idx = min(args.sample, len(X_test) - 1)
    if not args.no_heatmap:
        plot_temporal_heatmap(
            shap_values        = shap_values,
            wavenumber_labels  = wavenumber_labels,
            output_dir         = out_dir,
            sample_idx         = sample_idx,
            top_k              = args.top_k,
        )

    plot_shap_spectrum_overlay(
        shap_values        = shap_values,
        X_test             = X_test,
        wavenumber_labels  = wavenumber_labels,
        output_dir         = out_dir,
        model_label        = model_label,
    )

    print(f"\n  All outputs saved to {out_dir}/")
    print("  Done.\n")


if __name__ == "__main__":
    main()
