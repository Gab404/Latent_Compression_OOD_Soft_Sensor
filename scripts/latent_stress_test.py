#!/usr/bin/env python
"""
scripts/latent_stress_test.py
=============================
Automated topological stress-tests of the CDAE / CVAE / β-VAE latent spaces.

The script encodes "difficult" point pairs, linearly interpolates in 64-D
latent space and decodes back to spectrum space.  Smooth interpolation
requires a well-organised, connected manifold; deterministic or
under-regularised encoders will exhibit visible artefacts (discontinuities,
ripples, unphysical peaks) along the interpolation path.

Three scenario types are automatically extracted from the IndPenSim CSV:

  Scenario 1 — Cross-Regime
      A : Batch 15  (Regime 1, recipe-driven)  at t ≈ 100 h
      B : Batch 95  (Faulty, Regime 4)         at t ≈ 100 h
      Probe: traversal across the health/fault boundary with no temporal
      transition.  The straight line in latent space cuts through a
      "void" that was never covered by training data.

  Scenario 2 — Failure vs. Success
      A : Final spectrum of the Regime 1 batch with the LOWEST final
          penicillin titer (poorest producer)
      B : Final spectrum of the Regime 1 batch with the HIGHEST final
          penicillin titer (best producer)
      Probe: the decoder must vary only the penicillin-related peaks while
      leaving the background unchanged.  Tests disentanglement quality.

  Scenario 3 — Turbulence Zone (exponential growth phase)
      A : Batch 10  at t ≈ 40 h  (onset of exponential growth)
      B : Batch 10  at t ≈ 80 h  (peak biomass, maximal broth complexity)
      Probe: the most violently changing segment of the batch profile.
      Biomass explodes, dissolved oxygen crashes, sugar depletes rapidly.

For each scenario the script produces two PDF pages:
  • Page (odd)  : Full wavenumber range waterfall  [~400 – 2400 cm⁻¹]
  • Page (even) : Zoomed window  [1500 – 1800 cm⁻¹]  where the
                  penicillin C=O and amide bands sit and artefacts
                  are most diagnostically informative.

Output
------
    outputs/latent_stress_test/stress_test.pdf   (6 pages total)
    outputs/latent_stress_test/stress_test_<scenario>.png  (full-range PNGs)

Run from project root
---------------------
    CUDA_VISIBLE_DEVICES=0  python scripts/latent_stress_test.py
    python scripts/latent_stress_test.py --n-steps 20 --output ./my_stress.pdf
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter

from src.autoencoder.model import CDAE_Raman, CVAE_Raman, BetaVAE_Raman
from src.data.dataset import RAMAN_START_COL_IDX


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CSV_PATH        = "./100_Batches_IndPenSim_V3.csv"
CDAE_CKPT       = "./checkpoints/cdae_best.pt"
CDAE_SCALER     = "./checkpoints/cdae_scaler.joblib"
CVAE_CKPT       = "./checkpoints/cvae_best.pt"
CVAE_SCALER     = "./checkpoints/cvae_scaler.joblib"
BETA_VAE_CKPT   = "./checkpoints/beta_vae_best.pt"
BETA_VAE_SCALER = "./checkpoints/beta_vae_scaler.joblib"
OUTPUT_PDF      = "./outputs/latent_stress_test/stress_test.pdf"

N_RAMAN   = 2001
SG_WINDOW = 15
SG_POLY   = 2
SG_DERIV  = 1
N_STEPS   = 15          # interpolation steps including both endpoints

ZOOM_LO   = 1500.0      # cm⁻¹ — lower bound of the zoomed region
ZOOM_HI   = 1800.0      # cm⁻¹ — upper bound of the zoomed region

# IndPenSim V3 batch ID ranges (0-indexed: batch 1 ≡ id 0)
REGIME_1_IDS = set(range(0,  30))   # batches  1-30  recipe-driven
REGIME_2_IDS = set(range(30, 60))   # batches 31-60  operator-controlled
REGIME_3_IDS = set(range(60, 90))   # batches 61-90  APC closed-loop
FAULTY_IDS   = set(range(90, 100))  # batches 91-100 process faults

# Scenario 1 — Cross-Regime
SC1_BATCH_A = 14      # 0-indexed → batch 15, Regime 1
SC1_BATCH_B = 94      # 0-indexed → batch 95, Faulty (Regime 4)
SC1_TIME    = 100.0   # target fermentation time (h) for both points

# Scenario 3 — Turbulence Zone
SC3_BATCH  = 9        # 0-indexed → batch 10, Regime 1
SC3_TIME_A = 40.0     # h  onset of exponential growth
SC3_TIME_B = 80.0     # h  peak biomass / maximal broth complexity

TIME_COL       = "Time (h)"
PENICILLIN_COL = "Penicillin concentration(P:g/L)"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> tuple[dict, np.ndarray, list]:
    """
    Load the full IndPenSim CSV, detect batch boundaries from time-resets,
    and extract the wavenumber axis.

    Returns
    -------
    batches     : dict[int → pd.DataFrame]  one entry per batch (0-indexed id)
    wavenumbers : (N_RAMAN,) float64  ascending wavenumber axis (cm⁻¹)
    raman_cols  : list[str]  Raman column names in their original CSV order
                  (descending wavenumber, i.e. the order the scalers expect)
    """
    print(f"[1] Loading {csv_path} …")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"    {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Detect batches via time resets ────────────────────────────────────────
    t_vals    = df[TIME_COL].values.astype(float)
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid
    df["_batch_id"] = batch_ids
    n_batches = int(bid + 1)
    print(f"    Detected {n_batches} batches")

    # ── Raman column names & wavenumber axis ──────────────────────────────────
    all_cols   = df.columns.tolist()
    raman_cols = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + N_RAMAN]

    wn_raw = []
    for c in raman_cols:
        try:
            wn_raw.append(float(c))
        except ValueError:
            wn_raw.append(float("nan"))
    wavenumbers = np.array(wn_raw, dtype=np.float64)

    # Reverse to ascending wavenumber order for display
    if wavenumbers[0] > wavenumbers[-1]:
        wavenumbers = wavenumbers[::-1].copy()

    # ── Per-batch DataFrames ──────────────────────────────────────────────────
    batches: dict[int, pd.DataFrame] = {}
    for b_id in range(n_batches):
        b_df = df[df["_batch_id"] == b_id].reset_index(drop=True)
        batches[b_id] = b_df

    return batches, wavenumbers, raman_cols


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Filtering helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_closest_row(batch_df: pd.DataFrame, target_time: float) -> int:
    """
    Return the integer row index (within batch_df) whose time value is
    closest to ``target_time``.  Works with any sampling density.
    """
    t = batch_df[TIME_COL].values.astype(float)
    return int(np.argmin(np.abs(t - target_time)))


def extract_spectrum(
    batch_df:   pd.DataFrame,
    row_idx:    int,
    raman_cols: list,
) -> np.ndarray:
    """
    Extract and sanitise a single Raman spectrum.

    Returns
    -------
    spec : (N_RAMAN,) float32  in CSV column order (descending wavenumber).
           Negative values and NaN are clamped to 0.
    """
    spec = batch_df[raman_cols].iloc[row_idx].values.astype(np.float32)
    spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
    spec = np.clip(spec, 0.0, None)
    return spec


def last_valid_raman_row(batch_df: pd.DataFrame, raman_cols: list) -> int:
    """
    Return the index of the latest (highest-time) row that has a non-zero
    Raman spectrum.  Warm-up rows at the start of a batch have all-zero Raman,
    but the end of the batch should always have valid signal.
    """
    X      = batch_df[raman_cols].values.astype(np.float32)
    energy = np.abs(X).sum(axis=1)
    valid  = np.where(energy > 0)[0]
    if len(valid) == 0:
        return len(batch_df) - 1    # fallback: last row

    t_vals = batch_df[TIME_COL].values.astype(float)[valid]
    return int(valid[np.argmax(t_vals)])


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Preprocessing  (must mirror training exactly)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(spec: np.ndarray, scaler) -> np.ndarray:
    """
    Apply SG first-derivative + StandardScaler to a single spectrum.

    Input / output : (N_RAMAN,) float32  in CSV order (descending wavenumber).
    The scaler was fitted on the training set in this same column order.
    """
    x_sg = savgol_filter(
        spec.astype(np.float64).reshape(1, -1),
        window_length = SG_WINDOW,
        polyorder     = SG_POLY,
        deriv         = SG_DERIV,
        axis          = 1,
    ).astype(np.float32)
    return scaler.transform(x_sg).astype(np.float32).squeeze(0)   # (N_RAMAN,)


def to_tensor(x: np.ndarray) -> torch.Tensor:
    """(N_RAMAN,) → (1, N_RAMAN) float32 tensor on DEVICE."""
    return torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_scalers() -> tuple:
    """Return (cdae_scaler, cvae_scaler, beta_vae_scaler)."""
    print("[2] Loading scalers …")
    return (
        joblib.load(CDAE_SCALER),
        joblib.load(CVAE_SCALER),
        joblib.load(BETA_VAE_SCALER),
    )


def load_models() -> tuple:
    """Return frozen (cdae, cvae, beta_vae) on DEVICE."""
    print("[3] Loading models …")

    def _load(ckpt_path, ModelCls, **extra):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m = ModelCls(
            input_length = int(state.get("input_length", N_RAMAN)),
            latent_dim   = int(state.get("latent_dim",   64)),
            **extra,
        ).to(DEVICE)
        m.load_state_dict(state["model_state_dict"])
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    cdae     = _load(CDAE_CKPT,     CDAE_Raman,    noise_std=0.0)
    cvae     = _load(CVAE_CKPT,     CVAE_Raman)
    beta_vae = _load(BETA_VAE_CKPT, BetaVAE_Raman)
    print("    All three models loaded and frozen.\n")
    return cdae, cvae, beta_vae


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Encode → interpolate → decode pipeline
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _encode(model, is_vae: bool, spec: np.ndarray, scaler) -> torch.Tensor:
    """Encode one spectrum; return latent (1, D)."""
    x = preprocess(spec, scaler)
    t = to_tensor(x)
    if is_vae:
        mu, _ = model.encode(t)
        return mu
    return model.encode(t)


@torch.no_grad()
def _interpolate_decode(
    model,
    scaler,
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    n_steps: int,
) -> np.ndarray:
    """
    Linear interpolation z_α = (1-α)·z_a + α·z_b, decode, invert scaler.

    Returns
    -------
    spectra : (n_steps, N_RAMAN) float32  in descending wavenumber order
              (same as the scaler's fitted order).
    """
    alphas  = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    spectra = []
    for alpha in alphas:
        z    = (1.0 - float(alpha)) * z_a + float(alpha) * z_b
        recon = model.decode(z).squeeze(0).cpu().numpy()
        recon_sg = scaler.inverse_transform(
            recon.reshape(1, -1)
        ).squeeze(0).astype(np.float32)
        spectra.append(recon_sg)
    return np.stack(spectra, axis=0)


def run_interpolation(
    spec_a:  np.ndarray,    # (N_RAMAN,) float32, descending wavenumber
    spec_b:  np.ndarray,
    cdae,    cvae,    beta_vae,
    cdae_sc, cvae_sc, bvae_sc,
    n_steps: int = N_STEPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full encode-interpolate-decode pipeline for one (A, B) pair.

    Returns
    -------
    alphas           : (n_steps,)
    spectra_cdae     : (n_steps, N_RAMAN) float32  ascending wavenumber
    spectra_cvae     : (n_steps, N_RAMAN) float32  ascending wavenumber
    spectra_beta_vae : (n_steps, N_RAMAN) float32  ascending wavenumber
    """
    z_a_cdae = _encode(cdae,     False, spec_a, cdae_sc)
    z_b_cdae = _encode(cdae,     False, spec_b, cdae_sc)
    z_a_cvae = _encode(cvae,     True,  spec_a, cvae_sc)
    z_b_cvae = _encode(cvae,     True,  spec_b, cvae_sc)
    z_a_bvae = _encode(beta_vae, True,  spec_a, bvae_sc)
    z_b_bvae = _encode(beta_vae, True,  spec_b, bvae_sc)

    dec_cdae = _interpolate_decode(cdae,     cdae_sc, z_a_cdae, z_b_cdae, n_steps)
    dec_cvae = _interpolate_decode(cvae,     cvae_sc, z_a_cvae, z_b_cvae, n_steps)
    dec_bvae = _interpolate_decode(beta_vae, bvae_sc, z_a_bvae, z_b_bvae, n_steps)

    alphas = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)

    # Reverse from descending (CSV / scaler order) to ascending (display)
    return (
        alphas,
        dec_cdae[:, ::-1].copy(),
        dec_cvae[:, ::-1].copy(),
        dec_bvae[:, ::-1].copy(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Scenario extraction functions
# ─────────────────────────────────────────────────────────────────────────────

def scenario_cross_regime(
    batches:    dict,
    raman_cols: list,
) -> tuple[np.ndarray, np.ndarray, str, str, str]:
    """
    Scenario 1 — Cross-Regime Interpolation.

    Straight line in latent space from a healthy recipe batch (Regime 1)
    to a faulty batch (Regime 4) at the same fermentation time.
    The interpolation path cuts through a region of latent space that was
    never covered during training → stress-tests extrapolation capability.

    Returns
    -------
    spec_a, spec_b, label_a, label_b, scenario_title
    """
    for bid, regime_label in [(SC1_BATCH_A, f"Regime 1, batch {SC1_BATCH_A+1}"),
                               (SC1_BATCH_B, f"Faulty,   batch {SC1_BATCH_B+1}")]:
        if bid not in batches:
            raise ValueError(
                f"Batch id {bid} ({regime_label}) not found in {CSV_PATH}."
            )

    df_a = batches[SC1_BATCH_A]
    df_b = batches[SC1_BATCH_B]

    idx_a = find_closest_row(df_a, SC1_TIME)
    idx_b = find_closest_row(df_b, SC1_TIME)
    t_a   = float(df_a[TIME_COL].iloc[idx_a])
    t_b   = float(df_b[TIME_COL].iloc[idx_b])

    spec_a = extract_spectrum(df_a, idx_a, raman_cols)
    spec_b = extract_spectrum(df_b, idx_b, raman_cols)

    label_a = (
        f"A — Batch {SC1_BATCH_A+1} (Regime 1, recipe)\n"
        f"t = {t_a:.1f} h   [healthy fermentation]"
    )
    label_b = (
        f"B — Batch {SC1_BATCH_B+1} (Faulty, Regime 4)\n"
        f"t = {t_b:.1f} h   [process fault]"
    )
    title = (
        "Scenario 1  —  Cross-Regime Interpolation\n"
        f"Batch {SC1_BATCH_A+1} Regime 1 (t ≈ {SC1_TIME:.0f} h)  "
        f"→  Batch {SC1_BATCH_B+1} Faulty (t ≈ {SC1_TIME:.0f} h)"
    )
    return spec_a, spec_b, label_a, label_b, title


def scenario_fail_vs_success(
    batches:    dict,
    raman_cols: list,
) -> tuple[np.ndarray, np.ndarray, str, str, str]:
    """
    Scenario 2 — Failure vs. Success (Penicillin Disentanglement).

    Compares the final spectrum of the worst producer vs the best producer
    among Regime 1 batches.  Both spectra share the same biological context
    (recipe-driven, similar process trajectory) but differ maximally in their
    final penicillin titer.  A well-disentangled decoder should vary only the
    penicillin-related peaks along this path.

    Selection criterion : penicillin concentration (online sensor) at the
    last time point that has valid Raman signal within the batch.

    Returns
    -------
    spec_a, spec_b, label_a, label_b, scenario_title
    """
    final_pen: dict[int, tuple[float, int]] = {}   # bid → (pen_g_L, row_idx)

    for bid in sorted(REGIME_1_IDS):
        df = batches.get(bid)
        if df is None:
            continue
        if PENICILLIN_COL not in df.columns:
            continue

        row_idx = last_valid_raman_row(df, raman_cols)
        pen     = float(df[PENICILLIN_COL].iloc[row_idx])

        if np.isnan(pen) or pen <= 0.0:
            # Fallback: maximum penicillin over the whole batch
            pen = float(df[PENICILLIN_COL].max())
        if np.isnan(pen) or pen <= 0.0:
            continue

        final_pen[bid] = (pen, row_idx)

    if len(final_pen) < 2:
        raise ValueError(
            f"Could not find ≥ 2 Regime 1 batches with valid penicillin "
            f"data (found {len(final_pen)}).  Check that '{PENICILLIN_COL}' "
            f"exists in the CSV."
        )

    batch_min = min(final_pen, key=lambda k: final_pen[k][0])
    batch_max = max(final_pen, key=lambda k: final_pen[k][0])

    pen_min, idx_min = final_pen[batch_min]
    pen_max, idx_max = final_pen[batch_max]

    df_min = batches[batch_min]
    df_max = batches[batch_max]
    t_min  = float(df_min[TIME_COL].iloc[idx_min])
    t_max  = float(df_max[TIME_COL].iloc[idx_max])

    # Log the full distribution for transparency
    pen_vals = sorted(final_pen.values(), key=lambda x: x[0])
    print(f"    Regime 1 final penicillin range: "
          f"{pen_vals[0][0]:.2f} g/L  →  {pen_vals[-1][0]:.2f} g/L  "
          f"(over {len(final_pen)} batches)")
    print(f"    Min producer : batch {batch_min+1}  ({pen_min:.3f} g/L,  t={t_min:.1f} h)")
    print(f"    Max producer : batch {batch_max+1}  ({pen_max:.3f} g/L,  t={t_max:.1f} h)")

    spec_a = extract_spectrum(df_min, idx_min, raman_cols)
    spec_b = extract_spectrum(df_max, idx_max, raman_cols)

    label_a = (
        f"A — Batch {batch_min+1} (Regime 1, worst producer)\n"
        f"Final P = {pen_min:.2f} g/L   t = {t_min:.1f} h"
    )
    label_b = (
        f"B — Batch {batch_max+1} (Regime 1, best producer)\n"
        f"Final P = {pen_max:.2f} g/L   t = {t_max:.1f} h"
    )
    title = (
        "Scenario 2  —  Failure vs. Success  (Penicillin Disentanglement)\n"
        f"Batch {batch_min+1}: {pen_min:.2f} g/L  "
        f"→  Batch {batch_max+1}: {pen_max:.2f} g/L   "
        f"(Regime 1 final titer)"
    )
    return spec_a, spec_b, label_a, label_b, title


def scenario_turbulence_zone(
    batches:    dict,
    raman_cols: list,
) -> tuple[np.ndarray, np.ndarray, str, str, str]:
    """
    Scenario 3 — Turbulence Zone (exponential growth phase).

    Within a single batch, interpolates from the onset of exponential growth
    (t ≈ 40 h) to the peak of biomass accumulation (t ≈ 80 h).
    During this window the broth composition changes most rapidly; biomass
    explodes, dissolved oxygen crashes, sugar is nearly depleted.

    Returns
    -------
    spec_a, spec_b, label_a, label_b, scenario_title
    """
    if SC3_BATCH not in batches:
        raise ValueError(
            f"Batch id {SC3_BATCH} (batch {SC3_BATCH+1}) not found in {CSV_PATH}."
        )

    df    = batches[SC3_BATCH]
    idx_a = find_closest_row(df, SC3_TIME_A)
    idx_b = find_closest_row(df, SC3_TIME_B)
    t_a   = float(df[TIME_COL].iloc[idx_a])
    t_b   = float(df[TIME_COL].iloc[idx_b])

    spec_a = extract_spectrum(df, idx_a, raman_cols)
    spec_b = extract_spectrum(df, idx_b, raman_cols)

    label_a = (
        f"A — Batch {SC3_BATCH+1} (Regime 1)\n"
        f"t = {t_a:.1f} h   [onset of exponential growth]"
    )
    label_b = (
        f"B — Batch {SC3_BATCH+1} (Regime 1)\n"
        f"t = {t_b:.1f} h   [peak biomass / broth complexity]"
    )
    title = (
        "Scenario 3  —  Turbulence Zone  (Exponential Growth Phase)\n"
        f"Batch {SC3_BATCH+1}: t ≈ {SC3_TIME_A:.0f} h  →  t ≈ {SC3_TIME_B:.0f} h"
    )
    return spec_a, spec_b, label_a, label_b, title


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Publication-quality waterfall figure
# ─────────────────────────────────────────────────────────────────────────────

_PANEL_TITLES = [
    "CDAE  —  Convolutional Denoising AE",
    "CVAE  —  Variational AE  (β = 1)",
    r"$\beta$-VAE  —  Disentangled VAE  ($\beta$ = 0.005)",
]

_MPL_PARAMS = {
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
}


def _minmax_norm(S: np.ndarray) -> np.ndarray:
    """Per-row min-max normalisation. S shape: (N, L)."""
    lo  = S.min(axis=1, keepdims=True)
    hi  = S.max(axis=1, keepdims=True)
    rng = np.where((hi - lo) > 0, hi - lo, 1.0)
    return (S - lo) / rng


def make_waterfall_figure(
    wavenumbers:      np.ndarray,   # (L,) ascending cm⁻¹
    alphas:           np.ndarray,   # (N,) ∈ [0, 1]
    spectra_cdae:     np.ndarray,   # (N, L) ascending
    spectra_cvae:     np.ndarray,
    spectra_beta_vae: np.ndarray,
    label_a:          str,          # multi-line description for α=0
    label_b:          str,          # multi-line description for α=1
    page_title:       str,          # figure suptitle
    x_lo:             float,        # wavenumber window lower bound
    x_hi:             float,        # wavenumber window upper bound
) -> plt.Figure:
    """
    Three-panel waterfall figure for one (scenario, wavenumber window).

    Parameters
    ----------
    x_lo, x_hi : restrict both the x-axis range AND the per-spectrum
                 normalisation to this window, so the zoomed view always
                 makes full use of the y-range (artefacts are magnified).
    """
    mpl.rcParams.update(_MPL_PARAMS)

    # ── Restrict to the requested wavenumber window ───────────────────────────
    mask = (wavenumbers >= x_lo - 0.5) & (wavenumbers <= x_hi + 0.5)
    if mask.sum() < 2:
        mask = np.ones(len(wavenumbers), dtype=bool)   # safety fallback
    wn = wavenumbers[mask]

    sc = spectra_cdae[:, mask]
    sv = spectra_cvae[:, mask]
    sb = spectra_beta_vae[:, mask]

    # ── Per-spectrum normalisation within the window ──────────────────────────
    sc_n = _minmax_norm(sc)
    sv_n = _minmax_norm(sv)
    sb_n = _minmax_norm(sb)

    # ── Waterfall offset (60 % of 90th-percentile peak-to-peak) ──────────────
    all_s       = np.concatenate([sc_n, sv_n, sb_n], axis=0)
    ptp         = all_s.max(axis=1) - all_s.min(axis=1)
    offset_step = float(np.percentile(ptp, 90)) * 0.60

    cmap = cm.coolwarm
    N    = len(alphas)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 7),
        sharey=False,
        constrained_layout=True,
    )

    panels = list(zip(axes, [sc_n, sv_n, sb_n], _PANEL_TITLES))

    for ax, spectra_n, model_title in panels:

        # Draw each interpolated spectrum as a coloured line
        for i in range(N):
            alpha  = float(alphas[i])
            color  = cmap(alpha)
            offset = i * offset_step
            lw     = 1.8 if (i == 0 or i == N - 1) else 0.85
            ax.plot(
                wn,
                spectra_n[i] + offset,
                color     = color,
                linewidth = lw,
                zorder    = N - i,
                alpha     = 0.90,
                rasterized= True,
            )

        # Annotations: Point A (bottom of stack, blue)
        ax.text(
            0.02, 0.03,
            r"$\alpha = 0$" + "\n" + label_a,
            transform = ax.transAxes,
            fontsize  = 7.5,
            color     = cmap(0.0),
            va        = "bottom",
            ha        = "left",
            bbox      = dict(fc="white", ec="none", alpha=0.85, pad=2),
        )
        # Annotations: Point B (top of stack, red)
        ax.text(
            0.02, 0.97,
            r"$\alpha = 1$" + "\n" + label_b,
            transform = ax.transAxes,
            fontsize  = 7.5,
            color     = cmap(1.0),
            va        = "top",
            ha        = "left",
            bbox      = dict(fc="white", ec="none", alpha=0.85, pad=2),
        )

        ax.set_title(model_title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(r"Wavenumber (cm$^{-1}$)", fontsize=11)
        ax.set_ylabel(
            "Normalised intensity + offset  (a.u.)",
            fontsize=11, labelpad=6,
        )
        ax.set_xlim(wn[0], wn[-1])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── Shared colorbar ───────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes,
        orientation = "vertical",
        fraction    = 0.012,
        pad         = 0.02,
        aspect      = 30,
    )
    cbar.set_label(
        r"Interpolation parameter $\alpha$",
        fontsize=11, labelpad=10,
    )
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(
        [r"0  (Point A)", "0.5", r"1  (Point B)"],
        fontsize=9,
    )

    # ── Page title ────────────────────────────────────────────────────────────
    fig.suptitle(page_title, fontsize=12, fontweight="bold")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Latent-space topological stress-tests (CDAE / CVAE / β-VAE)"
    )
    p.add_argument("--csv",     default=CSV_PATH,   help="Path to IndPenSim CSV")
    p.add_argument("--output",  default=OUTPUT_PDF, help="Output PDF path")
    p.add_argument(
        "--n-steps", type=int, default=N_STEPS,
        help="Number of interpolation steps (default: 15)",
    )
    p.add_argument(
        "--save-png", action="store_true",
        help="Also save a PNG for each scenario.",
    )
    p.add_argument(
        "--zoom-only", action="store_true",
        help=(
            "Output only the zoomed 1500–1800 cm⁻¹ pages "
            "(one per scenario, 3-page PDF).  Skips full-range pages."
        ),
    )
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    n_steps = args.n_steps

    print(f"\nDevice : {DEVICE}")
    print(f"Steps  : {n_steps}")
    print(f"Output : {args.output}\n")

    # ── Load everything ───────────────────────────────────────────────────────
    batches, wavenumbers, raman_cols = load_data(args.csv)
    cdae_sc, cvae_sc, bvae_sc        = load_scalers()
    cdae, cvae, beta_vae             = load_models()

    wn_min = float(wavenumbers[0])
    wn_max = float(wavenumbers[-1])

    # ── Scenario registry ─────────────────────────────────────────────────────
    scenarios = [
        ("Cross-Regime",       scenario_cross_regime),
        ("Failure vs Success", scenario_fail_vs_success),
        ("Turbulence Zone",    scenario_turbulence_zone),
    ]

    # ── Output directory ──────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Open PDF and iterate scenarios ────────────────────────────────────────
    with PdfPages(str(out_path)) as pdf:

        # ── PDF metadata ──────────────────────────────────────────────────────
        info = pdf.infodict()
        info["Title"]   = "Latent Space Topological Stress-Tests"
        info["Subject"] = "CDAE vs CVAE vs β-VAE  —  IndPenSim V3"
        info["Keywords"]= "Raman, autoencoder, interpolation, fermentation"

        for sc_idx, (sc_name, sc_fn) in enumerate(scenarios, start=1):
            print(f"\n{'─' * 65}")
            print(f"  Scenario {sc_idx}: {sc_name}")

            try:
                # ── Extract pair ──────────────────────────────────────────────
                spec_a, spec_b, label_a, label_b, title = sc_fn(
                    batches, raman_cols,
                )
                print(f"    {label_a.splitlines()[0]}")
                print(f"    {label_b.splitlines()[0]}")

                # ── Encode → interpolate → decode ─────────────────────────────
                alphas, dec_cdae, dec_cvae, dec_bvae = run_interpolation(
                    spec_a, spec_b,
                    cdae,    cvae,    beta_vae,
                    cdae_sc, cvae_sc, bvae_sc,
                    n_steps = n_steps,
                )
                print(f"    Decoded spectra: {dec_cdae.shape}  ✓")

                # ── Page (full range) — skipped when --zoom-only ──────────────
                if not args.zoom_only:
                    fig_full = make_waterfall_figure(
                        wavenumbers      = wavenumbers,
                        alphas           = alphas,
                        spectra_cdae     = dec_cdae,
                        spectra_cvae     = dec_cvae,
                        spectra_beta_vae = dec_bvae,
                        label_a          = label_a,
                        label_b          = label_b,
                        page_title       = (
                            f"{title}\n"
                            f"[Full range  {wn_min:.0f}–{wn_max:.0f} cm⁻¹]"
                        ),
                        x_lo = wn_min,
                        x_hi = wn_max,
                    )
                    pdf.savefig(fig_full, bbox_inches="tight")

                    if args.save_png:
                        png_name = sc_name.lower().replace(" ", "_")
                        png_path = out_path.parent / f"stress_{png_name}_full.png"
                        fig_full.savefig(str(png_path), dpi=200, bbox_inches="tight")
                        print(f"    PNG saved → {png_path.name}")

                    plt.close(fig_full)
                    print(f"    Page (full range)  → PDF")

                # ── Page (zoom 1500–1800 cm⁻¹) ────────────────────────────────
                fig_zoom = make_waterfall_figure(
                    wavenumbers      = wavenumbers,
                    alphas           = alphas,
                    spectra_cdae     = dec_cdae,
                    spectra_cvae     = dec_cvae,
                    spectra_beta_vae = dec_bvae,
                    label_a          = label_a,
                    label_b          = label_b,
                    page_title       = (
                        f"{title}\n"
                        f"[Zoom  {ZOOM_LO:.0f}–{ZOOM_HI:.0f} cm⁻¹"
                        "  —  penicillin C=O / amide region]"
                    ),
                    x_lo = ZOOM_LO,
                    x_hi = ZOOM_HI,
                )
                pdf.savefig(fig_zoom, bbox_inches="tight")

                if args.save_png:
                    png_name = sc_name.lower().replace(" ", "_")
                    png_path = out_path.parent / f"stress_{png_name}_zoom.png"
                    fig_zoom.savefig(str(png_path), dpi=200, bbox_inches="tight")
                    print(f"    PNG saved → {png_path.name}")

                plt.close(fig_zoom)
                print(f"    Page (zoom {ZOOM_LO:.0f}–{ZOOM_HI:.0f} cm⁻¹)  → PDF")

            except Exception as exc:
                print(f"\n    [ERROR] Scenario '{sc_name}' failed:")
                traceback.print_exc()
                # Gracefully insert an error page so the PDF remains valid
                fig_err = plt.figure(figsize=(18, 7))
                fig_err.text(
                    0.5, 0.5,
                    f"Scenario '{sc_name}' could not be generated.\n\n{exc}",
                    ha="center", va="center",
                    fontsize=13, color="red",
                    wrap=True,
                )
                pdf.savefig(fig_err, bbox_inches="tight")
                plt.close(fig_err)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  PDF saved  →  {out_path}")
    n_pages = len(scenarios) if args.zoom_only else len(scenarios) * 2
    print(f"  Total pages: {n_pages}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
