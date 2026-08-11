#!/usr/bin/env python
"""
scripts/latent_smoothness.py
=============================
Quantitative latent-space smoothness evaluation for CDAE, CVAE, and beta-VAE
via L2-distance profiling along a linear interpolation path.

Protocol
--------
Scenario  :  Turbulence Zone  (Batch 10, Regime 1)
  Point A  :  t ~ 40 h  (onset of exponential growth)
  Point B  :  t ~ 80 h  (peak biomass / maximal broth complexity)

For each model:
  1. Preprocess x_A, x_B  (clamp → SG d1 filter → StandardScaler)
  2. Encode  →  z_A, z_B  (posterior mean mu for VAE-type models)
  3. Interpolate  z_alpha = (1 - alpha) * z_A + alpha * z_B
     over alpha in [0, 1] with N_STEPS linearly-spaced steps.
  4. Decode each z_alpha  →  x_hat_alpha  (decoder output space)
  5. Compute  L2(alpha) = ||x_hat_alpha - x_hat_0||_2

Output
------
  outputs/latent_smoothness/latent_smoothness.png   (300 dpi)
  outputs/latent_smoothness/latent_smoothness.pdf   (vector)
  outputs/latent_smoothness/smoothness_values.npz   (raw arrays)

Run from project root:
    python scripts/latent_smoothness.py [--batch N] [--t-a H] [--t-b H]
                                        [--n-steps N] [--device cuda|cpu]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

OUTPUT_DIR      = Path("./outputs/latent_smoothness")

# Turbulence Zone defaults (Batch 10 = index 9, Regime 1)
DEFAULT_BATCH  = 9     # 0-indexed
DEFAULT_T_A    = 40.0  # h  — onset of exponential growth
DEFAULT_T_B    = 80.0  # h  — peak biomass
DEFAULT_STEPS  = 100   # number of interpolation steps (alpha grid)

# Savitzky-Golay params — must match autoencoder training
SG_WINDOW = 15
SG_POLY   = 2
SG_DERIV  = 1

N_RAMAN = 2001

# Visual style
MODEL_STYLES = {
    "CDAE":    {"color": "#e74c3c", "lw": 2.2, "ls": "-",  "zorder": 3},
    "CVAE":    {"color": "#2980b9", "lw": 2.2, "ls": "-",  "zorder": 3},
    "beta-VAE":{"color": "#27ae60", "lw": 2.2, "ls": "-",  "zorder": 3},
}


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_batches(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Split flat DataFrame into per-batch DataFrames using time-reset detection."""
    t_vals    = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid
    df = df.copy()
    df["_batch_id"] = batch_ids
    return {
        int(b): grp.drop(columns=["_batch_id"]).reset_index(drop=True)
        for b, grp in df.groupby("_batch_id")
    }


def _find_closest_row(batch_df: pd.DataFrame, t_target: float) -> int:
    """Return row index whose 'Time (h)' is closest to t_target."""
    return int(np.argmin(np.abs(batch_df["Time (h)"].values - t_target)))


def load_spectra(
    csv_path: str,
    batch_index: int,
    t_a: float,
    t_b: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load two raw Raman spectra from one batch.

    Returns
    -------
    wavenumbers : (N_RAMAN,) float64  — ascending wavenumber axis (cm-1)
    x_a         : (N_RAMAN,) float32  — spectrum at t_a
    x_b         : (N_RAMAN,) float32  — spectrum at t_b
    """
    print(f"[data]  Reading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    batches = _detect_batches(df)

    if batch_index not in batches:
        raise ValueError(
            f"Batch index {batch_index} (batch {batch_index + 1}) "
            f"not found. Available: {sorted(batches.keys())}"
        )

    batch_df = batches[batch_index]
    batch_t  = batch_df["Time (h)"].values.astype(float)

    idx_a = _find_closest_row(batch_df, t_a)
    idx_b = _find_closest_row(batch_df, t_b)
    print(
        f"[data]  Batch {batch_index + 1}: "
        f"A @ t = {batch_t[idx_a]:.1f} h (row {idx_a}),  "
        f"B @ t = {batch_t[idx_b]:.1f} h (row {idx_b})"
    )

    all_cols   = batch_df.columns.tolist()
    raman_cols = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + N_RAMAN]
    if len(raman_cols) < N_RAMAN:
        raise ValueError(
            f"Expected {N_RAMAN} Raman columns from index {RAMAN_START_COL_IDX}; "
            f"found {len(raman_cols)}."
        )

    # Parse wavenumber axis from column names
    wavenumbers = np.array(
        [float(c) if c.replace(".", "", 1).lstrip("-").isdigit() else np.nan
         for c in raman_cols],
        dtype=np.float64,
    )

    x_a = batch_df[raman_cols].iloc[idx_a].values.astype(np.float32)
    x_b = batch_df[raman_cols].iloc[idx_b].values.astype(np.float32)

    # Reverse to ascending wavenumber order if stored descending
    if wavenumbers[0] > wavenumbers[-1]:
        wavenumbers = wavenumbers[::-1].copy()
        x_a = x_a[::-1].copy()
        x_b = x_b[::-1].copy()

    # Clamp negative detector artefacts
    x_a = np.clip(x_a, 0.0, None)
    x_b = np.clip(x_b, 0.0, None)

    return wavenumbers, x_a, x_b


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(spectrum: np.ndarray, scaler) -> np.ndarray:
    """
    Apply SG first-derivative filter followed by StandardScaler normalisation.

    Replicates the exact training pipeline so latent codes lie in the
    distribution the models were trained on.

    Parameters
    ----------
    spectrum : (L,) raw Raman intensities
    scaler   : fitted sklearn StandardScaler

    Returns
    -------
    x_proc : (L,) preprocessed float32
    """
    x_sg = savgol_filter(
        spectrum.astype(np.float64).reshape(1, -1),
        window_length=SG_WINDOW,
        polyorder=SG_POLY,
        deriv=SG_DERIV,
        axis=1,
    ).astype(np.float32)                           # (1, L)
    x_scaled = scaler.transform(x_sg).astype(np.float32)  # (1, L)
    return x_scaled[0]                             # (L,)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_cdae(ckpt_path: str, device: torch.device) -> CDAE_Raman:
    state        = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    input_length = int(state.get("input_length", N_RAMAN))
    latent_dim   = int(state.get("latent_dim",   64))
    model        = CDAE_Raman(input_length=input_length, latent_dim=latent_dim)
    model.load_state_dict(state["model_state_dict"])
    model.eval().to(device)
    print(f"[load]  CDAE     input_length={input_length}  latent_dim={latent_dim}")
    return model


def load_cvae(ckpt_path: str, device: torch.device) -> CVAE_Raman:
    state        = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    input_length = int(state.get("input_length", N_RAMAN))
    latent_dim   = int(state.get("latent_dim",   64))
    model        = CVAE_Raman(input_length=input_length, latent_dim=latent_dim)
    model.load_state_dict(state["model_state_dict"])
    model.eval().to(device)
    print(f"[load]  CVAE     input_length={input_length}  latent_dim={latent_dim}")
    return model


def load_beta_vae(ckpt_path: str, device: torch.device) -> BetaVAE_Raman:
    state        = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    input_length = int(state.get("input_length", N_RAMAN))
    latent_dim   = int(state.get("latent_dim",   64))
    model        = BetaVAE_Raman(input_length=input_length, latent_dim=latent_dim)
    model.load_state_dict(state["model_state_dict"])
    model.eval().to(device)
    print(f"[load]  beta-VAE input_length={input_length}  latent_dim={latent_dim}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Core: encode, interpolate, decode, compute L2 profile
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_l2_profile(
    model,
    x_a_proc: np.ndarray,
    x_b_proc: np.ndarray,
    n_steps:  int,
    device:   torch.device,
    is_vae:   bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode two preprocessed spectra, interpolate in latent space,
    decode all N_STEPS+1 points, and compute L2 distance from the origin.

    Parameters
    ----------
    model     : CDAE_Raman | CVAE_Raman | BetaVAE_Raman  (already .eval())
    x_a_proc  : (L,) preprocessed spectrum for Point A
    x_b_proc  : (L,) preprocessed spectrum for Point B
    n_steps   : number of interpolation steps (alpha grid size, includes 0 and 1)
    device    : torch device
    is_vae    : True  → encode() returns (mu, log_var); use mu only
                False → encode() returns z directly (CDAE)

    Returns
    -------
    alphas     : (n_steps,) float64  — interpolation parameter in [0, 1]
    l2_profile : (n_steps,) float64  — L2 distance from decoded origin
    """
    t_a = torch.tensor(x_a_proc[None], dtype=torch.float32, device=device)  # (1, L)
    t_b = torch.tensor(x_b_proc[None], dtype=torch.float32, device=device)  # (1, L)

    # ── Encode ────────────────────────────────────────────────────────────────
    if is_vae:
        mu_a, _ = model.encode(t_a)   # (1, latent_dim)
        mu_b, _ = model.encode(t_b)
        z_a = mu_a
        z_b = mu_b
    else:
        z_a = model.encode(t_a)       # (1, latent_dim)
        z_b = model.encode(t_b)

    # ── Build interpolation grid ───────────────────────────────────────────────
    alphas = np.linspace(0.0, 1.0, n_steps)   # shape (n_steps,)

    # Stack all interpolated latents into one batch → (n_steps, latent_dim)
    alpha_t = torch.tensor(alphas, dtype=torch.float32, device=device)[:, None]
    z_interp = (1.0 - alpha_t) * z_a + alpha_t * z_b  # (n_steps, latent_dim)

    # ── Decode all at once ────────────────────────────────────────────────────
    x_hat = model.decode(z_interp)   # (n_steps, L)

    # ── L2 distance from the origin reconstruction (alpha = 0) ───────────────
    x_hat_0   = x_hat[0:1]                              # (1, L)  — decoded origin
    diff      = x_hat - x_hat_0                         # (n_steps, L)
    l2_profile = diff.pow(2).sum(dim=1).sqrt()          # (n_steps,)

    return alphas, l2_profile.cpu().numpy().astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_smoothness(
    results:   dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    batch_index: int,
    t_a: float,
    t_b: float,
) -> None:
    """
    Single-panel line chart: L2(alpha) for CDAE, CVAE, and beta-VAE.

    Parameters
    ----------
    results     : {model_name: (alphas, l2_profile)}
    output_dir  : destination directory
    batch_index : 0-indexed batch used (for title annotation)
    t_a, t_b    : endpoint times in hours
    """
    rc = {
        "font.family":      "sans-serif",
        "font.size":        11,
        "axes.linewidth":   0.9,
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "xtick.direction":  "out",
        "ytick.direction":  "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "legend.frameon":   False,
        "figure.dpi":       150,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(8, 5))

        for name, (alphas, l2) in results.items():
            style = MODEL_STYLES[name]
            ax.plot(
                alphas, l2,
                label=name,
                color=style["color"],
                linewidth=style["lw"],
                linestyle=style["ls"],
                zorder=style["zorder"],
            )

        ax.set_xlabel(r"Interpolation parameter $\alpha$", fontsize=12)
        ax.set_ylabel(
            r"$\Delta L_2(\alpha) = \|\hat{X}_\alpha - \hat{X}_0\|_2$",
            fontsize=12,
        )
        ax.set_title(
            f"Latent Space Smoothness — Turbulence Zone\n"
            f"Batch {batch_index + 1}  |  "
            f"A @ t = {t_a:.0f} h  \u2192  B @ t = {t_b:.0f} h",
            fontsize=12,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(bottom=0.0)
        ax.legend(fontsize=11, loc="upper left")

        # Vertical reference lines at alpha = 0 and 1
        for x_ref in (0.0, 1.0):
            ax.axvline(x_ref, color="grey", linewidth=0.7, linestyle="--", zorder=1)

        plt.tight_layout()

        png_path = output_dir / "latent_smoothness.png"
        pdf_path = output_dir / "latent_smoothness.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path,           bbox_inches="tight")
        plt.close(fig)

    print(f"[plot]  Saved PNG  →  {png_path}")
    print(f"[plot]  Saved PDF  →  {pdf_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Latent-space smoothness profiling (L2 interpolation metric)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",      default=CSV_PATH,         help="Path to IndPenSim V3 CSV")
    p.add_argument("--batch",    type=int, default=DEFAULT_BATCH,
                   help="0-indexed batch number (default: 9 = batch 10)")
    p.add_argument("--t-a",      type=float, default=DEFAULT_T_A,
                   help="Time (h) for Point A")
    p.add_argument("--t-b",      type=float, default=DEFAULT_T_B,
                   help="Time (h) for Point B")
    p.add_argument("--n-steps",  type=int, default=DEFAULT_STEPS,
                   help="Number of alpha steps (including both endpoints)")
    p.add_argument("--device",   default="auto", choices=["auto", "cuda", "cpu"],
                   help="Compute device")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR),
                   help="Directory to save outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[init]  Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load raw spectra ───────────────────────────────────────────────────────
    wavenumbers, x_a_raw, x_b_raw = load_spectra(
        args.csv, args.batch, args.t_a, args.t_b
    )

    # ── Load scalers ───────────────────────────────────────────────────────────
    print("[load]  Loading StandardScalers ...")
    scaler_cdae    = joblib.load(CDAE_SCALER)
    scaler_cvae    = joblib.load(CVAE_SCALER)
    scaler_beta    = joblib.load(BETA_VAE_SCALER)

    # ── Preprocess once per model (each scaler is fitted on its own SG-filtered data)
    print("[prep]  Preprocessing spectra ...")
    x_a_cdae = preprocess(x_a_raw, scaler_cdae)
    x_b_cdae = preprocess(x_b_raw, scaler_cdae)

    x_a_cvae = preprocess(x_a_raw, scaler_cvae)
    x_b_cvae = preprocess(x_b_raw, scaler_cvae)

    x_a_beta = preprocess(x_a_raw, scaler_beta)
    x_b_beta = preprocess(x_b_raw, scaler_beta)

    # ── Load models ────────────────────────────────────────────────────────────
    print("[load]  Loading autoencoder models ...")
    cdae    = load_cdae(CDAE_CKPT,       device)
    cvae    = load_cvae(CVAE_CKPT,       device)
    beta    = load_beta_vae(BETA_VAE_CKPT, device)

    # ── Compute L2 profiles ───────────────────────────────────────────────────
    print(f"[interp]  Computing L2 profiles ({args.n_steps} steps) ...")

    alphas_cdae, l2_cdae = compute_l2_profile(
        cdae, x_a_cdae, x_b_cdae, args.n_steps, device, is_vae=False)
    print(f"  CDAE     — L2 at alpha=0.5: {l2_cdae[args.n_steps // 2]:.4f}  "
          f"  max: {l2_cdae.max():.4f}")

    alphas_cvae, l2_cvae = compute_l2_profile(
        cvae, x_a_cvae, x_b_cvae, args.n_steps, device, is_vae=True)
    print(f"  CVAE     — L2 at alpha=0.5: {l2_cvae[args.n_steps // 2]:.4f}  "
          f"  max: {l2_cvae.max():.4f}")

    alphas_beta, l2_beta = compute_l2_profile(
        beta, x_a_beta, x_b_beta, args.n_steps, device, is_vae=True)
    print(f"  beta-VAE — L2 at alpha=0.5: {l2_beta[args.n_steps // 2]:.4f}  "
          f"  max: {l2_beta.max():.4f}")

    # ── Save raw arrays ────────────────────────────────────────────────────────
    npz_path = out_dir / "smoothness_values.npz"
    np.savez(
        npz_path,
        alphas_cdae=alphas_cdae, l2_cdae=l2_cdae,
        alphas_cvae=alphas_cvae, l2_cvae=l2_cvae,
        alphas_beta=alphas_beta, l2_beta=l2_beta,
        wavenumbers=wavenumbers,
        batch_index=np.array(args.batch),
        t_a=np.array(args.t_a),
        t_b=np.array(args.t_b),
    )
    print(f"[save]  Raw arrays  →  {npz_path}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    results = {
        "CDAE":     (alphas_cdae, l2_cdae),
        "CVAE":     (alphas_cvae, l2_cvae),
        "beta-VAE": (alphas_beta, l2_beta),
    }
    plot_smoothness(results, out_dir, args.batch, args.t_a, args.t_b)

    print(f"\n[done]  All outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
