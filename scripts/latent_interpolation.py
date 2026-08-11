#!/usr/bin/env python
"""
scripts/latent_interpolation.py
================================
Latent-space interpolation between two real Raman spectra for CDAE vs CVAE vs β-VAE.

Scientific goal
---------------
Compare the latent manifold geometry of three autoencoders:
  CDAE   — deterministic, no probabilistic prior
  CVAE   — variational (β=1), KL-regularised
  β-VAE  — variational (β=0.005, sum/batch loss), stronger disentanglement prior

Method
------
1. Extract two spectra from batch 1: one at t ≈ 20 h (early fermentation)
   and one at t ≈ 150 h (late fermentation, high penicillin titre).
2. Encode each spectrum through all three autoencoders (posterior mean μ for
   the two VAEs).
3. Linearly interpolate N=15 vectors between the start/end latents.
4. Decode back to spectrum space and invert the StandardScaler.
5. Plot a three-panel publication-quality waterfall figure (coolwarm colormap).

Run from project root:
    python scripts/latent_interpolation.py

Outputs:
    outputs/latent_interpolation/latent_interpolation.png  (300 dpi)
    outputs/latent_interpolation/latent_interpolation.pdf  (vector, for journals)
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Path bootstrap: ensure project root is importable when called as a script ──
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter

from src.autoencoder.model import CDAE_Raman, CVAE_Raman, BetaVAE_Raman
from src.data.dataset import RAMAN_START_COL_IDX


# ─────────────────────────────────────────────────────────────────────────────
# Configuration  ── adjust these paths if your layout differs
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH        = "./100_Batches_IndPenSim_V3.csv"
CDAE_CKPT       = "./checkpoints/cdae_best.pt"
CDAE_SCALER     = "./checkpoints/cdae_scaler.joblib"
CVAE_CKPT       = "./checkpoints/cvae_best.pt"
CVAE_SCALER     = "./checkpoints/cvae_scaler.joblib"
BETA_VAE_CKPT   = "./checkpoints/beta_vae_best.pt"
BETA_VAE_SCALER = "./checkpoints/beta_vae_scaler.joblib"
OUTPUT_PATH     = "./outputs/latent_interpolation/latent_interpolation.png"

BATCH_INDEX = 0      # 0-indexed batch; batch 1 in the paper = index 0
T_START     = 20.0   # fermentation time for x_start (hours)
T_END       = 150.0  # fermentation time for x_end   (hours)
N_STEPS     = 15     # total steps including both endpoints

# Savitzky-Golay parameters (must match training)
SG_WINDOW = 15
SG_POLY   = 2
SG_DERIV  = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_RAMAN = 2001       # number of wavenumber channels


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_spectra(
    csv_path: str,
    batch_index: int,
    t_start: float,
    t_end: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the IndPenSim CSV and extract two raw Raman spectra from one batch.

    Parameters
    ----------
    csv_path    : path to 100_Batches_IndPenSim_V3.csv
    batch_index : 0-indexed batch number
    t_start     : fermentation time (h) for the first spectrum
    t_end       : fermentation time (h) for the second spectrum

    Returns
    -------
    wavenumbers : (L,)  float64 — wavenumber axis in ascending order (cm⁻¹)
    x_start     : (L,)  float32 — raw spectrum closest to t_start
    x_end       : (L,)  float32 — raw spectrum closest to t_end
    """
    print(f"[1] Loading {csv_path} …")
    df = pd.read_csv(csv_path, low_memory=False)

    # ── Detect batch boundaries via time resets ────────────────────────────
    t_vals    = df["Time (h)"].values.astype(float)
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid

    batch_df = df[batch_ids == batch_index].reset_index(drop=True)
    if len(batch_df) == 0:
        raise ValueError(f"Batch index {batch_index} not found in {csv_path}.")

    batch_t = batch_df["Time (h)"].values.astype(float)

    # ── Select rows closest to the requested times ─────────────────────────
    idx_s = int(np.argmin(np.abs(batch_t - t_start)))
    idx_e = int(np.argmin(np.abs(batch_t - t_end)))
    print(
        f"    Batch {batch_index + 1}: "
        f"x_start @ t = {batch_t[idx_s]:.1f} h  (row {idx_s}),  "
        f"x_end   @ t = {batch_t[idx_e]:.1f} h  (row {idx_e})"
    )

    # ── Extract Raman columns ──────────────────────────────────────────────
    all_cols   = batch_df.columns.tolist()
    raman_cols = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + N_RAMAN]
    if len(raman_cols) < N_RAMAN:
        raise ValueError(
            f"Expected {N_RAMAN} Raman columns starting at index "
            f"{RAMAN_START_COL_IDX}; found {len(raman_cols)}."
        )

    # Column names are wavenumber values stored as strings (e.g. '2400', '2399', …)
    # They are in descending order in the CSV; reverse to get ascending x-axis.
    raw_nums = []
    for c in raman_cols:
        try:
            raw_nums.append(float(c))
        except ValueError:
            raw_nums.append(float("nan"))

    wavenumbers = np.array(raw_nums, dtype=np.float64)  # may be descending
    x_start_raw = batch_df[raman_cols].iloc[idx_s].values.astype(np.float32)
    x_end_raw   = batch_df[raman_cols].iloc[idx_e].values.astype(np.float32)

    # Reverse to ascending wavenumber order (conventional Raman presentation)
    if wavenumbers[0] > wavenumbers[-1]:
        wavenumbers = wavenumbers[::-1].copy()
        x_start_raw = x_start_raw[::-1].copy()
        x_end_raw   = x_end_raw[::-1].copy()

    # Clamp negative detector artefacts to zero
    x_start_raw = np.clip(x_start_raw, 0.0, None)
    x_end_raw   = np.clip(x_end_raw,   0.0, None)

    return wavenumbers, x_start_raw, x_end_raw


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Preprocessing  (must mirror the autoencoder training pipeline exactly)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(spectrum: np.ndarray, scaler) -> np.ndarray:
    """
    Apply SG first-derivative filter followed by StandardScaler normalisation.

    This reproduces the exact preprocessing used during CDAE / CVAE training
    so that encoded latents lie in the distribution the models were trained on.

    Parameters
    ----------
    spectrum : (L,) raw Raman intensities
    scaler   : fitted sklearn StandardScaler

    Returns
    -------
    x_proc : (L,) preprocessed spectrum  (float32)
    """
    # SG filter operates along axis=1; reshape to (1, L)
    x_sg = savgol_filter(
        spectrum.astype(np.float64).reshape(1, -1),
        window_length=SG_WINDOW,
        polyorder=SG_POLY,
        deriv=SG_DERIV,
        axis=1,
    ).astype(np.float32)                            # (1, L)

    # StandardScaler.transform expects (N, L)
    x_scaled = scaler.transform(x_sg).astype(np.float32)   # (1, L)

    return x_scaled.squeeze(0)                      # (L,)


def to_tensor(x: np.ndarray) -> torch.Tensor:
    """
    Convert a (L,) numpy array to a (1, L) float32 tensor on DEVICE.
    The batch dimension is required by CDAE_Raman.encode / CVAE_Raman.encode.
    """
    return torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_cdae(ckpt_path: str) -> CDAE_Raman:
    """Load and freeze a CDAE_Raman from a checkpoint."""
    print("[2] Loading CDAE …")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = CDAE_Raman(
        input_length=int(state.get("input_length", N_RAMAN)),
        latent_dim  =int(state.get("latent_dim",   64)),
        noise_std   =0.0,       # deterministic inference; no denoising noise
    ).to(DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"    latent_dim={model.latent_dim}, input_length={model.input_length}")
    return model


def load_cvae(ckpt_path: str) -> CVAE_Raman:
    """Load and freeze a CVAE_Raman from a checkpoint."""
    print("[3] Loading CVAE …")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = CVAE_Raman(
        input_length=int(state.get("input_length", N_RAMAN)),
        latent_dim  =int(state.get("latent_dim",   64)),
    ).to(DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"    latent_dim={model.latent_dim}, input_length={model.input_length}")
    return model


def load_beta_vae(ckpt_path: str) -> BetaVAE_Raman:
    """Load and freeze a BetaVAE_Raman from a checkpoint."""
    print("[3b] Loading β-VAE …")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = BetaVAE_Raman(
        input_length=int(state.get("input_length", N_RAMAN)),
        latent_dim  =int(state.get("latent_dim",   64)),
    ).to(DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    beta_info = state.get("beta_schedule", {})
    print(
        f"    latent_dim={model.latent_dim}, input_length={model.input_length}  "
        f"β_max={beta_info.get('max_beta', '?')}"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Latent-space interpolation & decoding
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def interpolate_and_decode(
    model,
    z_start:    torch.Tensor,   # (1, D)  start latent
    z_end:      torch.Tensor,   # (1, D)  end latent
    n_steps:    int,
    scaler,                     # fitted StandardScaler for inverse transform
    model_type: str,            # "CDAE" or "CVAE" (controls encode call)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate N linearly interpolated decoded spectra between z_start and z_end.

    The interpolation formula is:
        z_α = (1 − α) · z_start + α · z_end,   α ∈ [0, 1]

    The decoded spectra are inverse-transformed through the StandardScaler so
    the y-axis represents SG first-derivative intensities in the same units
    as the model's training space.

    Parameters
    ----------
    model      : CDAE_Raman or CVAE_Raman (must have .decode())
    z_start    : (1, D) tensor — latent vector for the start spectrum
    z_end      : (1, D) tensor — latent vector for the end spectrum
    n_steps    : total number of spectra (including endpoints)
    scaler     : fitted StandardScaler — used to invert the scaling
    model_type : informational string for logging only

    Returns
    -------
    alphas  : (N,)    interpolation parameters
    spectra : (N, L)  decoded and inverse-scaled spectra
    """
    alphas  = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    spectra = []

    for alpha in alphas:
        # Linear convex combination in 64-D latent space
        z_alpha = (1.0 - float(alpha)) * z_start + float(alpha) * z_end  # (1, D)

        # Decode: (1, D) → (1, L)
        recon = model.decode(z_alpha).squeeze(0).cpu().numpy()            # (L,)

        # Invert StandardScaler: map from model output space → SG deriv. space
        # The decoder was trained to output StandardScaler-normalised spectra
        recon_sg = scaler.inverse_transform(
            recon.reshape(1, -1)
        ).squeeze(0).astype(np.float32)                                   # (L,)

        spectra.append(recon_sg)

    return alphas, np.stack(spectra, axis=0)   # (N, L)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Publication-quality figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_interpolation(
    wavenumbers:      np.ndarray,     # (L,)  ascending wavenumbers
    alphas:           np.ndarray,     # (N,)  α ∈ [0, 1]
    spectra_cdae:     np.ndarray,     # (N, L)
    spectra_cvae:     np.ndarray,     # (N, L)
    spectra_beta_vae: np.ndarray,     # (N, L)
    output_path:      str,
) -> None:
    """
    Three-panel waterfall plot: CDAE | CVAE | β-VAE, coolwarm colormap.

    Visual conventions
    ------------------
    • Color encodes α:  blue (α=0, t≈20 h) → red (α=1, t≈150 h)
    • Successive spectra are offset upward (waterfall aesthetic)
    • All spectra are min-max normalised per-spectrum
    • Endpoint spectra drawn with heavier line weight
    • Top and right spines removed (academic style)
    • Shared colorbar encodes α
    """
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.labelsize":    11,
        "axes.titlesize":    12,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    cmap = cm.coolwarm
    N    = len(alphas)

    def minmax_norm(S: np.ndarray) -> np.ndarray:
        lo  = S.min(axis=1, keepdims=True)
        hi  = S.max(axis=1, keepdims=True)
        rng = np.where((hi - lo) > 0, hi - lo, 1.0)
        return (S - lo) / rng

    sc_n  = minmax_norm(spectra_cdae)
    sv_n  = minmax_norm(spectra_cvae)
    sb_n  = minmax_norm(spectra_beta_vae)

    # Common vertical spacing: 60% of the 90th-percentile peak-to-peak range
    all_stacked = np.concatenate([sc_n, sv_n, sb_n], axis=0)
    ptp         = all_stacked.max(axis=1) - all_stacked.min(axis=1)
    offset_step = float(np.percentile(ptp, 90)) * 0.60

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 7),
        sharey=False,
        constrained_layout=True,
    )

    panels = [
        (axes[0], sc_n, "CDAE  —  Convolutional Denoising AE"),
        (axes[1], sv_n, "CVAE  —  Variational AE  (β = 1)"),
        (axes[2], sb_n, r"$\beta$-VAE  —  Disentangled VAE  ($\beta$ = 0.005)"),
    ]

    for ax, spectra_n, title in panels:
        for i in range(N):
            alpha  = float(alphas[i])
            color  = cmap(alpha)
            offset = i * offset_step
            lw     = 1.8 if (i == 0 or i == N - 1) else 0.85
            zorder = N - i

            ax.plot(
                wavenumbers,
                spectra_n[i] + offset,
                color=color,
                linewidth=lw,
                zorder=zorder,
                alpha=0.90,
                rasterized=True,
            )

        ax.text(
            0.02, 0.04,
            r"$\alpha = 0$" "\n" r"($t \approx 20\,\mathrm{h}$)",
            transform=ax.transAxes,
            fontsize=8.5, color=cmap(0.0),
            va="bottom", ha="left",
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=2),
        )
        ax.text(
            0.02, 0.96,
            r"$\alpha = 1$" "\n" r"($t \approx 150\,\mathrm{h}$)",
            transform=ax.transAxes,
            fontsize=8.5, color=cmap(1.0),
            va="top", ha="left",
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=2),
        )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(r"Wavenumber (cm$^{-1}$)", fontsize=11)
        ax.set_ylabel(
            "Normalised intensity + offset  (a.u.)",
            fontsize=11, labelpad=6,
        )
        ax.set_xlim(wavenumbers[0], wavenumbers[-1])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes,
        orientation="vertical",
        fraction=0.012, pad=0.02,
        aspect=30,
    )
    cbar.set_label(
        r"Interpolation parameter $\alpha$",
        fontsize=11, labelpad=10,
    )
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(
        [r"0  ($t\approx 20\,\mathrm{h}$)", "0.5",
         r"1  ($t\approx 150\,\mathrm{h}$)"],
        fontsize=9,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    fig.savefig(str(out.with_suffix(".pdf")), bbox_inches="tight")
    print(f"\n[6] Figure saved:")
    print(f"    {out}")
    print(f"    {out.with_suffix('.pdf')}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Device: {DEVICE}\n")

    # ── 1. Load spectra ───────────────────────────────────────────────────────
    wavenumbers, x_start_raw, x_end_raw = load_spectra(
        CSV_PATH, BATCH_INDEX, T_START, T_END
    )

    # ── 2. Load scalers ───────────────────────────────────────────────────────
    cdae_scaler     = joblib.load(CDAE_SCALER)
    cvae_scaler     = joblib.load(CVAE_SCALER)
    beta_vae_scaler = joblib.load(BETA_VAE_SCALER)

    # ── 3. Preprocess ─────────────────────────────────────────────────────────
    # Spectra must be passed in the CSV's original descending wavenumber order
    # (that is the order the scalers were fitted on during training).
    x_start_desc = x_start_raw[::-1].copy()
    x_end_desc   = x_end_raw[::-1].copy()

    x_start_cdae     = preprocess(x_start_desc, cdae_scaler)
    x_end_cdae       = preprocess(x_end_desc,   cdae_scaler)
    x_start_cvae     = preprocess(x_start_desc, cvae_scaler)
    x_end_cvae       = preprocess(x_end_desc,   cvae_scaler)
    x_start_beta_vae = preprocess(x_start_desc, beta_vae_scaler)
    x_end_beta_vae   = preprocess(x_end_desc,   beta_vae_scaler)

    # ── 4. Load models ────────────────────────────────────────────────────────
    cdae     = load_cdae(CDAE_CKPT)
    cvae     = load_cvae(CVAE_CKPT)
    beta_vae = load_beta_vae(BETA_VAE_CKPT)

    # ── 5. Encode ─────────────────────────────────────────────────────────────
    print("[4] Encoding spectra …")
    with torch.no_grad():
        z_start_cdae = cdae.encode(to_tensor(x_start_cdae))     # (1, 64)
        z_end_cdae   = cdae.encode(to_tensor(x_end_cdae))

        mu_start_cvae, _ = cvae.encode(to_tensor(x_start_cvae))
        mu_end_cvae,   _ = cvae.encode(to_tensor(x_end_cvae))

        # BetaVAE_Raman.encode() has the same signature as CVAE_Raman.encode()
        mu_start_bvae, _ = beta_vae.encode(to_tensor(x_start_beta_vae))
        mu_end_bvae,   _ = beta_vae.encode(to_tensor(x_end_beta_vae))

    print(
        f"    CDAE   z_start: {tuple(z_start_cdae.shape)}\n"
        f"    CVAE   μ_start: {tuple(mu_start_cvae.shape)}\n"
        f"    β-VAE  μ_start: {tuple(mu_start_bvae.shape)}"
    )

    # ── 6. Interpolate & decode ───────────────────────────────────────────────
    print(f"\n[5] Interpolating ({N_STEPS} steps) …")

    alphas, spectra_cdae = interpolate_and_decode(
        cdae, z_start_cdae, z_end_cdae,
        n_steps=N_STEPS, scaler=cdae_scaler, model_type="CDAE",
    )
    spectra_cdae = spectra_cdae[:, ::-1].copy()   # descending → ascending

    _, spectra_cvae = interpolate_and_decode(
        cvae, mu_start_cvae, mu_end_cvae,
        n_steps=N_STEPS, scaler=cvae_scaler, model_type="CVAE",
    )
    spectra_cvae = spectra_cvae[:, ::-1].copy()

    _, spectra_beta_vae = interpolate_and_decode(
        beta_vae, mu_start_bvae, mu_end_bvae,
        n_steps=N_STEPS, scaler=beta_vae_scaler, model_type="β-VAE",
    )
    spectra_beta_vae = spectra_beta_vae[:, ::-1].copy()

    print(f"    CDAE  decoded: {spectra_cdae.shape}")
    print(f"    CVAE  decoded: {spectra_cvae.shape}")
    print(f"    β-VAE decoded: {spectra_beta_vae.shape}")

    # ── 7. Figure ─────────────────────────────────────────────────────────────
    print("\n[6] Generating figure …")
    plot_interpolation(
        wavenumbers      = wavenumbers,
        alphas           = alphas,
        spectra_cdae     = spectra_cdae,
        spectra_cvae     = spectra_cvae,
        spectra_beta_vae = spectra_beta_vae,
        output_path      = OUTPUT_PATH,
    )

    # ── 8. Export raw data ────────────────────────────────────────────────────
    out_dir = Path(OUTPUT_PATH).parent
    print("\n[7] Exporting raw data …")

    wn_cols = [f"{int(round(w))}" for w in wavenumbers]

    for tag, spectra in [
        ("cdae",     spectra_cdae),
        ("cvae",     spectra_cvae),
        ("beta_vae", spectra_beta_vae),
    ]:
        df_out = pd.DataFrame(spectra, columns=wn_cols)
        df_out.insert(0, "alpha",         alphas.astype(np.float64))
        df_out.insert(1, "approx_time_h",
                      (alphas * (T_END - T_START) + T_START).astype(np.float64))
        csv_path = out_dir / f"interpolation_{tag}.csv"
        df_out.to_csv(csv_path, index=False, float_format="%.6f")
        print(f"    {csv_path}  ({df_out.shape[0]} steps × {df_out.shape[1]} cols)")

    npz_path = out_dir / "interpolation_data.npz"
    np.savez(
        str(npz_path),
        wavenumbers      = wavenumbers,
        alphas           = alphas,
        spectra_cdae     = spectra_cdae,
        spectra_cvae     = spectra_cvae,
        spectra_beta_vae = spectra_beta_vae,
    )
    print(f"    {npz_path}  (numpy archive)")

    print("\nDone.")


if __name__ == "__main__":
    main()
