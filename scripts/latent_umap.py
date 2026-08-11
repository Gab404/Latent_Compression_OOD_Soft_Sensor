#!/usr/bin/env python
"""
scripts/latent_umap.py
======================
3-D UMAP visualisation of the latent spaces learned by CDAE, CVAE and β-VAE
on IndPenSim V3 Raman spectra.

Scientific goal
---------------
Compare how each autoencoder organises the 64-D latent manifold with respect
to two fermentation state variables:
  • Fermentation time  (h)          — proxy for batch progress
  • Penicillin concentration (g/L)  — the process target

Method
------
1. Load the full CSV; detect batch boundaries from time-resets.
2. Optionally sub-sample to ``--n-samples`` points (stratified per batch
   so that every batch contributes proportionally).
3. Preprocess Raman spectra for each model:
       SG(d=1, w=15, p=2)  →  StandardScaler  →  encoder (frozen, μ only)
4. Fit a separate UMAP(n_components=3) on each model's latent matrix.
5. Generate a single 3×2 figure:
       rows  =  CDAE | CVAE | β-VAE
       cols  =  coloured by time | coloured by penicillin

Run from project root
---------------------
    CUDA_VISIBLE_DEVICES=0  python scripts/latent_umap.py
    python scripts/latent_umap.py --n-samples 8000 --umap-neighbors 20

Outputs
-------
    outputs/latent_umap/latent_umap.png   (300 dpi)
    outputs/latent_umap/latent_umap.pdf   (vector)
    outputs/latent_umap/latent_umap.npz   (raw arrays for further analysis)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from scipy.signal import savgol_filter

from src.autoencoder.model import CDAE_Raman, CVAE_Raman, BetaVAE_Raman
from src.data.dataset import RAMAN_START_COL_IDX

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH        = "./100_Batches_IndPenSim_V3.csv"
CDAE_CKPT       = "./checkpoints/cdae_best.pt"
CDAE_SCALER     = "./checkpoints/cdae_scaler.joblib"
CVAE_CKPT       = "./checkpoints/cvae_best.pt"
CVAE_SCALER     = "./checkpoints/cvae_scaler.joblib"
BETA_VAE_CKPT   = "./checkpoints/beta_vae_best.pt"
BETA_VAE_SCALER = "./checkpoints/beta_vae_scaler.joblib"
OUTPUT_DIR      = "./outputs/latent_umap"

N_RAMAN   = 2001
SG_WINDOW = 15
SG_POLY   = 2
SG_DERIV  = 1

TIME_COL        = "Time (h)"
PENICILLIN_COL  = "Penicillin concentration(P:g/L)"   # online (dense)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3-D UMAP of CDAE / CVAE / β-VAE latents")
    p.add_argument("--csv",            default=CSV_PATH)
    p.add_argument("--n-samples",      type=int, default=10_000,
                   help="Total rows to sub-sample (stratified by batch). "
                        "0 = use all rows (slow).")
    p.add_argument("--umap-neighbors", type=int, default=30,
                   help="UMAP n_neighbors parameter.")
    p.add_argument("--umap-min-dist",  type=float, default=0.1,
                   help="UMAP min_dist parameter.")
    p.add_argument("--encode-batch",   type=int, default=2048,
                   help="Mini-batch size for GPU encoding.")
    p.add_argument("--output-dir",     default=OUTPUT_DIR)
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_and_sample(
    csv_path: str,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the IndPenSim CSV, detect batches, and sub-sample.

    Returns
    -------
    X_raw     : (N, N_RAMAN) float32   — raw Raman intensities (CSV column order)
    time_vals : (N,)         float32   — fermentation time (h)
    pen_vals  : (N,)         float32   — penicillin concentration (g/L)
    batch_ids : (N,)         int32     — 0-indexed batch id
    """
    print(f"\n[1] Loading {csv_path} …")
    needed_cols = [TIME_COL, PENICILLIN_COL]
    df_meta = pd.read_csv(csv_path, usecols=needed_cols, low_memory=False)
    df_raman = pd.read_csv(csv_path, low_memory=False,
                           usecols=lambda c: c not in needed_cols)

    # Re-load full for simplicity (CSV is ~1 GB so load once)
    print("  Re-loading full CSV …")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # ── Detect batch boundaries ───────────────────────────────────────────────
    t_vals    = df[TIME_COL].values.astype(np.float32)
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid
    n_batches = int(bid + 1)
    print(f"  Detected {n_batches} batches")

    # ── Extract Raman columns ─────────────────────────────────────────────────
    all_cols   = df.columns.tolist()
    raman_cols = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + N_RAMAN]
    X_raw      = df[raman_cols].values.astype(np.float32)    # (N_total, N_RAMAN)

    # ── Keep only rows with valid Raman signal ────────────────────────────────
    valid_mask = np.abs(X_raw).sum(axis=1) > 0
    print(f"  Valid Raman rows: {valid_mask.sum():,} / {len(df):,}")

    X_raw_v      = X_raw[valid_mask]
    t_vals_v     = t_vals[valid_mask]
    pen_vals_v   = df[PENICILLIN_COL].values.astype(np.float32)[valid_mask]
    batch_ids_v  = batch_ids[valid_mask]

    # ── Sub-sample (stratified by batch) ─────────────────────────────────────
    if n_samples > 0 and n_samples < len(X_raw_v):
        rng        = np.random.default_rng(seed)
        keep_idx   = []
        n_per_batch = max(1, n_samples // n_batches)

        for b in range(n_batches):
            b_idx = np.where(batch_ids_v == b)[0]
            if len(b_idx) == 0:
                continue
            chosen = rng.choice(
                b_idx,
                size=min(n_per_batch, len(b_idx)),
                replace=False,
            )
            keep_idx.append(chosen)

        keep_idx = np.concatenate(keep_idx)
        keep_idx = np.sort(keep_idx)   # keep chronological order within figure

        # Trim to exactly n_samples if over
        if len(keep_idx) > n_samples:
            keep_idx = rng.choice(keep_idx, size=n_samples, replace=False)
            keep_idx = np.sort(keep_idx)

        X_raw_v     = X_raw_v[keep_idx]
        t_vals_v    = t_vals_v[keep_idx]
        pen_vals_v  = pen_vals_v[keep_idx]
        batch_ids_v = batch_ids_v[keep_idx]

        print(f"  Sub-sampled to {len(X_raw_v):,} rows")
    else:
        print(f"  Using all {len(X_raw_v):,} valid rows")

    # Clamp negatives / NaN
    X_raw_v = np.nan_to_num(X_raw_v, nan=0.0, posinf=0.0, neginf=0.0)
    X_raw_v = np.clip(X_raw_v, 0.0, None)

    return X_raw_v, t_vals_v, pen_vals_v, batch_ids_v


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Preprocessing helper
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(X_raw: np.ndarray, scaler) -> np.ndarray:
    """SG first-derivative → StandardScaler.  Returns float32 (N, L)."""
    X_sg     = savgol_filter(
        X_raw.astype(np.float64),
        window_length=SG_WINDOW,
        polyorder=SG_POLY,
        deriv=SG_DERIV,
        axis=1,
    ).astype(np.float32)
    X_scaled = scaler.transform(X_sg).astype(np.float32)
    return X_scaled


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Encoding helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_cdae(
    model: CDAE_Raman,
    X_scaled: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """CDAE: (N, L) → (N, D)."""
    parts = []
    for start in range(0, len(X_scaled), batch_size):
        chunk = torch.tensor(
            X_scaled[start : start + batch_size],
            dtype=torch.float32, device=DEVICE,
        )
        parts.append(model.encode(chunk).cpu().numpy())
    return np.concatenate(parts, axis=0)


@torch.no_grad()
def encode_vae(
    model,
    X_scaled: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """CVAE / β-VAE: (N, L) → (N, D)  using posterior mean μ."""
    parts = []
    for start in range(0, len(X_scaled), batch_size):
        chunk = torch.tensor(
            X_scaled[start : start + batch_size],
            dtype=torch.float32, device=DEVICE,
        )
        mu, _ = model.encode(chunk)
        parts.append(mu.cpu().numpy())
    return np.concatenate(parts, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_cdae_model(ckpt: str) -> CDAE_Raman:
    state  = torch.load(ckpt, map_location="cpu", weights_only=False)
    model  = CDAE_Raman(
        input_length=int(state.get("input_length", N_RAMAN)),
        latent_dim  =int(state.get("latent_dim",   64)),
        noise_std   =0.0,
    ).to(DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_cvae_model(ckpt: str) -> CVAE_Raman:
    state  = torch.load(ckpt, map_location="cpu", weights_only=False)
    model  = CVAE_Raman(
        input_length=int(state.get("input_length", N_RAMAN)),
        latent_dim  =int(state.get("latent_dim",   64)),
    ).to(DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_beta_vae_model(ckpt: str) -> BetaVAE_Raman:
    state  = torch.load(ckpt, map_location="cpu", weights_only=False)
    model  = BetaVAE_Raman(
        input_length=int(state.get("input_length", N_RAMAN)),
        latent_dim  =int(state.get("latent_dim",   64)),
    ).to(DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5.  UMAP fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_umap(
    latents:    np.ndarray,
    n_neighbors: int,
    min_dist:    float,
    seed:        int,
) -> np.ndarray:
    """
    Fit UMAP(n_components=3) on ``latents`` (N, D) and return (N, 3).
    """
    reducer = umap.UMAP(
        n_components  = 3,
        n_neighbors   = n_neighbors,
        min_dist      = min_dist,
        metric        = "euclidean",
        random_state  = seed,
        low_memory    = False,
        verbose       = False,
    )
    return reducer.fit_transform(latents.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Publication-quality 3D plot
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(
    umap_cdae:     np.ndarray,   # (N, 3)
    umap_cvae:     np.ndarray,
    umap_beta_vae: np.ndarray,
    color_time:    np.ndarray,   # (N,)  fermentation time (h)
    color_pen:     np.ndarray,   # (N,)  penicillin conc  (g/L)
    output_dir:    str,
) -> None:
    """
    3×2 figure:
        rows  = CDAE | CVAE | β-VAE
        cols  = coloured by time (viridis) | coloured by penicillin (plasma)
    """
    mpl.rcParams.update({
        "font.family":     "sans-serif",
        "font.size":       10,
        "axes.titlesize":  11,
        "axes.labelsize":  9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })

    fig = plt.figure(figsize=(16, 18), constrained_layout=False)
    fig.subplots_adjust(
        left=0.04, right=0.90,
        top=0.95,  bottom=0.03,
        hspace=0.12, wspace=0.05,
    )

    models = [
        (umap_cdae,     "CDAE\nConvolutional Denoising AE"),
        (umap_cvae,     "CVAE\nVariational AE  (\u03b2 = 1)"),
        (umap_beta_vae, r"$\beta$-VAE" + "\nDisentangled VAE  (\u03b2 = 0.005)"),
    ]
    col_specs = [
        (color_time, "viridis", "Fermentation time (h)",          "time"),
        (color_pen,  "plasma",  "Penicillin concentration (g/L)", "penicillin"),
    ]

    point_size = max(0.8, 20_000 / len(color_time))   # shrink dots for large N

    # Store axes and scatter objects to build shared colorbars
    scatter_by_col: list[list] = [[], []]   # [col_idx][row_idx] → scatter

    for row_idx, (emb, row_title) in enumerate(models):
        for col_idx, (c_vals, cmap_name, cb_label, _) in enumerate(col_specs):
            # Subplot index: 3 rows × 2 cols
            ax = fig.add_subplot(3, 2, row_idx * 2 + col_idx + 1, projection="3d")

            # Sort by color value so high-value points render on top
            sort_order = np.argsort(c_vals)
            sc = ax.scatter(
                emb[sort_order, 0],
                emb[sort_order, 1],
                emb[sort_order, 2],
                c         = c_vals[sort_order],
                cmap      = cmap_name,
                s         = point_size,
                alpha     = 0.65,
                linewidths= 0,
                rasterized= True,
            )
            scatter_by_col[col_idx].append(sc)

            # Axis labels only on edge subplots to reduce clutter
            ax.set_xlabel("UMAP-1", fontsize=7, labelpad=1)
            ax.set_ylabel("UMAP-2", fontsize=7, labelpad=1)
            ax.set_zlabel("UMAP-3", fontsize=7, labelpad=1)
            ax.tick_params(labelsize=6, pad=0)

            # Row title on the left column
            if col_idx == 0:
                ax.set_title(row_title, fontsize=11, fontweight="bold",
                             pad=6, loc="left")

            # Viewing angle
            ax.view_init(elev=22, azim=-55)

            # Light pane colour for readability
            pane_col = (0.96, 0.96, 0.96, 0.5)
            ax.xaxis.pane.fill = True;  ax.xaxis.pane.set_facecolor(pane_col)
            ax.yaxis.pane.fill = True;  ax.yaxis.pane.set_facecolor(pane_col)
            ax.zaxis.pane.fill = True;  ax.zaxis.pane.set_facecolor(pane_col)
            ax.grid(True, linewidth=0.3, color="white")

    # ── Column header text ────────────────────────────────────────────────────
    fig.text(0.25, 0.975, "Coloured by fermentation time",
             ha="center", va="top", fontsize=12, fontweight="bold")
    fig.text(0.72, 0.975, "Coloured by penicillin concentration",
             ha="center", va="top", fontsize=12, fontweight="bold")

    # ── Shared colorbars ──────────────────────────────────────────────────────
    # One colorbar per column, placed to the right of each column
    cb_ax_time = fig.add_axes([0.44, 0.10, 0.012, 0.75])  # right of col-1
    cb_ax_pen  = fig.add_axes([0.91, 0.10, 0.012, 0.75])  # right of col-2

    for cb_ax, sc_list, cmap_name, cb_label in [
        (cb_ax_time, scatter_by_col[0], col_specs[0][1], col_specs[0][2]),
        (cb_ax_pen,  scatter_by_col[1], col_specs[1][1], col_specs[1][2]),
    ]:
        # Normalise across all three model embeddings
        all_c = np.concatenate([sc.get_array() for sc in sc_list])
        vmin, vmax = float(np.nanpercentile(all_c, 1)), float(np.nanpercentile(all_c, 99))
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm   = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
        sm.set_array([])
        cb   = fig.colorbar(sm, cax=cb_ax)
        cb.set_label(cb_label, fontsize=10, labelpad=8)
        cb.ax.tick_params(labelsize=8)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "latent_umap.png"
    pdf_path = out_dir / "latent_umap.pdf"

    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)
    print(f"\n[7] Figures saved:")
    print(f"    {png_path}")
    print(f"    {pdf_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    print(f"Device: {DEVICE}")
    print(f"n_samples={args.n_samples}  umap_neighbors={args.umap_neighbors}  "
          f"min_dist={args.umap_min_dist}  seed={args.seed}")

    # ── 1. Load & sample data ─────────────────────────────────────────────────
    X_raw, t_vals, pen_vals, batch_ids = load_and_sample(
        args.csv, args.n_samples, args.seed,
    )
    N = len(X_raw)

    # ── 2. Load scalers ───────────────────────────────────────────────────────
    print("\n[2] Loading scalers …")
    cdae_scaler     = joblib.load(CDAE_SCALER)
    cvae_scaler     = joblib.load(CVAE_SCALER)
    beta_vae_scaler = joblib.load(BETA_VAE_SCALER)

    # ── 3. Preprocess (each model has its own scaler) ─────────────────────────
    print("\n[3] Preprocessing …")
    X_cdae     = preprocess(X_raw, cdae_scaler)
    X_cvae     = preprocess(X_raw, cvae_scaler)
    X_beta_vae = preprocess(X_raw, beta_vae_scaler)

    # ── 4. Load models ────────────────────────────────────────────────────────
    print("\n[4] Loading models …")
    cdae     = load_cdae_model(CDAE_CKPT)
    cvae     = load_cvae_model(CVAE_CKPT)
    beta_vae = load_beta_vae_model(BETA_VAE_CKPT)

    # ── 5. Encode ─────────────────────────────────────────────────────────────
    print(f"\n[5] Encoding {N:,} spectra through each model …")
    print("    CDAE …", end=" ", flush=True)
    Z_cdae     = encode_cdae(cdae,     X_cdae,     args.encode_batch)
    print(f"done  {Z_cdae.shape}")

    print("    CVAE …", end=" ", flush=True)
    Z_cvae     = encode_vae(cvae,      X_cvae,     args.encode_batch)
    print(f"done  {Z_cvae.shape}")

    print("    β-VAE …", end=" ", flush=True)
    Z_beta_vae = encode_vae(beta_vae,  X_beta_vae, args.encode_batch)
    print(f"done  {Z_beta_vae.shape}")

    # Free GPU memory before UMAP (runs on CPU)
    del cdae, cvae, beta_vae
    torch.cuda.empty_cache()

    # ── 6. UMAP 3-D ──────────────────────────────────────────────────────────
    print(f"\n[6] Fitting UMAP (n_neighbors={args.umap_neighbors}, "
          f"min_dist={args.umap_min_dist}) …")

    print("    CDAE …", end=" ", flush=True)
    emb_cdae = fit_umap(Z_cdae,     args.umap_neighbors, args.umap_min_dist, args.seed)
    print("done")

    print("    CVAE …", end=" ", flush=True)
    emb_cvae = fit_umap(Z_cvae,     args.umap_neighbors, args.umap_min_dist, args.seed)
    print("done")

    print("    β-VAE …", end=" ", flush=True)
    emb_bvae = fit_umap(Z_beta_vae, args.umap_neighbors, args.umap_min_dist, args.seed)
    print("done")

    # ── 7. Figure ─────────────────────────────────────────────────────────────
    print("\n[7] Generating figure …")
    make_figure(
        umap_cdae     = emb_cdae,
        umap_cvae     = emb_cvae,
        umap_beta_vae = emb_bvae,
        color_time    = t_vals,
        color_pen     = pen_vals,
        output_dir    = args.output_dir,
    )

    # ── 8. Save raw arrays ────────────────────────────────────────────────────
    out_dir  = Path(args.output_dir)
    npz_path = out_dir / "latent_umap.npz"
    np.savez(
        str(npz_path),
        umap_cdae     = emb_cdae,
        umap_cvae     = emb_cvae,
        umap_beta_vae = emb_bvae,
        latent_cdae   = Z_cdae,
        latent_cvae   = Z_cvae,
        latent_bvae   = Z_beta_vae,
        time          = t_vals,
        penicillin    = pen_vals,
        batch_id      = batch_ids,
    )
    print(f"    {npz_path}  (numpy archive)")

    print("\nDone.")


if __name__ == "__main__":
    main()
