#!/usr/bin/env python
"""
latent_correlation.py
=====================
Visualise how well each latent dimension of CDAE, CVAE, and β-VAE
correlates with five physical ground-truth variables from IndPenSim V3:

    • Penicillin  (Penicillin concentration, online)
    • Glucose     (Substrate concentration, online)
    • Biomass     (Offline Biomass, sparse → non-zero rows only)
    • pH
    • DO          (Dissolved oxygen)

Preprocessing pipeline (matches training exactly):
    raw Raman (2001 channels)
        → clamp NaN / negatives → 0
        → Savitzky-Golay 1st-derivative  (window=15, poly=2, deriv=1)
        → StandardScaler.transform()     (model-specific scaler, fitted on training set)
        → encoder (frozen)               → latents (N, 64)

NOTE: The preprocessing order is SG → StandardScaler, NOT the reverse.

For each model the script:
    1. Encodes all fault-free batch rows through the frozen encoder.
    2. Computes the absolute Pearson |r| between each of the 64 latent
        dimensions and each of the 5 physical variables.
    3. Filters to "active" dimensions (max |r| across variables > threshold).
    4. Plots a 3-panel annotated heatmap (CDAE | CVAE | β-VAE).

All three encoders are trained on the full 90 fault-free batches
(random 80/20 split, seed=42) — no regime-ablation models are used.

Usage
-----
    PYTHONPATH=. python scripts/latent_correlation.py
    PYTHONPATH=. python scripts/latent_correlation.py --threshold 0.3 --batches all
    PYTHONPATH=. python scripts/latent_correlation.py --output-dir outputs/latent_correlation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from scipy.stats import pearsonr

# ── project root on sys.path ─────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.autoencoder.model import CDAE_Raman, CVAE_Raman, BetaVAE_Raman
from src.data.dataset import RAMAN_START_COL_IDX, FAULT_COL

# ── Constants ─────────────────────────────────────────────────────────────────
N_RAMAN       = 2001
SG_WINDOW     = 15
SG_POLY       = 2
SG_DERIV      = 1
ENCODE_BATCH  = 1024
DEFAULT_THR   = 0.4          # |r| threshold to classify a dimension as "active"
CSV_PATH      = "./100_Batches_IndPenSim_V3.csv"
OUTPUT_DIR    = Path("./outputs/latent_correlation")

# ── Physical variable column names (exact CSV headers) ───────────────────────
PHYS_COLS = {
    "Penicillin":   "Penicillin concentration(P:g/L)",
    "Glucose":      "Substrate concentration(S:g/L)",
    "Biomass":      "Offline Biomass concentratio(X_offline:X(g L^{-1}))",
    "Temperature":  "Temperature(T:K)",
    "PAA":          "PAA concentration offline(PAA_offline:PAA (g L^{-1}))",
    "NH3":          "NH_3 concentration off-line(NH3_offline:NH3 (g L^{-1}))",
    "Viscosity":    "Viscosity(Viscosity_offline:centPoise)",
}

# Offline (sparse) measurements: rows where value == 0 mean "not measured".
# Correlations for these variables are computed on non-zero rows only.
SPARSE_VARS = {"Biomass", "PAA", "NH3", "Viscosity"}

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "CDAE": {
        "cls":        CDAE_Raman,
        "ckpt":       "/home/gabriel/ReaKt/outputs/seed_sweep/cdae_pi_lstm/p2/encoders/cdae_s42.pt",
        "scaler":     "/home/gabriel/ReaKt/outputs/seed_sweep/cdae_pi_lstm/p2/encoders/cdae_scaler_s42.joblib",
        "encode_fn":  "direct",        # encode() returns z directly
        "label":      "CDAE",
        "color":      "#e07b39",       # orange accent for title bar
    },
    "CVAE": {
        "cls":        CVAE_Raman,
        "ckpt":       "/home/gabriel/ReaKt/outputs/seed_sweep/cvae_pi_lstm/p2/encoders/cvae_s42.pt",
        "scaler":     "/home/gabriel/ReaKt/outputs/seed_sweep/cvae_pi_lstm/p2/encoders/cvae_scaler_s42.joblib",
        "encode_fn":  "vae",           # encode() returns (mu, log_var)
        "label":      "CVAE",
        "color":      "#3a7ebf",
    },
    "BetaVAE": {
        "cls":        BetaVAE_Raman,
        "ckpt":       "/home/gabriel/ReaKt/outputs/seed_sweep/beta_vae_pi_lstm/p2/encoders/beta_vae_s42.pt",
        "scaler":     "/home/gabriel/ReaKt/outputs/seed_sweep/beta_vae_pi_lstm/p2/encoders/beta_vae_scaler_s42.joblib",
        "encode_fn":  "vae",
        "label":      "β-VAE",
        "color":      "#2ca02c",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(csv_path: str) -> pd.DataFrame:
    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def get_fault_free_mask(df: pd.DataFrame) -> np.ndarray:
    """Boolean mask: True for rows that belong to fault-free batches."""
    time_col = df["Time (h)"].values
    fault_col = df[FAULT_COL].values if FAULT_COL in df.columns else np.zeros(len(df))

    # Detect batch boundaries (time resets)
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if time_col[i] < time_col[i - 1]:
            bid += 1
        batch_ids[i] = bid

    n_batches = int(bid + 1)
    print(f"  Detected {n_batches} batches")

    # Mark entire batch as faulty if any row has fault > 0
    faulty_batch_ids = set()
    for b in range(n_batches):
        mask = batch_ids == b
        if fault_col[mask].max() > 0:
            faulty_batch_ids.add(b)

    fault_free_mask = np.array([batch_ids[i] not in faulty_batch_ids
                                 for i in range(len(df))])
    n_clean = int(fault_free_mask.sum())
    print(f"  Fault-free rows: {n_clean:,} / {len(df):,}  "
          f"({len(faulty_batch_ids)} faulty batches excluded)")
    return fault_free_mask


def extract_raman(df: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """Return (N, 2001) raw Raman array for selected rows."""
    all_cols   = df.columns.tolist()
    raman_cols = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + N_RAMAN]
    X_raw = df.loc[mask, raman_cols].values.astype(np.float32)
    print(f"  Raman block: {X_raw.shape}  (cols {RAMAN_START_COL_IDX}–"
          f"{RAMAN_START_COL_IDX + N_RAMAN - 1}  |  "
          f"{raman_cols[0]} → {raman_cols[-1]})")
    return X_raw


def extract_physical(df: pd.DataFrame, mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Return dict of physical variable arrays (float64), NaN where not available."""
    sub = df.loc[mask]
    out: Dict[str, np.ndarray] = {}
    for name, col in PHYS_COLS.items():
        if col not in df.columns:
            print(f"  [WARN] column not found: {col!r} — skipping {name}")
            continue
        vals = sub[col].values.astype(np.float64)
        # Offline (sparse) variables: treat 0 as NaN (not measured)
        if name in SPARSE_VARS:
            vals[vals <= 0] = np.nan
        n_valid = int(np.isfinite(vals).sum())
        print(f"  {name:12s}: {n_valid:,} valid rows  (col: {col!r})")
        out[name] = vals
    return out


def preprocess_raman(X_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the training preprocessing pipeline and return (valid_mask, X_sg_raw).

    Pipeline
    --------
    1. valid_mask : rows with non-zero Raman energy (instrument running)
    2. clamp NaN / negatives → 0
    3. Savitzky-Golay 1st-derivative  (window=15, poly=2, deriv=1)

    The StandardScaler step is model-specific and applied later.

    Returns
    -------
    valid_mask : (N,) bool — rows to encode
    X_sg       : (N_valid, 2001) float32 — SG-filtered spectra (not yet scaled)
    """
    energy     = np.abs(X_raw).sum(axis=1)
    valid_mask = energy > 0
    X_valid    = X_raw[valid_mask].copy()

    # Clamp
    X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)
    X_valid = np.clip(X_valid, 0.0, None)

    # SG first-derivative
    X_sg = savgol_filter(
        X_valid.astype(np.float64),
        window_length=SG_WINDOW,
        polyorder=SG_POLY,
        deriv=SG_DERIV,
        axis=1,
    ).astype(np.float32)

    print(f"  SG-filtered: {X_sg.shape}  "
          f"({valid_mask.sum():,} valid / {len(valid_mask):,} total rows)")
    return valid_mask, X_sg


# ─────────────────────────────────────────────────────────────────────────────
# Encoder helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_encoder(cfg: dict, device: torch.device):
    """Load a frozen encoder from checkpoint."""
    state        = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
    input_length = int(state.get("input_length", N_RAMAN))
    latent_dim   = int(state.get("latent_dim",   64))

    if cfg["cls"] is CDAE_Raman:
        model = cfg["cls"](input_length=input_length, latent_dim=latent_dim, noise_std=0.0)
    else:
        model = cfg["cls"](input_length=input_length, latent_dim=latent_dim)

    model.load_state_dict(state["model_state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval().to(device)
    return model, latent_dim


@torch.no_grad()
def encode(model, X_scaled: np.ndarray, encode_fn: str,
           device: torch.device, latent_dim: int) -> np.ndarray:
    """Encode scaled spectra → latents (N, latent_dim)."""
    parts = []
    for start in range(0, len(X_scaled), ENCODE_BATCH):
        chunk = torch.tensor(
            X_scaled[start : start + ENCODE_BATCH],
            dtype=torch.float32, device=device,
        )
        if encode_fn == "direct":
            z = model.encode(chunk)
        else:  # vae: returns (mu, log_var); use mu
            z, _ = model.encode(chunk)
        parts.append(z.cpu().numpy())
    return np.concatenate(parts, axis=0)   # (N, latent_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Correlation computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_correlations(
    Z: np.ndarray,                   # (N, 64) latents for valid rows
    phys: Dict[str, np.ndarray],     # physical vars (N_total rows, may have NaN)
    valid_mask: np.ndarray,          # (N_total,) bool — which rows were encoded
) -> Dict[str, np.ndarray]:
    """
    Compute |Pearson r| between each latent dim and each physical variable.

    Returns
    -------
    corr : dict  variable_name → (latent_dim,) float64  absolute correlations
    """
    n_latent = Z.shape[1]
    corr: Dict[str, np.ndarray] = {}

    for var, vals_full in phys.items():
        # Align physical variable to encoded rows
        vals_valid = vals_full[valid_mask]

        # For sparse variables, further restrict to rows where measurement exists
        finite_mask = np.isfinite(vals_valid)
        n_finite    = finite_mask.sum()

        if n_finite < 30:
            print(f"  [WARN] {var}: only {n_finite} valid rows — skipping correlation")
            corr[var] = np.zeros(n_latent)
            continue

        Z_sub  = Z[finite_mask]
        y      = vals_valid[finite_mask]
        rs     = np.array([abs(pearsonr(Z_sub[:, j], y)[0]) for j in range(n_latent)])
        corr[var] = rs

        top_idx = int(np.argmax(rs))
        print(f"  {var:12s}: max |r|={rs[top_idx]:.3f} at z{top_idx+1:02d}  "
              f"(n={n_finite:,})")

    return corr


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _per_model_active_dims(
    corr:    Dict[str, np.ndarray],
    top_n:   int,
) -> List[int]:
    """
    For one model, return the indices of the top-N latent dimensions ranked
    by their maximum |r| across all physical variables, sorted descending.
    """
    if not corr:
        return []
    mat         = np.stack(list(corr.values()), axis=0)   # (n_vars, 64)
    max_per_dim = mat.max(axis=0)                         # (64,)
    ranked      = np.argsort(max_per_dim)[::-1]           # descending
    return ranked[:top_n].tolist()


def _active_variables(
    all_corr:   Dict[str, Dict[str, np.ndarray]],
    min_var_r:  float,
) -> List[str]:
    """
    Return variable names whose max |r| (across any model × any latent dim)
    exceeds min_var_r.  This removes variables like pH/DO that are never
    well-captured by any encoder.
    """
    active = []
    for var in PHYS_COLS.keys():
        best = 0.0
        for corr in all_corr.values():
            if var in corr:
                best = max(best, float(corr[var].max()))
        if best >= min_var_r:
            active.append(var)
    return active


def plot_heatmaps(
    all_corr:   Dict[str, Dict[str, np.ndarray]],
    model_cfgs: Dict[str, dict],
    threshold:  float,        # kept for backward-compat; controls min_var_r
    output_dir: Path,
    top_n:      int   = 12,   # top-N dims per model panel
    min_var_r:  float = 0.3,  # variables below this max-r are hidden
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_keys = list(model_cfgs.keys())
    var_names  = _active_variables(all_corr, min_var_r)

    if not var_names:
        print(f"  [plot] No variables exceed min_var_r={min_var_r}. "
              f"Try lowering --threshold.")
        return

    n_vars   = len(var_names)
    n_models = len(model_keys)

    # Each panel has its own top-N dims (per-model, sorted by max |r|)
    panel_dims: Dict[str, List[int]] = {
        key: _per_model_active_dims(all_corr[key], top_n)
        for key in model_keys
    }

    # Fixed cell size for readability (Adjusted for vertical layout)
    CELL_W   = 0.70   # inches per latent-dim column
    CELL_H   = 0.85   # inches per variable row
    CBAR_W   = 0.80   # colorbar & right margin
    LEFT_M   = 1.50   # left margin for y-axis labels
    TOP_M    = 1.00   # top margin for suptitle
    BOTTOM_M = 1.00   # bottom margin for x-labels
    GAP      = 1.00   # vertical gap between panels

    panel_w = top_n * CELL_W
    panel_h = n_vars * CELL_H
    
    fig_w   = LEFT_M + panel_w + CBAR_W
    fig_h   = TOP_M + n_models * panel_h + (n_models - 1) * GAP + BOTTOM_M

    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        10,
        "axes.titlesize":   12,
        "axes.titleweight": "bold",
        "axes.labelsize":   10,
        "xtick.labelsize":  9,
        "ytick.labelsize":  10,
    })

    # Changed from 1xN to Nx1 for vertical layout
    fig, axes = plt.subplots(
        n_models, 1,
        figsize=(fig_w, fig_h),
        gridspec_kw={"hspace": GAP / panel_h},
    )
    if n_models == 1:
        axes = [axes]

    im_ref = None   # keep last imshow for the shared colourbar

    for ax, key in zip(axes, model_keys):
        cfg      = model_cfgs[key]
        corr     = all_corr[key]
        dims     = panel_dims[key]
        n_active = len(dims)

        # Build matrix: rows = variables, cols = top-N latent dims
        mat = np.zeros((n_vars, n_active))
        for vi, var in enumerate(var_names):
            if var in corr:
                mat[vi, :] = corr[var][dims]

        im_ref = ax.imshow(mat, cmap="Reds", vmin=0.0, vmax=1.0,
                           aspect="auto", interpolation="nearest")

        # ── Cell annotations ──────────────────────────────────────────────
        for vi in range(n_vars):
            for di in range(n_active):
                val   = mat[vi, di]
                color = "white" if val > 0.60 else ("#333333" if val > 0.15 else "#aaaaaa")
                ax.text(di, vi, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=8.5, color=color, fontweight="bold")

        # ── X-axis: latent dimension labels ───────────────────────────────
        ax.set_xticks(range(n_active))
        ax.set_xticklabels(
            [f"z{dims[i]+1}" for i in range(n_active)],
            rotation=45, ha="right", fontsize=9,
        )
        ax.set_xlabel("Latent dimension  (sorted by max |r|)", fontsize=9,
                      labelpad=4)

        # ── Y-axis: variable names on EVERY panel for clarity ─────────────
        ax.set_yticks(range(n_vars))
        ax.set_yticklabels(var_names, fontsize=10)
        ax.tick_params(axis="y", pad=4)

        # ── Title bar ─────────────────────────────────────────────────────
        ax.set_title(
            cfg["label"],
            color="white",
            fontsize=12,
            fontweight="bold",
            bbox=dict(facecolor=cfg["color"], edgecolor="none",
                      boxstyle="round,pad=0.35"),
            pad=10,
        )

        # ── Grid lines between cells ──────────────────────────────────────
        ax.set_xticks(np.arange(-0.5, n_active, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_vars,   1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

        # ── Spine styling ─────────────────────────────────────────────────
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")

    # ── Shared colourbar ──────────────────────────────────────────────────────
    fig.subplots_adjust(
        left   = LEFT_M / fig_w,
        right  = 1.0 - CBAR_W / fig_w,
        top    = 1.0 - TOP_M / fig_h,
        bottom = BOTTOM_M / fig_h,
    )
    
    # Vertically stretch the colorbar relative to the full figure bounds
    cax_width = 0.15 / fig_w
    cax_left = 1.0 - (CBAR_W * 0.7) / fig_w
    cax  = fig.add_axes([cax_left, BOTTOM_M / fig_h, cax_width, 1.0 - (TOP_M + BOTTOM_M) / fig_h])
    cbar = fig.colorbar(im_ref, cax=cax)
    cbar.set_label("|Pearson r|", fontsize=10, labelpad=8)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # ── Super-title (above all title bars) ────────────────────────────────────
    fig.suptitle(
        f"Latent–Physical Correlation Heatmap\n"
        f"Top {top_n} latent dims per model  (sorted by max |r|)  "
        f"·  Variables shown: max |r| > {min_var_r}",
        fontsize=11,
        y=1.0 - (TOP_M * 0.3) / fig_h,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = output_dir / f"latent_correlation.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out}")
    plt.close(fig)


def save_csv(
    all_corr:   Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    """Save full (64 × 5) correlation matrices per model to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    var_names = list(PHYS_COLS.keys())
    for key, corr in all_corr.items():
        rows = []
        for d in range(64):
            row = {"latent_dim": f"z{d+1}"}
            for var in var_names:
                row[var] = corr.get(var, np.zeros(64))[d]
            rows.append(row)
        out = output_dir / f"correlations_{key.lower()}.csv"
        pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4f")
        print(f"  CSV saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Latent–physical correlation heatmap for CDAE, CVAE, β-VAE"
    )
    parser.add_argument("--csv",        default=CSV_PATH)
    parser.add_argument("--threshold",  type=float, default=DEFAULT_THR,
                        help="Min |r| to show a variable row (default 0.4)")
    parser.add_argument("--top-n",      type=int,   default=12,
                        help="Top-N latent dims to show per model panel, "
                             "sorted by max |r| descending (default 12)")
    parser.add_argument("--min-var-r",  type=float, default=0.3,
                        help="Hide variable rows whose best |r| across all "
                             "models is below this value (default 0.3)")
    parser.add_argument("--batches",    default="fault-free",
                        choices=["fault-free", "all"],
                        help="Which batches to encode (default: fault-free only)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--no-csv",     action="store_true",
                        help="Skip saving per-model CSV files")
    parser.add_argument("--beta-vae-ckpt", default=None,
                        help="Override the β-VAE encoder checkpoint "
                             "(e.g. a per-seed encoder from the benchmark sweep)")
    parser.add_argument("--beta-vae-scaler", default=None,
                        help="Override the β-VAE StandardScaler for --beta-vae-ckpt")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=list(MODEL_CONFIGS),
                        help=f"Subset of models to encode (default: all -> {list(MODEL_CONFIGS)})")
    args = parser.parse_args()

    # Apply β-VAE checkpoint overrides before anything is loaded.
    if args.beta_vae_ckpt:
        MODEL_CONFIGS["BetaVAE"]["ckpt"] = args.beta_vae_ckpt
        if args.beta_vae_scaler:
            MODEL_CONFIGS["BetaVAE"]["scaler"] = args.beta_vae_scaler
        print(f"\n[override] β-VAE ckpt   : {MODEL_CONFIGS['BetaVAE']['ckpt']}")
        print(f"[override] β-VAE scaler : {MODEL_CONFIGS['BetaVAE']['scaler']}")
    elif args.beta_vae_scaler:
        parser.error("--beta-vae-scaler requires --beta-vae-ckpt")

    selected = args.models or list(MODEL_CONFIGS)
    active_configs = {k: v for k, v in MODEL_CONFIGS.items() if k in selected}

    device     = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    threshold  = args.threshold

    print(f"\nDevice : {device}")
    print(f"Top-N dims per panel : {args.top_n}")
    print(f"Min variable |r|     : {args.min_var_r}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df = load_csv(args.csv)

    if args.batches == "fault-free":
        row_mask = get_fault_free_mask(df)
    else:
        row_mask = np.ones(len(df), dtype=bool)
        print(f"  Using all {row_mask.sum():,} rows (fault-free + faulty)")

    print(f"\nExtracting physical variables ...")
    phys = extract_physical(df, row_mask)

    print(f"\nExtracting Raman spectra ...")
    X_raw = extract_raman(df, row_mask)

    # ── 2. Common SG preprocessing (before per-model scaling) ─────────────────
    print(f"\nApplying SG filter (window={SG_WINDOW}, poly={SG_POLY}, deriv={SG_DERIV}) ...")
    valid_mask, X_sg = preprocess_raman(X_raw)

    # ── 3. Encode with each model ─────────────────────────────────────────────
    all_corr: Dict[str, Dict[str, np.ndarray]] = {}

    for key, cfg in active_configs.items():
        print(f"\n{'─'*55}")
        print(f"  Model: {cfg['label']}  |  ckpt: {Path(cfg['ckpt']).name}")
        print(f"{'─'*55}")

        # Load scaler and apply (SG is already done; scaler is the second step)
        scaler   = joblib.load(cfg["scaler"])
        X_scaled = scaler.transform(X_sg).astype(np.float32)

        # Load frozen encoder
        model, latent_dim = load_encoder(cfg, device)
        print(f"  Encoder loaded — latent_dim={latent_dim}")

        # Encode
        Z = encode(model, X_scaled, cfg["encode_fn"], device, latent_dim)
        print(f"  Latents: {Z.shape}  mean={Z.mean():.4f}  std={Z.std():.4f}")

        # Correlations
        print(f"\n  Computing |Pearson r| ...")
        all_corr[key] = compute_correlations(Z, phys, valid_mask)

    # ── 4. Plot ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  Plotting heatmaps  (top_n={args.top_n}, min_var_r={args.min_var_r}) ...")
    plot_heatmaps(all_corr, active_configs, threshold, output_dir,
                  top_n=args.top_n, min_var_r=args.min_var_r)

    # ── 5. Save CSVs ──────────────────────────────────────────────────────────
    if not args.no_csv:
        print(f"\n  Saving correlation CSVs ...")
        save_csv(all_corr, output_dir)

    print(f"\nDone.  Outputs in {output_dir}/")


if __name__ == "__main__":
    main()