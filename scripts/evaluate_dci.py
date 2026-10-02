"""
scripts/evaluate_dci.py
========================
Disentanglement–Completeness–Informativeness (DCI) score for CDAE, CVAE,
and β-VAE Raman encoders on the IndPenSim V3 bioreactor dataset.

Background
----------
DCI (Eastwood & Williams, 2018 "A Framework for the Quantitative Evaluation
of Disentangled Representations") uses the feature-importance matrix of a
GradientBoosting predictor trained to recover each physical factor from the
full latent representation.

  Importance matrix  R ∈ ℝ^{K × D}
    K = number of physical factors (ground-truth generative factors)
    D = latent dimension (64 for all models here)
    R[k, d] = importance of latent dim d for predicting factor k

Three aggregate scalars (all ∈ [0, 1], higher = more disentangled):

  Disentanglement  D_d = 1 − H_K(R_col / Σ_k R_col)   per dim,
                   then weighted-average over dims by total column importance.
                   → "each latent dim is used by at most one factor"

  Completeness     C_k = 1 − H_D(R_row / Σ_d R_row)   per factor,
                   then averaged over factors.
                   → "each factor is captured by at most one latent dim"

  Informativeness  I_k = R²(f_k, test split)            per factor,
                   then averaged over factors.
                   → "the latent space actually contains information about V"

where H_p(·) is the normalised Shannon entropy with log(p).

Physical factors used (11, same as evaluate_mig.py):
    Penicillin, Glucose, DO2, pH, Temperature, Vessel Vol,
    Aeration, Agitator RPM, CO2 offgas, O2 offgas, Batch Time (norm)

Usage
-----
    PYTHONPATH=. python scripts/evaluate_dci.py [--csv PATH] [--n-samples N]

Output
------
    outputs/dci/
        dci_importance_<model>.png    : heatmap of R[K × top-16 dims]
        dci_per_factor_<model>.png    : C_k bar + I_k bar per factor
        dci_comparison.png            : grouped bar of D / C / I across models
        dci_results.csv               : numeric table for all models & factors
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

# ── Project root ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset import RAMAN_START_COL_IDX

# ── ───────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ── ───────────────────────────────────────────────────────────────────────────

CSV_PATH   = "./100_Batches_IndPenSim_V3.csv"
OUTPUT_DIR = Path("./outputs/dci")

# Fault-free batch IDs (0-indexed): R1 + R2 + R3 = 0..89
FAULT_FREE_IDS = set(range(90))

# Same 10 physical factors as evaluate_mig.py
# Same physical factors as evaluate_mig.py (Updated with PAA, NH3, Viscosity)
PHYSICAL_FACTORS: Dict[str, str] = {
    "Penicillin (g/L)":    "Penicillin concentration(P:g/L)",
    "Glucose (g/L)":       "Substrate concentration(S:g/L)",
    "DO2 (mg/L)":          "Dissolved oxygen concentration(DO2:mg/L)",
    "pH":                  "pH(pH:pH)",
    "Temperature (K)":     "Temperature(T:K)",
    "Vessel Vol (L)":      "Vessel Volume(V:L)",
    "Aeration (L/h)":      "Aeration rate(Fg:L/h)",
    "Agitator (RPM)":      "Agitator RPM(RPM:RPM)",
    "CO2 offgas (%)":      "carbon dioxide percent in off-gas(CO2outgas:%)",
    "O2 offgas (%)":       "Oxygen in percent in off-gas(O2:O2  (%))",
    "Batch Time (norm)":   "__TIME_NORM__",
    "PAA (g/L)":           "PAA concentration offline(PAA_offline:PAA (g L^{-1}))",
    "NH3 (g/L)":           "NH_3 concentration off-line(NH3_offline:NH3 (g L^{-1}))",
    "Viscosity (cP)":      "Viscosity(Viscosity_offline:centPoise)",
}

# Encoder definitions (Updated with your exact Seed 42 paths)
ENCODERS: Dict[str, Dict] = {
    "CDAE": {
        "class":       "CDAERamanEncoderV2",
        "module":      "src.data.cdae_encoder",
        "ckpt":        "/home/gabriel/ReaKt/outputs/seed_sweep/cdae_pi_lstm/p2/encoders/cdae_s42.pt",
        "scaler":      "/home/gabriel/ReaKt/outputs/seed_sweep/cdae_pi_lstm/p2/encoders/cdae_scaler_s42.joblib",
        "latent_dim":  64,
    },
    "CVAE": {
        "class":       "CVAERamanEncoderV2",
        "module":      "src.data.cvae_encoder",
        "ckpt":        "/home/gabriel/ReaKt/outputs/seed_sweep/cvae_pi_lstm/p2/encoders/cvae_s42.pt",
        "scaler":      "/home/gabriel/ReaKt/outputs/seed_sweep/cvae_pi_lstm/p2/encoders/cvae_scaler_s42.joblib",
        "latent_dim":  64,
    },
    "Beta-VAE": {
        "class":       "BetaVAERamanEncoderV2",
        "module":      "src.data.beta_vae_encoder",
        "ckpt":        "/home/gabriel/ReaKt/outputs/seed_sweep/beta_vae_pi_lstm/p1/encoders/beta_vae_s42.pt",
        "scaler":      "/home/gabriel/ReaKt/outputs/seed_sweep/beta_vae_pi_lstm/p1/encoders/beta_vae_scaler_s42.joblib",
        "latent_dim":  64,
    },
}

# DCI hyper-parameters
N_SAMPLES     = 20_000   # rows subsampled from fault-free batches (same as MIG)
GBR_ESTIMATORS = 200     # GradientBoosting trees per factor
GBR_DEPTH      = 3       # max tree depth
GBR_MIN_LEAF   = 5
GBR_SUBSAMPLE  = 0.8     # stochastic GB subsample fraction
TEST_FRAC      = 0.20    # fraction held out for Informativeness R²
RANDOM_STATE   = 42


# ── ───────────────────────────────────────────────────────────────────────────
# DATA TYPES
# ── ───────────────────────────────────────────────────────────────────────────

class DCIResult(NamedTuple):
    D:               float           # overall Disentanglement score
    C:               float           # overall Completeness score
    I:               float           # overall Informativeness score
    D_per_dim:       np.ndarray      # (D,) per-latent-dim disentanglement
    C_per_factor:    np.ndarray      # (K,) per-factor completeness
    I_per_factor:    np.ndarray      # (K,) per-factor informativeness (R²)
    R:               np.ndarray      # (K, D) raw importance matrix
    factor_names:    List[str]


# ── ───────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ── ───────────────────────────────────────────────────────────────────────────

def load_data(
    csv_path:  str = CSV_PATH,
    n_samples: int = N_SAMPLES,
    seed:      int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load IndPenSim V3, keep fault-free rows, subsample, return (df_sub, V).

    V : (n_samples, K) float32 — physical factor matrix
    """
    print(f"\n[DATA]  Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"[DATA]  Raw shape: {df.shape[0]:,} × {df.shape[1]}")

    # Detect batch boundaries by time reset
    t_vals    = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid
    df["_batch_id"] = batch_ids

    # Keep fault-free rows with non-zero Raman spectra
    raman_cols   = df.columns[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + 2001]
    raman_energy = df[raman_cols].values.sum(axis=1)
    mask_ff      = df["_batch_id"].isin(FAULT_FREE_IDS)
    mask_raman   = raman_energy > 0
    df_ff = df[mask_ff & mask_raman].copy().reset_index(drop=True)
    print(f"[DATA]  Fault-free rows with Raman: {len(df_ff):,}")

    # Dense-interpolate physical factor columns (sparse offline measurements)
    factor_cols = list(PHYSICAL_FACTORS.values())
    for col in factor_cols:
        if col == "__TIME_NORM__":
            continue          # computed below, no NaNs
        df_ff[col] = df_ff[col].interpolate(method="linear").ffill().bfill()

    # Compute normalised batch time (min-max to [0,1] per batch)
    df_ff["__TIME_NORM__"] = df_ff.groupby("_batch_id")["Time (h)"].transform(
        lambda t: (t - t.min()) / (t.max() - t.min() + 1e-12)
    )

    # Subsample
    rng   = np.random.default_rng(seed)
    idx   = rng.choice(len(df_ff), size=min(n_samples, len(df_ff)), replace=False)
    df_sub = df_ff.iloc[idx].reset_index(drop=True)
    print(f"[DATA]  Subsampled: {len(df_sub):,} rows  (seed={seed})")

    V = df_sub[factor_cols].values.astype(np.float32)
    return df_sub, V


# ── ───────────────────────────────────────────────────────────────────────────
# ENCODER LOADING
# ── ───────────────────────────────────────────────────────────────────────────

def encode_with(
    model_name: str,
    df:         pd.DataFrame,
    device:     torch.device,
) -> np.ndarray:
    """
    Load the frozen encoder for model_name, encode df, return Z (N, D).
    """
    import importlib
    cfg = ENCODERS[model_name]
    mod = importlib.import_module(cfg["module"])
    EncoderClass = getattr(mod, cfg["class"])

    print(f"\n[ENC]  Loading {model_name} encoder ...")
    enc = EncoderClass(
        ckpt_path   = cfg["ckpt"],
        scaler_path = cfg["scaler"],
        device      = device,
        latent_dim  = cfg["latent_dim"],
    )

    print(f"[ENC]  Encoding {len(df):,} rows ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = enc.encode_dataframe(df)    # (N, D)

    print(f"[ENC]  Z shape: {Z.shape}  mean={Z.mean():.4f}  std={Z.std():.4f}")
    return Z.astype(np.float64)


# ── ───────────────────────────────────────────────────────────────────────────
# DCI CORE
# ── ───────────────────────────────────────────────────────────────────────────

def _entropy_nats(p: np.ndarray) -> float:
    """Shannon entropy in nats for a probability vector p."""
    p = np.clip(p, 1e-12, None)
    return float(-(p * np.log(p)).sum())


def compute_dci(
    Z:            np.ndarray,
    V:            np.ndarray,
    factor_names: List[str],
    seed:         int = RANDOM_STATE,
    n_estimators: int = GBR_ESTIMATORS,
    max_depth:    int = GBR_DEPTH,
    min_leaf:     int = GBR_MIN_LEAF,
    subsample:    float = GBR_SUBSAMPLE,
    test_frac:    float = TEST_FRAC,
) -> DCIResult:
    """
    Compute DCI scores from pre-encoded Z and factor matrix V.

    Parameters
    ----------
    Z            : (N, D) latent representations
    V            : (N, K) physical factor values
    factor_names : list of K factor labels
    seed         : random seed for GBR and train/test split

    Returns
    -------
    DCIResult namedtuple
    """
    N, D = Z.shape
    K    = V.shape[1]
    assert K == len(factor_names), "V.shape[1] must equal len(factor_names)"

    # ── Train / test split ─────────────────────────────────────────────────────
    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(N)
    n_tr  = int((1.0 - test_frac) * N)
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]

    Z_tr, Z_te = Z[tr_idx], Z[te_idx]

    R            = np.zeros((K, D), dtype=np.float64)   # importance matrix
    info_scores  = np.full(K, np.nan,  dtype=np.float64)

    print(f"\n[DCI]  Fitting {K} GradientBoosting regressors "
          f"(D={D}, N_tr={n_tr}, N_te={N - n_tr}) ...")

    for ki, fname in enumerate(factor_names):
        v   = V[:, ki].astype(np.float64)
        v_tr, v_te = v[tr_idx], v[te_idx]

        # Drop rows with NaN in either v or Z
        valid_tr = np.isfinite(v_tr) & np.isfinite(Z_tr).all(axis=1)
        valid_te = np.isfinite(v_te) & np.isfinite(Z_te).all(axis=1)
        if valid_tr.sum() < 50:
            print(f"    {fname:<24}  SKIPPED (only {valid_tr.sum()} valid train rows)")
            continue

        gbr = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            subsample=subsample,
            random_state=seed,
        )
        gbr.fit(Z_tr[valid_tr], v_tr[valid_tr])

        R[ki, :] = gbr.feature_importances_

        if valid_te.sum() >= 10:
            info_scores[ki] = max(0.0, gbr.score(Z_te[valid_te], v_te[valid_te]))
        else:
            info_scores[ki] = max(0.0, gbr.score(Z_tr[valid_tr], v_tr[valid_tr]))

        print(f"    {fname:<24}  I(R²)={info_scores[ki]:.4f}  "
              f"top dim: z{R[ki].argmax():02d} "
              f"(imp={R[ki].max():.4f})")

    # ── Disentanglement D ──────────────────────────────────────────────────────
    # For each latent dim d: how exclusively is its importance owned by one factor?
    # R_norm_col[:,d] is the distribution over factors for dim d
    col_totals = R.sum(axis=0)           # (D,)  total importance of each dim
    col_sum    = col_totals.sum()
    R_norm_col = R / (col_totals[None, :] + 1e-12)   # (K, D)  sums to 1 over K

    max_H_K = np.log(K) if K > 1 else 1.0
    D_d = np.array([
        1.0 - _entropy_nats(R_norm_col[:, d]) / max_H_K
        for d in range(D)
    ])
    # Weighted average: dims with zero importance don't contribute
    D_score = float(np.dot(D_d, col_totals) / (col_sum + 1e-12))

    # ── Completeness C ─────────────────────────────────────────────────────────
    # For each factor k: how concentrated is it on one latent dim?
    row_totals = R.sum(axis=1)           # (K,)
    R_norm_row = R / (row_totals[:, None] + 1e-12)   # (K, D)  sums to 1 over D

    max_H_D = np.log(D) if D > 1 else 1.0
    C_k = np.array([
        1.0 - _entropy_nats(R_norm_row[ki, :]) / max_H_D
        for ki in range(K)
    ])
    C_score = float(C_k.mean())

    # ── Informativeness I ──────────────────────────────────────────────────────
    valid_info = info_scores[np.isfinite(info_scores)]
    I_score    = float(valid_info.mean()) if len(valid_info) > 0 else np.nan

    print(f"\n  >>> D={D_score:.4f}  C={C_score:.4f}  I={I_score:.4f}")

    return DCIResult(
        D            = D_score,
        C            = C_score,
        I            = I_score,
        D_per_dim    = D_d,
        C_per_factor = C_k,
        I_per_factor = info_scores,
        R            = R,
        factor_names = factor_names,
    )


# ── ───────────────────────────────────────────────────────────────────────────
# PLOTS
# ── ───────────────────────────────────────────────────────────────────────────

def plot_importance_heatmap(
    result:     DCIResult,
    model_name: str,
    save_path:  str,
) -> None:
    """
    Heatmap of R[K × top-16 dims] — rows = factors, columns = latent dims.
    Annotated with per-cell importance values for the top dims.
    """
    R = result.R
    K, D = R.shape

    # Select top 16 dims by max importance across all factors
    top_dims = np.argsort(R.max(axis=0))[::-1][:16]
    R_sub    = R[:, top_dims]

    fig, ax = plt.subplots(figsize=(13, 4 + 0.35 * K))
    im = ax.imshow(R_sub, aspect="auto", cmap="YlOrRd", interpolation="nearest",
                   vmin=0.0)
    plt.colorbar(im, ax=ax, label="Feature importance", fraction=0.03, pad=0.02)

    ax.set_xticks(range(len(top_dims)))
    ax.set_xticklabels([f"z{d:02d}" for d in top_dims], rotation=45, ha="right",
                       fontsize=8)
    ax.set_yticks(range(K))
    ax.set_yticklabels(result.factor_names, fontsize=9)
    ax.set_xlabel("Latent dimension  [top-16 by max importance]", fontsize=9)
    ax.set_ylabel("Physical factor", fontsize=9)
    ax.set_title(
        f"{model_name} — DCI importance matrix\n"
        f"D={result.D:.3f}  C={result.C:.3f}  I={result.I:.3f}",
        fontsize=11, fontweight="bold",
    )

    # Annotate cells with importance > 0.05
    for ki in range(K):
        for di, d in enumerate(top_dims):
            val = R[ki, d]
            if val > 0.05:
                ax.text(di, ki, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if val < 0.4 else "white",
                        fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def plot_per_factor(
    result:     DCIResult,
    model_name: str,
    save_path:  str,
) -> None:
    """
    Two-panel per-factor breakdown:
      Left  : C_k  (completeness) — how concentrated factor k is on one dim
      Right : I_k  (informativeness) — R² of GBR for factor k
    Sorted by completeness descending.
    """
    K      = len(result.factor_names)
    order  = np.argsort(result.C_per_factor)[::-1]   # sort by C descending
    names  = [result.factor_names[i] for i in order]
    C_vals = result.C_per_factor[order]
    I_vals = result.I_per_factor[order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 0.55 * K + 3))
    y = np.arange(K)

    # Left: Completeness
    ax = axes[0]
    colors_c = ["#27ae60" if v > 0.5 else "#e74c3c" for v in C_vals]
    ax.barh(y, C_vals, height=0.6, color=colors_c)
    ax.axvline(result.C, color="k", linestyle="--", linewidth=1.5,
               label=f"Mean C = {result.C:.3f}")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Completeness  C_k", fontsize=9)
    ax.set_title("Per-factor Completeness\n(1 = one latent dim per factor)", fontsize=9)
    ax.set_xlim(0, 1.05); ax.legend(fontsize=8)
    for i, v in enumerate(C_vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7)

    # Right: Informativeness
    ax2 = axes[1]
    I_clean = np.nan_to_num(I_vals, nan=0.0)
    colors_i = ["#2980b9" if v > 0.7 else "#f39c12" if v > 0.3 else "#c0392b"
                for v in I_clean]
    ax2.barh(y, I_clean, height=0.6, color=colors_i)
    ax2.axvline(result.I, color="k", linestyle="--", linewidth=1.5,
                label=f"Mean I = {result.I:.3f}")
    ax2.set_yticks(y); ax2.set_yticklabels([], fontsize=9)
    ax2.set_xlabel("Informativeness  I_k  (R²)", fontsize=9)
    ax2.set_title("Per-factor Informativeness\n(R² of GBR predictor)", fontsize=9)
    ax2.set_xlim(0, 1.05); ax2.legend(fontsize=8)
    for i, v in enumerate(I_clean):
        ax2.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7)

    fig.suptitle(
        f"{model_name} — DCI per-factor decomposition  "
        f"[D={result.D:.3f}  C={result.C:.3f}  I={result.I:.3f}]",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def plot_dci_comparison(
    results:      Dict[str, DCIResult],
    save_path:    str,
) -> None:
    """
    Two-panel grouped-bar chart:
      Left : D / C / I scores side-by-side for all three models
      Right: Per-factor Completeness heatmap (K rows × 3 model cols)
    """
    model_names = list(results.keys())
    M = len(model_names)
    K = len(next(iter(results.values())).factor_names)
    factor_names = next(iter(results.values())).factor_names

    # ── Left panel: grouped bar D / C / I ────────────────────────────────────
    metrics   = ["Disentanglement (D)", "Completeness (C)", "Informativeness (I)"]
    metric_keys = ["D", "C", "I"]
    palette   = ["#2980b9", "#27ae60", "#e67e22"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5 + 0.3 * K))

    ax = axes[0]
    x  = np.arange(len(metrics))
    width = 0.22
    offsets = np.linspace(-(M - 1) / 2, (M - 1) / 2, M) * width

    for i, (mname, col) in enumerate(zip(model_names, palette)):
        r = results[mname]
        vals = [getattr(r, k) for k in metric_keys]
        bars = ax.bar(x + offsets[i], vals, width=width, label=mname,
                      color=col, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8,
                        fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Score  [0 – 1]", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title("DCI scores — CDAE / CVAE / β-VAE\n"
                 f"(N={N_SAMPLES:,} fault-free rows, K={K} physical factors)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(0, color="k", linewidth=0.5)

    # ── Right panel: C_k per-factor heatmap (K × M) ──────────────────────────
    ax2 = axes[1]
    C_matrix = np.array([results[m].C_per_factor for m in model_names]).T  # (K, M)
    im = ax2.imshow(C_matrix, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0,
                    interpolation="nearest")
    plt.colorbar(im, ax=ax2, label="Completeness C_k", fraction=0.05, pad=0.03)

    ax2.set_xticks(range(M))
    ax2.set_xticklabels(model_names, fontsize=10)
    ax2.set_yticks(range(K))
    ax2.set_yticklabels(factor_names, fontsize=9)
    ax2.set_title("Per-factor Completeness C_k\n(green = well-captured by one dim)",
                  fontsize=10, fontweight="bold")

    for ki in range(K):
        for mi in range(M):
            v = C_matrix[ki, mi]
            ax2.text(mi, ki, f"{v:.2f}", ha="center", va="center",
                     fontsize=8, color="black" if 0.2 < v < 0.85 else "white",
                     fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# ── ───────────────────────────────────────────────────────────────────────────
# RESULTS TABLE  (console + CSV)
# ── ───────────────────────────────────────────────────────────────────────────

def print_comparison_table(results: Dict[str, DCIResult]) -> None:
    model_names  = list(results.keys())
    factor_names = next(iter(results.values())).factor_names
    col_w = 10

    sep = "─" * (30 + col_w * len(model_names) * 2)
    print(f"\n{'='*len(sep)}")
    print(f"  DCI SCORE — {', '.join(model_names)}")
    print(f"{'='*len(sep)}")

    # Header
    header = f"  {'Metric / Factor':<28}"
    for m in model_names:
        header += f"{'C_k ['+m+']':>{col_w}}  {'I_k':>{col_w}}"
    print(header)
    print(sep)

    for fname in factor_names:
        row = f"  {fname:<28}"
        for m in model_names:
            r   = results[m]
            ki  = r.factor_names.index(fname)
            c_k = r.C_per_factor[ki]
            i_k = r.I_per_factor[ki]
            i_s = f"{i_k:.4f}" if np.isfinite(i_k) else "  N/A"
            row += f"{c_k:>{col_w}.4f}  {i_s:>{col_w}}"
        print(row)

    print(sep)
    summary = f"  {'D (Disentanglement)':<28}"
    for m in model_names:
        summary += f"{results[m].D:>{col_w}.4f}  {'':>{col_w}}"
    print(summary)
    summary = f"  {'C (Completeness)':<28}"
    for m in model_names:
        summary += f"{results[m].C:>{col_w}.4f}  {'':>{col_w}}"
    print(summary)
    summary = f"  {'I (Informativeness)':<28}"
    for m in model_names:
        summary += f"{results[m].I:>{col_w}.4f}  {'':>{col_w}}"
    print(summary)
    print(f"{'='*len(sep)}")


def save_csv(results: Dict[str, DCIResult], path: Path) -> None:
    model_names  = list(results.keys())
    factor_names = next(iter(results.values())).factor_names

    rows = []
    # Overall scores
    for m in model_names:
        r = results[m]
        rows.append({"model": m, "metric": "D_overall",   "value": r.D})
        rows.append({"model": m, "metric": "C_overall",   "value": r.C})
        rows.append({"model": m, "metric": "I_overall",   "value": r.I})

    # Per-factor
    for m in model_names:
        r = results[m]
        for ki, fname in enumerate(factor_names):
            rows.append({"model": m, "metric": f"C_k|{fname}",
                         "value": float(r.C_per_factor[ki])})
            rows.append({"model": m, "metric": f"I_k|{fname}",
                         "value": float(r.I_per_factor[ki]) if np.isfinite(r.I_per_factor[ki]) else float("nan")})

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Results saved → {path}")


# ── ───────────────────────────────────────────────────────────────────────────
# MAIN
# ── ───────────────────────────────────────────────────────────────────────────

def main(
    csv_path:  str = CSV_PATH,
    n_samples: int = N_SAMPLES,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DCI]  Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    factor_names = list(PHYSICAL_FACTORS.keys())

    # ── 1. Load data (shared across all models) ───────────────────────────────
    df_sub, V = load_data(csv_path=csv_path, n_samples=n_samples)
    print(f"\n[DATA]  V shape: {V.shape}   factors: {factor_names}")

    # ── 2. Compute DCI for each encoder ──────────────────────────────────────
    results: Dict[str, DCIResult] = {}

    for model_name in ENCODERS:
        print(f"\n{'─'*60}")
        print(f"  Model: {model_name}")
        print(f"{'─'*60}")

        Z = encode_with(model_name, df_sub, device)

        result = compute_dci(Z, V, factor_names, seed=RANDOM_STATE,
                             n_estimators=GBR_ESTIMATORS)
        results[model_name] = result

        slug = model_name.lower().replace("-", "_").replace(" ", "_")
        plot_importance_heatmap(
            result, model_name,
            save_path=str(OUTPUT_DIR / f"dci_importance_{slug}.png"),
        )
        plot_per_factor(
            result, model_name,
            save_path=str(OUTPUT_DIR / f"dci_per_factor_{slug}.png"),
        )

    # ── 3. Comparison plots & table ───────────────────────────────────────────
    plot_dci_comparison(
        results,
        save_path=str(OUTPUT_DIR / "dci_comparison.png"),
    )
    print_comparison_table(results)
    save_csv(results, OUTPUT_DIR / "dci_results.csv")
    print(f"\n  All DCI outputs in {OUTPUT_DIR}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DCI score for IndPenSim V3 Raman encoders (CDAE / CVAE / β-VAE).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv",       default=CSV_PATH,  help="Path to IndPenSim V3 CSV")
    parser.add_argument("--n-samples", default=N_SAMPLES, type=int,
                        help="Number of rows to subsample from fault-free batches")
    args = parser.parse_args()
    main(csv_path=args.csv, n_samples=args.n_samples)
