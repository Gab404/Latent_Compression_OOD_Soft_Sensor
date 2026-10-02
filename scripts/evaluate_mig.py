"""
scripts/evaluate_mig.py
========================
Mutual Information Gap (MIG) score for continuous ground-truth physical
variables from the IndPenSim V3 bioreactor dataset.

Background
----------
MIG (Chen et al., 2018 "Isolating Sources of Disentanglement in VAEs") was
originally defined for discrete generative factors.  For continuous physical
variables we adopt three adaptations:

  1. MI estimation  : sklearn's mutual_info_regression (Kraskov KNN estimator)
                      instead of empirical histograms.  The KNN approach handles
                      continuous distributions without discretisation and is
                      consistent for any smooth distribution.

  2. Entropy normalization : h(v_k) estimated via histogram with
                             Freedman-Diaconis bin width.  This gives a
                             strictly positive, scale-invariant denominator.
                             Differential entropy from KNN degenerates at
                             I(X;X) (nearest-neighbour distance → 0), so
                             we keep estimation and normalisation in separate
                             estimators on purpose.

  3. MIG formula (unchanged):
        gap_k  = ( MI_{(1)}(z, v_k) - MI_{(2)}(z, v_k) ) / H(v_k)
        MIG    = (1/K) * Σ_k  gap_k

     gap_k ∈ [0, 1] if the MI estimator is consistent.
     MIG → 1  : each physical factor is captured by a unique latent dimension
     MIG → 0  : factors are entangled (spread across many dimensions)

Usage
-----
    PYTHONPATH=. python scripts/evaluate_mig.py [--csv PATH] [--n-samples N]

Output
------
    • Per-factor MI heat-map      : outputs/mig/mi_heatmap_<model>.png
    • Per-factor top-2 bar chart  : outputs/mig/mig_bargap_<model>.png
    • Console comparison table    : all three models side-by-side
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from sklearn.feature_selection import mutual_info_regression

# ── Project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset import RAMAN_START_COL_IDX

# ── ──────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ── ──────────────────────────────────────────────────────────────────────────

CSV_PATH   = "./100_Batches_IndPenSim_V3.csv"
OUTPUT_DIR = Path("./outputs/mig")

# Fault-free batch IDs (0-indexed)
FAULT_FREE_IDS = set(range(90))

# Physical ground-truth factors V — dense (interpolated) process variables.
# Chosen to cover the main biological / physical axes of variation:
#   metabolic output (Pen, Biomass proxy via CO2/O2)
#   substrate (glucose, DO2, pH)
#   environmental setpoints (T, RPM, aeration)
#   reactor state (vessel volume)
#   temporal progression (normalised batch time)
PHYSICAL_FACTORS: Dict[str, str] = {
    "Penicillin (g/L)":    "Penicillin concentration(P:g/L)",
    "Glucose (g/L)":       "Substrate concentration(S:g/L)",
    "DO2 (mg/L)":          "Dissolved oxygen concentration(DO2:mg/L)",
    "pH":                  "pH(pH:pH)",
    "Temperature (K)":     "Temperature(T:K)",
    "Vessel Vol (L)":      "Vessel Volume(V:L)",
    "Aeration (L/h)":      "Aeration rate(Fg:L/h)",
    "Batch Time (norm)":   "__TIME_NORM__",   # computed per-batch, not a CSV column
    "PAA (g/L)":           "PAA concentration offline(PAA_offline:PAA (g L^{-1}))",
    "NH3 (g/L)":           "NH_3 concentration off-line(NH3_offline:NH3 (g L^{-1}))",
    "Viscosity (cP)":      "Viscosity(Viscosity_offline:centPoise)",
}

# Encoder definitions
# Encoder definitions
ENCODERS = {
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
    "Beta-VAE": { # Beta-VAE Heavy (Protocole 1)
        "class":       "BetaVAERamanEncoderV2",
        "module":      "src.data.beta_vae_encoder",
        "ckpt":        "/home/gabriel/ReaKt/outputs/seed_sweep/beta_vae_pi_lstm/p1/encoders/beta_vae_s42.pt",
        "scaler":      "/home/gabriel/ReaKt/outputs/seed_sweep/beta_vae_pi_lstm/p1/encoders/beta_vae_scaler_s42.joblib",
        "latent_dim":  64,
    },
}

# MIG hyper-parameters
N_SAMPLES    = 20_000   # random subsample from fault-free rows (for speed)
N_NEIGHBORS  = 5        # Kraskov KNN neighbours for MI estimation
RANDOM_STATE = 42


# ── ──────────────────────────────────────────────────────────────────────────
# ENTROPY ESTIMATION
# ── ──────────────────────────────────────────────────────────────────────────

def _fd_bins(v: np.ndarray) -> int:
    """
    Freedman-Diaconis rule for histogram bin count.

    Δ = 2 · IQR · n^{-1/3}  →  n_bins = ceil((max-min) / Δ)

    Falls back to Sturges' rule if IQR=0 (constant or near-constant variable).
    """
    iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
    n   = len(v)
    if iqr < 1e-10:
        return max(int(np.ceil(np.log2(n) + 1)), 10)   # Sturges
    delta   = 2.0 * iqr * (n ** (-1.0 / 3.0))
    n_bins  = int(np.ceil((v.max() - v.min()) / delta))
    return max(n_bins, 10)   # at least 10 bins


def entropy_histogram(v: np.ndarray) -> float:
    """
    Estimate the Shannon entropy of a continuous scalar via histogram
    discretisation (Freedman-Diaconis bins).

    H_hist = -Σ p_i · log(p_i)     [nats]

    This is the standard discrete entropy of the binned variable.  It is
    always ≥ 0 and provides a consistent upper bound on differential entropy
    (h(X) ≤ H_hist when Δ < 1).  Using it as the denominator in the MIG gap
    ensures gap ∈ [0, 1] as long as MI ≤ H_hist (which holds empirically for
    the Kraskov estimator at moderate sample sizes).
    """
    n_bins = _fd_bins(v)
    counts, _ = np.histogram(v, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


# ── ──────────────────────────────────────────────────────────────────────────
# MIG CORE
# ── ──────────────────────────────────────────────────────────────────────────

class FactorResult(NamedTuple):
    name:     str
    gap:      float     # normalised gap
    mi_top1:  float     # MI of best latent dimension  [nats]
    mi_top2:  float     # MI of 2nd-best latent dimension  [nats]
    H:        float     # estimated entropy of this factor  [nats]
    top1_dim: int       # index of best latent dimension
    top2_dim: int       # index of 2nd-best latent dimension


def calculate_mig_continuous(
    Z:           np.ndarray,
    V:           np.ndarray,
    factor_names: List[str],
    n_neighbors:  int = N_NEIGHBORS,
    random_state: int = RANDOM_STATE,
) -> Tuple[float, List[FactorResult], np.ndarray]:
    """
    Calculate continuous MIG score.

    Parameters
    ----------
    Z            : (N, d_z)   latent representation matrix
    V            : (N, d_v)   ground-truth physical factor matrix
    factor_names : list of d_v strings — human-readable factor labels
    n_neighbors  : KNN parameter for mutual_info_regression
    random_state : reproducibility seed

    Returns
    -------
    mig          : float               overall MIG score (mean across factors)
    per_factor   : list[FactorResult]  per-factor details
    MI           : (d_v, d_z) ndarray  full MI matrix [nats]
    """
    N,  d_z = Z.shape
    d_v     = V.shape[1]
    assert d_v == len(factor_names), "factor_names must have length V.shape[1]"

    # ── 1. MI matrix  (d_v × d_z) ─────────────────────────────────────────────
    # mutual_info_regression(X, y) returns shape (d_z,) — one MI value per
    # column of X.  Z columns = latent dimensions; y = one physical factor.
    print("  Computing MI matrix ...")
    MI = np.zeros((d_v, d_z), dtype=np.float64)
    for k, name in enumerate(factor_names):
        MI[k] = mutual_info_regression(
            Z, V[:, k],
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        top1 = MI[k].max()
        print(f"    {name:<22}  max_MI={top1:.4f} nats  "
              f"(z{MI[k].argmax():02d})")

    # ── 2. Entropy of each factor ──────────────────────────────────────────────
    H = np.array([entropy_histogram(V[:, k]) for k in range(d_v)])

    # ── 3. Per-factor normalised gap ──────────────────────────────────────────
    per_factor: List[FactorResult] = []
    for k in range(d_v):
        sorted_idx = np.argsort(MI[k])[::-1]   # descending
        top1_dim   = sorted_idx[0]
        top2_dim   = sorted_idx[1]
        mi1        = MI[k, top1_dim]
        mi2        = MI[k, top2_dim]
        Hk         = H[k]
        gap        = (mi1 - mi2) / (Hk + 1e-10)

        per_factor.append(FactorResult(
            name     = factor_names[k],
            gap      = float(gap),
            mi_top1  = float(mi1),
            mi_top2  = float(mi2),
            H        = float(Hk),
            top1_dim = int(top1_dim),
            top2_dim = int(top2_dim),
        ))

    mig = float(np.mean([r.gap for r in per_factor]))
    return mig, per_factor, MI


# ── ──────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ── ──────────────────────────────────────────────────────────────────────────

def load_data(
    csv_path:  str = CSV_PATH,
    n_samples: int = N_SAMPLES,
    seed:      int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load IndPenSim V3, keep fault-free rows, subsample, return DataFrame + V.

    Returns
    -------
    df_sub  : subsampled DataFrame (used by encoders)
    V       : (n_samples, d_v)  physical factor matrix, float32
    """
    print(f"\n[DATA]  Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"[DATA]  Raw shape: {df.shape[0]:,} × {df.shape[1]}")

    # ── Detect batch boundaries ────────────────────────────────────────────────
    t_vals    = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid
    df["_batch_id"] = batch_ids

    # ── Keep fault-free rows with valid Raman ─────────────────────────────────
    raman_cols = df.columns[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + 2001]
    raman_energy = df[raman_cols].values.sum(axis=1)
    mask_ff    = df["_batch_id"].isin(FAULT_FREE_IDS)
    mask_raman = raman_energy > 0
    df_ff = df[mask_ff & mask_raman].copy().reset_index(drop=True)
    print(f"[DATA]  Fault-free rows with Raman: {len(df_ff):,}")

    # ── Compute normalised batch time (min-max to [0,1] per batch) ──────────
    df_ff["__TIME_NORM__"] = df_ff.groupby("_batch_id")["Time (h)"].transform(
        lambda t: (t - t.min()) / (t.max() - t.min() + 1e-12)
    )

    # ── Interpolate dense targets (some physical cols have NaNs) ──────────────
    factor_cols = list(PHYSICAL_FACTORS.values())
    for col in factor_cols:
        if col == "__TIME_NORM__":
            continue          # already computed, no NaNs
        df_ff[col] = df_ff[col].interpolate(method="linear").ffill().bfill()

    # ── Subsample ─────────────────────────────────────────────────────────────
    rng      = np.random.default_rng(seed)
    idx      = rng.choice(len(df_ff), size=min(n_samples, len(df_ff)), replace=False)
    df_sub   = df_ff.iloc[idx].reset_index(drop=True)
    print(f"[DATA]  Subsampled: {len(df_sub):,} rows  (seed={seed})")

    # ── Build V matrix ────────────────────────────────────────────────────────
    V = df_sub[factor_cols].values.astype(np.float32)
    return df_sub, V


# ── ──────────────────────────────────────────────────────────────────────────
# ENCODER LOADING + LATENT EXTRACTION
# ── ──────────────────────────────────────────────────────────────────────────

def encode_with(
    model_name: str,
    df:         pd.DataFrame,
    device:     torch.device,
) -> np.ndarray:
    """
    Load the frozen encoder for model_name, encode df, return Z (N, d_z).
    """
    cfg = ENCODERS[model_name]

    # Dynamic import of the encoder class
    import importlib
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
        Z = enc.encode_dataframe(df)    # (N, d_z)

    print(f"[ENC]  Z shape: {Z.shape}  mean={Z.mean():.4f}  std={Z.std():.4f}")
    return Z.astype(np.float64)


# ── ──────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ── ──────────────────────────────────────────────────────────────────────────

def plot_mi_heatmap(
    MI:           np.ndarray,
    factor_names: List[str],
    model_name:   str,
    save_path:    str,
) -> None:
    """
    Heatmap of MI[factor, latent_dim].  Latent dims on x-axis, factors on y-axis.
    Only the top-16 most informative latent dims per model are shown to keep
    the figure readable.
    """
    # Select top 16 latent dims by max MI across all factors
    top_dims = np.argsort(MI.max(axis=0))[::-1][:16]
    MI_sub   = MI[:, top_dims]

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(MI_sub, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="MI (nats)")

    ax.set_xticks(range(len(top_dims)))
    ax.set_xticklabels([f"z{d:02d}" for d in top_dims], rotation=45, ha="right",
                       fontsize=8)
    ax.set_yticks(range(len(factor_names)))
    ax.set_yticklabels(factor_names, fontsize=9)
    ax.set_title(f"{model_name} — MI(latent dim, physical factor)  "
                 f"[top-16 dims shown]", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def plot_mig_bar(
    per_factor:  List[FactorResult],
    model_name:  str,
    save_path:   str,
) -> None:
    """
    Stacked horizontal bar: MI_top1 (teal) + (MI_top1 - MI_top2) gap (orange)
    per physical factor, sorted by gap descending.
    """
    per_factor = sorted(per_factor, key=lambda r: r.gap, reverse=True)
    names      = [r.name       for r in per_factor]
    mi1        = np.array([r.mi_top1  for r in per_factor])
    mi2        = np.array([r.mi_top2  for r in per_factor])
    gaps       = np.array([r.gap      for r in per_factor])
    H          = np.array([r.H        for r in per_factor])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    y = np.arange(len(names))

    # ── Left: MI bar (top1 = base, gap = additional bar) ────────────────────
    ax = axes[0]
    ax.barh(y, mi2,      height=0.6, color="#4c9be8", label="MI top-2")
    ax.barh(y, mi1 - mi2, height=0.6, left=mi2,
            color="#f4a428", label="Gap (top1 − top2)")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mutual Information (nats)")
    ax.set_title("MI breakdown per factor")
    ax.legend(fontsize=8)

    # Annotate top-1 latent dim
    for i, r in enumerate(per_factor):
        ax.text(r.mi_top1 + 0.002, i, f"z{r.top1_dim:02d}", va="center",
                fontsize=7, color="#333")

    # ── Right: normalised gap bar ────────────────────────────────────────────
    ax2 = axes[1]
    colors = ["#27ae60" if g > 0.05 else "#e74c3c" for g in gaps]
    ax2.barh(y, gaps, height=0.6, color=colors)
    ax2.axvline(np.mean(gaps), color="k", linestyle="--", linewidth=1.5,
                label=f"MIG = {np.mean(gaps):.4f}")
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("Normalised gap  (MI gap / H(factor))")
    ax2.set_title("MIG contribution per factor")
    ax2.set_xlim(0, max(gaps.max() * 1.15, 0.1))
    ax2.legend(fontsize=9)

    fig.suptitle(f"{model_name} — Continuous MIG decomposition", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# ── ──────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ── ──────────────────────────────────────────────────────────────────────────

def print_comparison_table(
    results: Dict[str, Tuple[float, List[FactorResult]]],
) -> None:
    """
    Print a compact side-by-side comparison table of MIG scores per model
    and per physical factor.
    """
    model_names  = list(results.keys())
    factor_names = [r.name for r in next(iter(results.values()))[1]]

    col_w = 12
    sep   = "-" * (26 + col_w * len(model_names))

    print(f"\n{'='*len(sep)}")
    print(f"  CONTINUOUS MIG SCORE — {', '.join(model_names)}")
    print(f"{'='*len(sep)}")
    header = f"  {'Physical Factor':<24}" + "".join(
        f"{m:>{col_w}}" for m in model_names
    )
    print(header)
    print(sep)

    for fname in factor_names:
        row = f"  {fname:<24}"
        for mname in model_names:
            _, per_factor = results[mname]
            res = next(r for r in per_factor if r.name == fname)
            row += f"{res.gap:>{col_w}.4f}"
        print(row)

    print(sep)
    overall_row = f"  {'MIG (mean)':<24}"
    for mname in model_names:
        mig, _ = results[mname]
        overall_row += f"{mig:>{col_w}.4f}"
    print(overall_row)
    print(f"{'='*len(sep)}")

    # Per-factor top-dim breakdown
    print(f"\n  Top latent dimension per factor and model:")
    print(f"  {'Physical Factor':<24}" + "".join(
        f"{m:>{col_w}}" for m in model_names
    ))
    print(sep)
    for fname in factor_names:
        row = f"  {fname:<24}"
        for mname in model_names:
            _, per_factor = results[mname]
            res = next(r for r in per_factor if r.name == fname)
            row += f"{'z'+str(res.top1_dim):>{col_w}}"
        print(row)
    print(sep)


# ── ──────────────────────────────────────────────────────────────────────────
# MAIN
# ── ──────────────────────────────────────────────────────────────────────────

def main(csv_path: str = CSV_PATH, n_samples: int = N_SAMPLES) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MIG]  Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    factor_names = list(PHYSICAL_FACTORS.keys())

    # ── 1. Load data once (shared across all models) ──────────────────────────
    df_sub, V = load_data(csv_path=csv_path, n_samples=n_samples)
    print(f"\n[DATA]  V shape: {V.shape}   factors: {factor_names}")
    print(f"[DATA]  V stats (mean ± std per factor):")
    for i, fname in enumerate(factor_names):
        print(f"         {fname:<22}  {V[:,i].mean():8.3f} ± {V[:,i].std():7.3f}")

    # ── 2. Compute MIG for each encoder ──────────────────────────────────────
    results: Dict[str, Tuple[float, List[FactorResult], np.ndarray]] = {}

    for model_name in ENCODERS:
        print(f"\n{'─'*60}")
        print(f"  Model: {model_name}")
        print(f"{'─'*60}")

        # Encode
        Z = encode_with(model_name, df_sub, device)

        # MIG
        print(f"\n[MIG]  Computing MIG for {model_name}  "
              f"(N={len(Z)}, d_z={Z.shape[1]}, K={len(factor_names)}, "
              f"n_neighbors={N_NEIGHBORS}) ...")
        mig, per_factor, MI = calculate_mig_continuous(
            Z, V, factor_names,
            n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE,
        )
        results[model_name] = (mig, per_factor, MI)
        print(f"\n  >>> {model_name}  MIG = {mig:.4f}")

        # Plots
        plot_mi_heatmap(
            MI, factor_names, model_name,
            save_path=str(OUTPUT_DIR / f"mi_heatmap_{model_name.lower().replace('-','_')}.png"),
        )
        plot_mig_bar(
            per_factor, model_name,
            save_path=str(OUTPUT_DIR / f"mig_bargap_{model_name.lower().replace('-','_')}.png"),
        )

    # ── 3. Comparison table ───────────────────────────────────────────────────
    print_comparison_table({k: (v[0], v[1]) for k, v in results.items()})

    # ── 4. Combined MIG bar chart ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    model_list    = list(results.keys())
    mig_scores    = [results[m][0] for m in model_list]
    colors        = ["#2980b9", "#27ae60", "#e67e22"]
    bars          = ax.bar(model_list, mig_scores, color=colors, width=0.5, edgecolor="white")

    for bar, score in zip(bars, mig_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{score:.4f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")

    ax.set_ylabel("MIG Score", fontsize=12)
    ax.set_title("Continuous MIG — β-VAE encoder comparison\n"
                 f"(N={n_samples:,} fault-free rows, K={len(factor_names)} physical factors)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(mig_scores) * 1.25)
    ax.axhline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    combined_path = str(OUTPUT_DIR / "mig_comparison.png")
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved {combined_path}")
    print(f"  All outputs in {OUTPUT_DIR}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Continuous MIG score for IndPenSim V3 Raman encoders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv",       default=CSV_PATH,  help="Path to IndPenSim V3 CSV")
    parser.add_argument("--n-samples", default=N_SAMPLES, type=int,
                        help="Number of rows to subsample from fault-free batches")
    args = parser.parse_args()
    main(csv_path=args.csv, n_samples=args.n_samples)
