#!/usr/bin/env python
"""
beta_sweep_eval_extended.py
============================
Post-hoc evaluation of the beta sweep checkpoints with three new metrics:

  1. RMSE on R2+R3 test set  (batches 30–89, unseen during training)
  2. MIG score               (continuous Mutual Information Gap, 6 physical factors)
  3. S_sep                   (separation score between Penicillin and Glucose axes)

S_sep definition
----------------
    S_sep = min( max_j |r(z_j, P)|,  max_j |r(z_j, S)| )
    subject to:  argmax_j |r(z_j, P)| ≠ argmax_j |r(z_j, S)|
    If the same latent dim is the best for both P and S, S_sep = 0.

This quantifies whether the encoder has separated metabolic output (Penicillin)
from substrate (Glucose) onto distinct latent axes. S_sep → 1 means both
factors are well-captured AND on different dims.  S_sep = 0 means either
(a) one factor is not captured, or (b) both factors collapse onto one dim.

Usage
-----
    PYTHONPATH=. python scripts/beta_sweep_eval_extended.py
    PYTHONPATH=. python scripts/beta_sweep_eval_extended.py --betas 0.0 0.05 0.1
    PYTHONPATH=. python scripts/beta_sweep_eval_extended.py --n-mig-samples 5000

Outputs
-------
    outputs/beta_sweep_e2e/eval_extended.csv
    outputs/beta_sweep_e2e/eval_extended.json
    Console summary table
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

# ── Project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.algorithms  # noqa: F401 — populate REGISTRY
from src.algorithms.base import BaseAlgorithm
from src.autoencoder.model import BetaVAE_Raman
from src.data.beta_vae_encoder import BetaVAERamanEncoderV2
from src.data.dataset import RAMAN_START_COL_IDX, TARGET_COLS

# ── Constants ─────────────────────────────────────────────────────────────────
N_RAMAN      = 2001
LATENT_DIM   = 64
SEED         = 42
SWEEP_DIR    = Path("outputs/beta_sweep_e2e/checkpoints")

REGIME_IDS = {
    "R1":    set(range(0,  30)),
    "R2":    set(range(30, 60)),
    "R3":    set(range(60, 90)),
    "Fault": set(range(90, 100)),
}

# MIG physical factors — dense columns from IndPenSim V3
PHYSICAL_FACTORS: Dict[str, str] = {
    "Penicillin":   "Penicillin concentration(P:g/L)",
    "Glucose":      "Substrate concentration(S:g/L)",
    "DO2":          "Dissolved oxygen concentration(DO2:mg/L)",
    "pH":           "pH(pH:pH)",
    "Temperature":  "Temperature(T:K)",
    "Vessel Vol":   "Vessel Volume(V:L)",
}

DEFAULT_BETAS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
N_MIG_SAMPLES = 5000   # subsample for MI estimation (per regime)
MI_NEIGHBORS  = 5


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_batches_by_regime(
    csv_path: str,
    min_batch_len: int = 50,
) -> Dict[str, List[pd.DataFrame]]:
    """Load CSV, split into regime buckets, add helper columns."""
    print(f"\n[DATA] Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"[DATA] Shape: {df.shape[0]:,} × {df.shape[1]}")

    # Batch boundaries
    t_vals = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid
    df["_batch_id"] = batch_ids

    # Sparse measurement grid (before interpolation)
    df["_penicillin_sparse"] = df[TARGET_COLS[1]].copy()
    df["_biomass_sparse"]    = df[TARGET_COLS[0]].copy()

    # Dense interpolation for LSTM targets
    for tc in TARGET_COLS:
        df[tc] = df[tc].interpolate(method="linear").ffill().bfill()

    # Split into regimes
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


# ─────────────────────────────────────────────────────────────────────────────
# Encoder helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_encoder(beta: float, device: torch.device) -> BetaVAERamanEncoderV2:
    """Load frozen β-VAE encoder for a given beta value."""
    tag = _tag(beta)
    vae_ckpt   = str(SWEEP_DIR / tag / "vae.pt")
    vae_scaler = str(SWEEP_DIR / tag / "vae_scaler.joblib")
    if not Path(vae_ckpt).exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt}")
    return BetaVAERamanEncoderV2(vae_ckpt, vae_scaler, device)


def encode_batches(
    batches: List[pd.DataFrame],
    encoder: BetaVAERamanEncoderV2,
) -> List[pd.DataFrame]:
    """Pre-encode Raman → _raman_0.._raman_63 columns."""
    encoded = []
    for b in batches:
        b2 = b.copy()
        latents = encoder.encode_dataframe(b2)
        for d in range(latents.shape[1]):
            b2[f"_raman_{d}"] = latents[:, d]
        encoded.append(b2)
    return encoded


def extract_latents_and_factors(
    batches: List[pd.DataFrame],
    encoder: BetaVAERamanEncoderV2,
    n_samples: int = N_MIG_SAMPLES,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Encode all valid Raman rows from batches, collect physical factors.

    Returns
    -------
    Z : (N, 64) latent representations
    V : (N, K)  physical factor matrix
    factor_names : list of K names
    """
    from scipy.signal import savgol_filter

    z_parts = []
    v_parts = []
    factor_cols = list(PHYSICAL_FACTORS.values())

    for b in batches:
        all_cols = b.columns.tolist()
        n_use = min(N_RAMAN, len(all_cols) - RAMAN_START_COL_IDX)
        raman_cols = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + n_use]
        X_raw = b[raman_cols].values.astype(np.float32)
        energy = np.abs(X_raw).sum(axis=1) > 0
        if not energy.any():
            continue

        # Encode
        X_valid = np.clip(np.nan_to_num(X_raw[energy], 0.0), 0.0, None)
        X_sg = savgol_filter(
            X_valid.astype(np.float64), window_length=15,
            polyorder=2, deriv=1, axis=1,
        ).astype(np.float32)
        X_scaled = encoder._scaler.transform(X_sg).astype(np.float32)
        mu = encoder.encode(X_scaled)
        z_parts.append(mu)

        # Physical factors (dense-interpolated)
        v_row = np.zeros((energy.sum(), len(factor_cols)), dtype=np.float32)
        for ki, col in enumerate(factor_cols):
            vals = b[col].interpolate(method="linear").ffill().bfill().values
            v_row[:, ki] = vals[energy]
        v_parts.append(v_row)

    if not z_parts:
        return np.zeros((0, LATENT_DIM)), np.zeros((0, len(factor_cols))), list(PHYSICAL_FACTORS.keys())

    Z = np.concatenate(z_parts, axis=0).astype(np.float64)
    V = np.concatenate(v_parts, axis=0).astype(np.float64)

    # Subsample for speed
    if n_samples > 0 and len(Z) > n_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(Z), size=n_samples, replace=False)
        Z = Z[idx]
        V = V[idx]

    return Z, V, list(PHYSICAL_FACTORS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Metric 1: RMSE on R2+R3
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_rmse_r2r3(
    beta: float,
    r2r3_enc: List[pd.DataFrame],
    device: torch.device,
) -> Dict[str, float]:
    """Load LSTM checkpoint for beta, evaluate on pre-encoded R2+R3 batches."""
    tag = _tag(beta)
    lstm_ckpt = str(SWEEP_DIR / tag / "lstm.pt")
    if not Path(lstm_ckpt).exists():
        print(f"    [WARN] LSTM checkpoint not found: {lstm_ckpt}")
        return {"rmse_r2r3": float("nan"), "mae_r2r3": float("nan"), "r2_r2r3": float("nan")}

    alg = BaseAlgorithm.load(lstm_ckpt, device)
    metrics = alg.evaluate(r2r3_enc)
    rmse = float(metrics["pen"]["RMSE"])
    mae  = float(metrics["pen"]["MAE"])
    r2   = float(metrics["pen"]["R2"])
    return {"rmse_r2r3": rmse, "mae_r2r3": mae, "r2_r2r3": r2}


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2: MIG (continuous)
# ─────────────────────────────────────────────────────────────────────────────

def _fd_bins(v: np.ndarray) -> int:
    """Freedman-Diaconis bin count."""
    iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
    n = len(v)
    if iqr < 1e-10:
        return max(int(np.ceil(np.log2(n) + 1)), 10)
    delta = 2.0 * iqr * (n ** (-1.0 / 3.0))
    return max(int(np.ceil((v.max() - v.min()) / delta)), 10)


def _entropy_hist(v: np.ndarray) -> float:
    """Discrete entropy of binned continuous variable."""
    counts, _ = np.histogram(v, bins=_fd_bins(v))
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def compute_mig(Z: np.ndarray, V: np.ndarray, factor_names: List[str]) -> float:
    """
    Continuous MIG score (Chen et al. 2018).

    MIG = (1/K) Σ_k (MI_top1 - MI_top2) / H(v_k)
    """
    N, D = Z.shape
    K = V.shape[1]
    if N < 100:
        return float("nan")

    MI = np.zeros((K, D), dtype=np.float64)
    for k in range(K):
        MI[k] = mutual_info_regression(
            Z, V[:, k], n_neighbors=MI_NEIGHBORS, random_state=SEED,
        )

    H = np.array([_entropy_hist(V[:, k]) for k in range(K)])

    gaps = []
    for k in range(K):
        sorted_mi = np.sort(MI[k])[::-1]
        mi1, mi2 = sorted_mi[0], sorted_mi[1]
        gap = (mi1 - mi2) / (H[k] + 1e-10)
        gaps.append(gap)

    return float(np.mean(gaps))


# ─────────────────────────────────────────────────────────────────────────────
# Metric 3: S_sep (separation score)
# ─────────────────────────────────────────────────────────────────────────────

def compute_s_sep(
    Z: np.ndarray,
    V: np.ndarray,
    factor_names: List[str],
) -> Dict[str, float]:
    """
    S_sep = min( max_j |r(z_j, P)|, max_j |r(z_j, S)| )
    subject to argmax_j |r(z_j, P)| ≠ argmax_j |r(z_j, S)|.

    If both maxima land on the same dim, S_sep = 0.

    Also returns the individual correlations and winning dim indices.
    """
    N, D = Z.shape
    pen_idx = factor_names.index("Penicillin")
    glc_idx = factor_names.index("Glucose")

    pen = V[:, pen_idx]
    glc = V[:, glc_idx]

    # Absolute Pearson r per dim
    r_pen = np.array([abs(pearsonr(Z[:, d], pen)[0]) for d in range(D)])
    r_glc = np.array([abs(pearsonr(Z[:, d], glc)[0]) for d in range(D)])

    best_pen_dim = int(np.argmax(r_pen))
    best_glc_dim = int(np.argmax(r_glc))
    max_r_pen = float(r_pen[best_pen_dim])
    max_r_glc = float(r_glc[best_glc_dim])

    if best_pen_dim == best_glc_dim:
        s_sep = 0.0
    else:
        s_sep = min(max_r_pen, max_r_glc)

    return {
        "s_sep":        s_sep,
        "max_r_pen":    max_r_pen,
        "max_r_glc":    max_r_glc,
        "best_pen_dim": best_pen_dim,
        "best_glc_dim": best_glc_dim,
        "same_dim":     best_pen_dim == best_glc_dim,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tag(beta: float) -> str:
    if beta == 0.0:
        return "b0"
    s = f"{beta:g}"
    return f"b{s}"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    csv_path:     str = "./100_Batches_IndPenSim_V3.csv",
    betas:        List[float] = DEFAULT_BETAS,
    n_mig_samples: int = N_MIG_SAMPLES,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Device: {device}")
    print(f"[EVAL] Betas: {betas}")
    print(f"[EVAL] MIG samples: {n_mig_samples}")

    # ── Load data ─────────────────────────────────────────────────────────────
    buckets = load_batches_by_regime(csv_path)
    r2r3_batches = buckets["R2"] + buckets["R3"]
    print(f"\n[EVAL] R2+R3 test set: {len(r2r3_batches)} batches")

    # ── Evaluate each beta ────────────────────────────────────────────────────
    results: List[Dict] = []

    for beta in betas:
        tag = _tag(beta)
        vae_ckpt = SWEEP_DIR / tag / "vae.pt"
        if not vae_ckpt.exists():
            print(f"\n  β={beta} — checkpoint not found, skipping.")
            continue

        print(f"\n{'='*70}")
        print(f"  β = {beta}   ({tag})")
        print(f"{'='*70}")
        t0 = time.time()

        row: Dict = {"beta": beta, "tag": tag}

        # ── Load encoder ──────────────────────────────────────────────────────
        encoder = load_encoder(beta, device)

        # ── 1. RMSE on R2+R3 ─────────────────────────────────────────────────
        print("  [1/3] Encoding R2+R3 and evaluating LSTM RMSE ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r2r3_enc = encode_batches(r2r3_batches, encoder)
        rmse_result = evaluate_rmse_r2r3(beta, r2r3_enc, device)
        row.update(rmse_result)
        print(f"        RMSE={rmse_result['rmse_r2r3']:.4f}  "
              f"MAE={rmse_result['mae_r2r3']:.4f}  "
              f"R²={rmse_result['r2_r2r3']:.4f}")

        # ── 2. MIG + S_sep (compute latents from R2+R3) ──────────────────────
        print("  [2/3] Computing MIG ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z, V, factor_names = extract_latents_and_factors(
                r2r3_batches, encoder, n_samples=n_mig_samples,
            )
        print(f"        Z: {Z.shape}  V: {V.shape}")

        mig = compute_mig(Z, V, factor_names)
        row["mig"] = mig
        print(f"        MIG = {mig:.4f}")

        # ── 3. S_sep ─────────────────────────────────────────────────────────
        print("  [3/3] Computing S_sep ...")
        sep = compute_s_sep(Z, V, factor_names)
        row.update(sep)
        print(f"        S_sep = {sep['s_sep']:.4f}  "
              f"(Pen: z{sep['best_pen_dim']} |r|={sep['max_r_pen']:.3f},  "
              f"Glc: z{sep['best_glc_dim']} |r|={sep['max_r_glc']:.3f})")
        if sep["same_dim"]:
            print(f"        ⚠ Both factors on SAME dim → S_sep forced to 0")

        elapsed = time.time() - t0
        row["elapsed_s"] = elapsed
        print(f"  Done in {elapsed:.1f}s")
        results.append(row)

        # Free memory
        del encoder, r2r3_enc, Z, V
        torch.cuda.empty_cache()

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir = Path("outputs/beta_sweep_e2e")
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out_dir / "eval_extended.json"
    json_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n  Saved → {json_path}")

    # CSV
    csv_path_out = out_dir / "eval_extended.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"  Saved → {csv_path_out}")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  β-SWEEP EXTENDED EVALUATION — R2+R3 test set ({len(r2r3_batches)} batches)")
    print(f"{'='*90}")
    print(f"  {'β':<8} {'RMSE':>8} {'MAE':>8} {'R²':>8} "
          f"{'MIG':>8} {'S_sep':>8} {'Pen dim':>9} {'Glc dim':>9} "
          f"{'|r_P|':>7} {'|r_S|':>7}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} "
          f"{'-'*8} {'-'*8} {'-'*9} {'-'*9} "
          f"{'-'*7} {'-'*7}")
    for r in results:
        print(f"  {r['beta']:<8.4f} {r['rmse_r2r3']:>8.4f} {r['mae_r2r3']:>8.4f} "
              f"{r['r2_r2r3']:>8.4f} {r['mig']:>8.4f} {r['s_sep']:>8.4f} "
              f"{'z'+str(r['best_pen_dim']):>9} {'z'+str(r['best_glc_dim']):>9} "
              f"{r['max_r_pen']:>7.3f} {r['max_r_glc']:>7.3f}")
    print(f"{'='*90}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extended β-sweep evaluation: RMSE R2+R3, MIG, S_sep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", default="./100_Batches_IndPenSim_V3.csv",
                        help="Path to IndPenSim V3 CSV")
    parser.add_argument("--betas", nargs="+", type=float, default=DEFAULT_BETAS,
                        help="Beta values to evaluate")
    parser.add_argument("--n-mig-samples", type=int, default=N_MIG_SAMPLES,
                        help="Subsample size for MIG computation")
    parser.add_argument("--sweep-dir", type=str, default=str(SWEEP_DIR),
                        help="Directory containing beta sweep checkpoints")
    args = parser.parse_args()

    SWEEP_DIR = Path(args.sweep_dir)
    main(csv_path=args.csv, betas=args.betas, n_mig_samples=args.n_mig_samples)
