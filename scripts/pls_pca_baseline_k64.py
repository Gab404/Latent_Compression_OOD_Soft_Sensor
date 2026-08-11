"""
scripts/pls_pca_baseline_k64.py
================================
PLS and PCA-PCR (K=64) direct Raman→Penicillin regression baselines.

Both models are static (per-timestep): a single spectrum maps directly to a
Penicillin concentration.  No temporal context is used (contrast with PI-LSTM).

Training split (default)
------------------------
    Train : batches 1–90  (IDs 0–89, R1 + R2 + R3)
    Test  : same 90 batches  (in-sample, to verify fit quality)
            + batches 91–100 (IDs 90–99, Fault regime — OOD test)

Regime-ablation usage
---------------------
    PYTHONPATH=. python scripts/pls_pca_baseline_k64.py --train-regime R1
    Trains only on R1 (batches 1–30) and tests on R1 / R2 / R3 / Fault.

K = 64 components for both models, matching the VAE/CDAE latent dimension.

Evaluation
----------
Metrics are computed only at sparse offline measurement time-points
(where Penicillin_offline is non-NaN), consistent with the PI-LSTM evaluation.

Models
------
  PLS    : PLSRegression(n_components=64, scale=True)
  PCA-PCR: StandardScaler → PCA(64) → LinearRegression
            (PCA on X, then linear regression on the 64 PC scores)

Usage
-----
    PYTHONPATH=. python scripts/pls_pca_baseline_k64.py
    PYTHONPATH=. python scripts/pls_pca_baseline_k64.py --csv path/to/data.csv
    PYTHONPATH=. python scripts/pls_pca_baseline_k64.py --train-regime R1
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── Project root ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset import RAMAN_START_COL_IDX, TARGET_COLS, FAULT_COL

# ── Config ─────────────────────────────────────────────────────────────────────
CSV_PATH      = "./100_Batches_IndPenSim_V3.csv"
CKPT_DIR      = Path("./checkpoints")
OUTPUT_DIR    = Path("./outputs/pls_pca_k64")
K             = 64
N_RAMAN       = 2001
SG_WINDOW, SG_POLY, SG_DERIV = 15, 2, 1
MIN_BATCH_LEN = 50

# Named regime ID sets (0-indexed)
REGIMES = {
    "R1":    set(range(0,   30)),
    "R2":    set(range(30,  60)),
    "R3":    set(range(60,  90)),
    "Fault": set(range(90, 100)),
}
ALL_IDS = set(range(100))


# ── Preprocessing ──────────────────────────────────────────────────────────────

def _sg(X: np.ndarray) -> np.ndarray:
    return savgol_filter(X, window_length=SG_WINDOW,
                         polyorder=SG_POLY, deriv=SG_DERIV, axis=1)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_batches(csv_path: str) -> Dict[str, List[Dict]]:
    """
    Load CSV, return batches bucketed by regime: R1 / R2 / R3 / Fault.

    Each entry:
        X_raman  : (T, N_RAMAN) float64  — SG-derivative spectra
        y_dense  : (T,)         float64  — interpolated penicillin (g/L)
        y_sparse : (T,)         float64  — raw offline measurements (NaN elsewhere)
        batch_id : int
    """
    print(f"\n[DATA]  Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"[DATA]  Shape: {df.shape[0]:,} × {df.shape[1]}")

    t_vals    = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid
    df["_batch_id"] = batch_ids
    print(f"[DATA]  Detected {bid + 1} batches")

    all_cols    = df.columns.tolist()
    raman_names = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + N_RAMAN]
    pen_col     = TARGET_COLS[1]

    buckets: Dict[str, List[Dict]] = {r: [] for r in REGIMES}

    for b_id in range(bid + 1):
        b = df[df["_batch_id"] == b_id].copy().reset_index(drop=True)
        if len(b) < MIN_BATCH_LEN:
            continue

        y_sparse = b[pen_col].values.copy()
        b[pen_col] = b[pen_col].interpolate(method="linear").ffill().bfill()

        X_raw = b[raman_names].values.astype(np.float64)
        X_raw = np.clip(np.nan_to_num(X_raw, 0.0), 0.0, None)
        mask  = np.abs(X_raw).sum(axis=1) > 0
        if mask.sum() < MIN_BATCH_LEN:
            continue

        entry = {
            "X_raman":  _sg(X_raw[mask]),
            "y_dense":  b[pen_col].values[mask],
            "y_sparse": y_sparse[mask],
            "batch_id": b_id,
        }

        for regime, ids in REGIMES.items():
            if b_id in ids:
                buckets[regime].append(entry)
                break

    for regime, batches in buckets.items():
        print(f"[DATA]  {regime}: {len(batches)} batches")
    return buckets


def _flat(batches: List[Dict], sparse_only: bool = False):
    """Flatten batches into (X, y, groups) arrays."""
    Xs, ys, gs = [], [], []
    for b in batches:
        if sparse_only:
            mask = ~np.isnan(b["y_sparse"])
            if mask.sum() == 0:
                continue
            Xs.append(b["X_raman"][mask])
            ys.append(b["y_sparse"][mask])
        else:
            Xs.append(b["X_raman"])
            ys.append(b["y_dense"])
        gs.append(np.full(len(Xs[-1]), b["batch_id"], dtype=np.int32))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)


# ── Evaluation helper ──────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "n": len(y_true)}


def evaluate_batches(
    predict_fn,           # callable: X (N, D) → y_pred (N,)
    batches:   List[Dict],
    label:     str,
    per_batch: bool = False,
) -> Dict:
    """
    Evaluate predict_fn on batches, using sparse measurement points only.
    Returns aggregated metrics and optionally per-batch breakdown.
    """
    all_true, all_pred = [], []
    pb_results = []

    for b in batches:
        mask = ~np.isnan(b["y_sparse"])
        if mask.sum() == 0:
            continue
        y_true = b["y_sparse"][mask]
        y_pred = predict_fn(b["X_raman"][mask])
        all_true.append(y_true)
        all_pred.append(y_pred)
        if per_batch:
            m = _metrics(y_true, y_pred)
            pb_results.append({"batch_id": b["batch_id"], **m})

    if not all_true:
        return {}

    agg = _metrics(np.concatenate(all_true), np.concatenate(all_pred))
    print(f"  [{label:<18}]  RMSE={agg['RMSE']:.4f} g/L  "
          f"MAE={agg['MAE']:.4f}  R²={agg['R2']:.4f}  n={agg['n']}")

    if per_batch:
        agg["per_batch"] = pb_results
    return agg


# ── PLS ────────────────────────────────────────────────────────────────────────

def fit_pls(X_train: np.ndarray, y_train: np.ndarray) -> PLSRegression:
    print(f"\n[PLS]   Fitting PLSRegression(n_components={K}, scale=True) "
          f"on {X_train.shape[0]:,} samples ...")
    pls = PLSRegression(n_components=K, scale=True)
    pls.fit(X_train, y_train)
    print("[PLS]   Done.")
    return pls


# ── PCA-PCR ────────────────────────────────────────────────────────────────────

class PCRModel:
    """StandardScaler → PCA(K) → LinearRegression."""

    def __init__(self, k: int = K):
        self.scaler = StandardScaler()
        self.pca    = PCA(n_components=k)
        self.lr     = LinearRegression()
        self.k      = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PCRModel":
        print(f"\n[PCR]   Fitting StandardScaler + PCA(n_components={self.k}) "
              f"+ LinearRegression on {X.shape[0]:,} samples ...")
        X_sc = self.scaler.fit_transform(X)
        Z    = self.pca.fit_transform(X_sc)
        self.lr.fit(Z, y)
        var_exp = self.pca.explained_variance_ratio_.cumsum()[-1]
        print(f"[PCR]   PCA variance explained by {self.k} components: {var_exp:.3%}")
        print("[PCR]   Done.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_sc = self.scaler.transform(X)
        Z    = self.pca.transform(X_sc)
        return self.lr.predict(Z).ravel()


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_per_batch_rmse(
    pls_pb:  List[Dict],
    pcr_pb:  List[Dict],
    title:   str,
    save_path: str,
) -> None:
    """Side-by-side per-batch RMSE bar chart for PLS and PCA-PCR."""
    ids    = [r["batch_id"] + 1 for r in pls_pb]   # 1-indexed display
    rmse_pls = [r["RMSE"] for r in pls_pb]
    rmse_pcr = [r["RMSE"] for r in pcr_pb]

    x   = np.arange(len(ids))
    w   = 0.38
    fig, ax = plt.subplots(figsize=(max(10, len(ids) * 0.55), 4.5))
    ax.bar(x - w/2, rmse_pls, width=w, label=f"PLS (K={K})",    color="#2980b9", alpha=0.85)
    ax.bar(x + w/2, rmse_pcr, width=w, label=f"PCA-PCR (K={K})", color="#e67e22", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in ids], rotation=45 if len(ids) > 20 else 0,
                       fontsize=8 if len(ids) > 20 else 9)
    ax.set_xlabel("Batch number", fontsize=10)
    ax.set_ylabel("Penicillin RMSE (g/L)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0)
    ax.axhline(np.mean(rmse_pls), color="#2980b9", linestyle="--", linewidth=1.2,
               alpha=0.6, label=f"PLS mean={np.mean(rmse_pls):.3f}")
    ax.axhline(np.mean(rmse_pcr), color="#e67e22", linestyle="--", linewidth=1.2,
               alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_summary_bar(results: Dict, split_keys: List[str], split_labels: List[str],
                     save_path: str) -> None:
    """Grouped bar: RMSE across evaluation splits for PLS and PCA-PCR."""
    models  = [f"PLS (K={K})", f"PCA-PCR (K={K})"]
    rmses   = [
        [results["pls"][k]["RMSE"] for k in split_keys],
        [results["pcr"][k]["RMSE"] for k in split_keys],
    ]

    x   = np.arange(len(split_labels))
    w   = 0.32
    fig, ax = plt.subplots(figsize=(max(7, len(split_labels) * 2.2), 5))
    colors = ["#2980b9", "#e67e22"]

    for i, (m, rm, col) in enumerate(zip(models, rmses, colors)):
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, rm, width=w, label=m, color=col, alpha=0.85)
        for bar, v in zip(bars, rm):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(split_labels, fontsize=11)
    ax.set_ylabel("Penicillin RMSE (g/L)", fontsize=10)
    ax.set_title(f"PLS vs PCA-PCR  (K={K})  —  sparse measurement points",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(csv_path: str = CSV_PATH, train_regime: str = "all90") -> None:
    """
    train_regime : "all90" → train on R1+R2+R3 (batches 1–90)
                   "R1"    → train on R1 only   (batches 1–30)
                   "R2"    → train on R2 only   (batches 31–60)
                   "R3"    → train on R3 only   (batches 61–90)
    """
    valid = {"all90", "R1", "R2", "R3"}
    if train_regime not in valid:
        raise ValueError(f"--train-regime must be one of {valid}, got {train_regime!r}")

    out_dir = OUTPUT_DIR if train_regime == "all90" else Path(f"./outputs/pls_pca_k64_{train_regime.lower()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    buckets = load_batches(csv_path)

    # Determine train set
    if train_regime == "all90":
        train_batches = buckets["R1"] + buckets["R2"] + buckets["R3"]
        # Evaluation splits: in-sample (all 90) + Fault OOD
        eval_splits = [
            ("train", "Train (1–90)",   train_batches),
            ("fault", "Fault (91–100)", buckets["Fault"]),
        ]
        ckpt_suffix = "all90"
    else:
        train_batches = buckets[train_regime]
        # Evaluation splits: in-sample + all other regimes as OOD
        eval_splits = [(train_regime, f"{train_regime} (in-sample)", train_batches)]
        for r in ("R1", "R2", "R3", "Fault"):
            if r != train_regime:
                lbl = f"Fault (91–100)" if r == "Fault" else f"{r} (OOD)"
                eval_splits.append((r, lbl, buckets[r]))
        ckpt_suffix = train_regime.lower()

    # Flat arrays for fitting (dense targets)
    X_train, y_train, _ = _flat(train_batches, sparse_only=False)
    print(f"\n[DATA]  Train regime={train_regime}  X_train={X_train.shape}  y_train={y_train.shape}")

    # ── 2. Fit models ──────────────────────────────────────────────────────────
    pls = fit_pls(X_train, y_train)
    pcr = PCRModel(k=K).fit(X_train, y_train)

    # ── 3. Evaluate ────────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  EVALUATION — sparse offline measurement points only")
    print(f"{'─'*55}")

    results: Dict = {"pls": {}, "pcr": {}}
    for model_name, predict_fn in [
        (f"PLS  (K={K})", lambda X: pls.predict(X).ravel()),
        (f"PCR  (K={K})", lambda X: pcr.predict(X)),
    ]:
        key = "pls" if "PLS" in model_name else "pcr"
        print(f"\n  {model_name}")
        for split_key, split_label, split_batches in eval_splits:
            m = evaluate_batches(predict_fn, split_batches, split_label, per_batch=True)
            results[key][split_key] = m

    # ── 4. Per-batch plots ─────────────────────────────────────────────────────
    for split_key, split_label, _ in eval_splits:
        plot_per_batch_rmse(
            results["pls"][split_key]["per_batch"],
            results["pcr"][split_key]["per_batch"],
            title=f"Per-batch Penicillin RMSE — {split_label}  (K={K})",
            save_path=str(out_dir / f"{split_key}_per_batch_rmse.png"),
        )

    split_keys   = [s[0] for s in eval_splits]
    split_labels = [s[1] for s in eval_splits]
    plot_summary_bar(results, split_keys, split_labels,
                     save_path=str(out_dir / "summary_bar.png"))

    # ── 5. Comparison table ────────────────────────────────────────────────────
    sep = "─" * 70
    print(f"\n{'='*70}")
    print(f"  PLS vs PCA-PCR  (K={K})  —  Penicillin RMSE (g/L)  [train={train_regime}]")
    print(f"{'='*70}")
    header = f"  {'Model':<20}" + "".join(f"  {lbl:>14}" for lbl in split_labels)
    print(header)
    print(sep)
    for label, key in [(f"PLS (K={K})", "pls"), (f"PCA-PCR (K={K})", "pcr")]:
        row = f"  {label:<20}"
        for sk in split_keys:
            row += f"  {results[key][sk]['RMSE']:>14.4f}"
        print(row)
    print(f"{'='*70}")

    # OOD delta vs first split (in-sample)
    first_key = split_keys[0]
    for sk, lbl in zip(split_keys[1:], split_labels[1:]):
        rt_pls = results["pls"][first_key]["RMSE"]
        rf_pls = results["pls"][sk]["RMSE"]
        ood_pls = (rf_pls - rt_pls) / (rt_pls + 1e-8) * 100
        rt_pcr = results["pcr"][first_key]["RMSE"]
        rf_pcr = results["pcr"][sk]["RMSE"]
        ood_pcr = (rf_pcr - rt_pcr) / (rt_pcr + 1e-8) * 100
        print(f"  OOD Δ vs {split_labels[0]} → {lbl}:  "
              f"PLS={ood_pls:+.1f}%  PCR={ood_pcr:+.1f}%")

    # ── 6. Save checkpoints ────────────────────────────────────────────────────
    pls_ckpt = CKPT_DIR / f"pls_k{K}_{ckpt_suffix}.joblib"
    pcr_ckpt = CKPT_DIR / f"pcr_k{K}_{ckpt_suffix}.joblib"
    joblib.dump(pls, pls_ckpt, compress=3)
    joblib.dump({"scaler": pcr.scaler, "pca": pcr.pca, "lr": pcr.lr}, pcr_ckpt, compress=3)
    print(f"\n  Checkpoints saved:")
    print(f"    {pls_ckpt}")
    print(f"    {pcr_ckpt}")
    print(f"  Plots in {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"PLS and PCA-PCR (K={K}) baselines with optional regime-ablation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", default=CSV_PATH, help="Path to IndPenSim CSV")
    parser.add_argument(
        "--train-regime",
        default="all90",
        choices=["all90", "R1", "R2", "R3"],
        help="Which regime to train on. 'all90' = R1+R2+R3 (original behaviour).",
    )
    args = parser.parse_args()
    main(csv_path=args.csv, train_regime=args.train_regime)
