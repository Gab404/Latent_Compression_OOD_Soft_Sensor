#!/usr/bin/env python3
"""
Baseline Temporelle — Régression Polynomiale (degré 3)
======================================================
Trois modèles comparés sur les mêmes données :

  Modèle A (Oracle)    : temps normalisé [0,1] par batch (nécessite durée totale)
  Modèle B (Réaliste)  : temps brut écoulé (capteur en ligne)
  Modèle C (Steelman)  : trajectoire moyenne empirique, non paramétrique

Évaluation en deux modes :
  ALL    : toutes les lignes (penicillin online, 11 525 pts)
  SPARSE : uniquement les points de mesure offline (~200 pts)
           → identique au protocole β-VAE-LSTM pour comparaison directe

Usage
-----
    PYTHONPATH=. python3 scripts/baseline_temporal_poly.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

CSV_PATH = _ROOT / "100_Batches_IndPenSim_V3.csv"
DEGREE = 3
MEAN_TRAJ_GRID = 200     # number of bins for the mean-trajectory baseline


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_indpensim(csv_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load IndPenSim V3 CSV → (df_train, df_test).

    Columns returned:
        batch_id, time, penicillin           (online sensor, every row)
        penicillin_sparse                    (offline lab, NaN where unmeasured)

    Train = fault-free batches (Regimes 1–3)
    Test  = faulty batches (Regime 4)
    """
    from src.data.dataset import FAULT_COL, TARGET_COLS

    PEN_ONLINE  = "Penicillin concentration(P:g/L)"
    PEN_OFFLINE = TARGET_COLS[1]   # "Offline Penicillin concentration(P_offline:...)"

    print(f"Loading {csv_path.name} ...")
    df = pd.read_csv(
        csv_path,
        usecols=["Time (h)", FAULT_COL, PEN_ONLINE, PEN_OFFLINE],
    )
    print(f"  Raw shape: {df.shape[0]:,} rows")

    # Detect batch boundaries (time resets)
    t = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t[i] < t[i - 1]:
            bid += 1
        batch_ids[i] = bid
    df["batch_id"] = batch_ids

    # Rename columns
    df = df.rename(columns={
        "Time (h)": "time",
        PEN_ONLINE: "penicillin",
        PEN_OFFLINE: "penicillin_sparse",
    })

    # Offline measurements: 0 means "not measured" → NaN
    df.loc[df["penicillin_sparse"] <= 0, "penicillin_sparse"] = np.nan

    # Identify faulty batches
    fault = df[FAULT_COL].values
    faulty_batch_ids = set()
    for b in range(bid + 1):
        mask = batch_ids == b
        if fault[mask].max() > 0:
            faulty_batch_ids.add(b)

    df = df[["batch_id", "time", "penicillin", "penicillin_sparse"]].copy()
    df = df.dropna(subset=["penicillin"]).reset_index(drop=True)

    # Split
    df_train = df[~df["batch_id"].isin(faulty_batch_ids)].reset_index(drop=True)
    df_test = df[df["batch_id"].isin(faulty_batch_ids)].reset_index(drop=True)

    n_sparse_test = df_test["penicillin_sparse"].notna().sum()
    print(f"  Train : {len(df_train):>7,} rows, {df_train['batch_id'].nunique()} batches")
    print(f"  Test  : {len(df_test):>7,} rows, {df_test['batch_id'].nunique()} batches"
          f"  ({n_sparse_test} sparse offline points)")

    return df_train, df_test


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def add_normalized_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'time_norm' column: min-max scaled to [0, 1] per batch."""
    df = df.copy()
    df["time_norm"] = df.groupby("batch_id")["time"].transform(
        lambda t: (t - t.min()) / (t.max() - t.min() + 1e-12)
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL BUILDING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def build_poly_pipeline(degree: int = 3, scale_input: bool = False):
    """Polynomial regression pipeline."""
    steps = []
    if scale_input:
        steps.append(StandardScaler())
    steps.append(PolynomialFeatures(degree=degree, include_bias=False))
    steps.append(LinearRegression())
    return make_pipeline(*steps)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE with guard for empty arrays."""
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_and_evaluate(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_col: str,
    degree: int = 3,
    scale_input: bool = False,
) -> dict:
    """
    Train polynomial model. Return predictions and RMSE on ALL rows
    and on SPARSE-only rows.
    """
    X_tr = df_train[[feature_col]].values
    y_tr = df_train["penicillin"].values
    X_te = df_test[[feature_col]].values
    y_te = df_test["penicillin"].values

    model = build_poly_pipeline(degree=degree, scale_input=scale_input)
    model.fit(X_tr, y_tr)

    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)

    # Sparse-only evaluation (matches β-VAE-LSTM protocol)
    sp_mask_te = df_test["penicillin_sparse"].notna().values
    sp_mask_tr = df_train["penicillin_sparse"].notna().values

    return {
        "model": model,
        "rmse_train_all":    compute_rmse(y_tr, pred_tr),
        "rmse_test_all":     compute_rmse(y_te, pred_te),
        "rmse_train_sparse": compute_rmse(y_tr[sp_mask_tr], pred_tr[sp_mask_tr]),
        "rmse_test_sparse":  compute_rmse(y_te[sp_mask_te], pred_te[sp_mask_te]),
        "n_test_all":        len(y_te),
        "n_test_sparse":     int(sp_mask_te.sum()),
        "n_train_sparse":    int(sp_mask_tr.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. STEELMAN — Mean Trajectory Baseline (non-parametric)
# ─────────────────────────────────────────────────────────────────────────────

def mean_trajectory_baseline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    grid: int = MEAN_TRAJ_GRID,
) -> dict:
    """
    Best possible time-only predictor: empirical mean penicillin at each
    normalised-time bin, learned from training batches.
    """
    def binned(df):
        d = add_normalized_time(df)
        d["bin"] = (d["time_norm"] * (grid - 1)).round().astype(int).clip(0, grid - 1)
        return d

    tr = binned(df_train)
    te = binned(df_test)

    # Build look-up table from training data
    lut = tr.groupby("bin")["penicillin"].mean()
    lut = lut.reindex(range(grid)).interpolate().bfill().ffill()

    pred_tr = tr["bin"].map(lut).values
    pred_te = te["bin"].map(lut).values

    sp_mask_te = df_test["penicillin_sparse"].notna().values
    sp_mask_tr = df_train["penicillin_sparse"].notna().values

    return {
        "rmse_train_all":    compute_rmse(tr["penicillin"].values, pred_tr),
        "rmse_test_all":     compute_rmse(te["penicillin"].values, pred_te),
        "rmse_train_sparse": compute_rmse(
            tr.loc[sp_mask_tr, "penicillin"].values, pred_tr[sp_mask_tr]),
        "rmse_test_sparse":  compute_rmse(
            te.loc[sp_mask_te, "penicillin"].values, pred_te[sp_mask_te]),
        "n_test_all":        len(te),
        "n_test_sparse":     int(sp_mask_te.sum()),
        "n_train_sparse":    int(sp_mask_tr.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. UNIT VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def print_unit_check(df_test: pd.DataFrame):
    """
    Print y_true statistics so we can confirm units are g/L.
    If max ≈ 40 → g/L (real scale).  If max ≈ 3 → standardised.
    """
    y = df_test["penicillin"].values
    sp = df_test["penicillin_sparse"].dropna().values

    print(f"\n{'═' * 60}")
    print(f"  UNIT VERIFICATION (test set)")
    print(f"{'═' * 60}")
    print(f"  Online  penicillin (all rows):  "
          f"min={y.min():.3f}  max={y.max():.3f}  "
          f"std={y.std():.3f}  mean={y.mean():.3f}")
    print(f"  Offline penicillin (sparse):    "
          f"min={sp.min():.3f}  max={sp.max():.3f}  "
          f"std={sp.std():.3f}  mean={sp.mean():.3f}  "
          f"n={len(sp)}")

    if y.max() > 10:
        print(f"  → Max ≈ {y.max():.0f} → CONFIRMED g/L (raw scale)")
    else:
        print(f"  ⚠ Max ≈ {y.max():.1f} → likely STANDARDISED, "
              f"multiply RMSE by scaler.scale_ to get g/L")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    df_train, df_test = load_indpensim(CSV_PATH)

    # Normalised time for Models A & C
    df_train = add_normalized_time(df_train)
    df_test = add_normalized_time(df_test)

    # ── Unit check ────────────────────────────────────────────────────────────
    print_unit_check(df_test)

    # ── Model A: Oracle (normalised time) ─────────────────────────────────────
    res_a = train_and_evaluate(
        df_train, df_test,
        feature_col="time_norm",
        degree=DEGREE, scale_input=False,
    )

    # ── Model B: Réaliste (raw elapsed time) ──────────────────────────────────
    res_b = train_and_evaluate(
        df_train, df_test,
        feature_col="time",
        degree=DEGREE, scale_input=True,
    )

    # ── Model C: Steelman (mean trajectory) ───────────────────────────────────
    res_c = mean_trajectory_baseline(df_train, df_test, grid=MEAN_TRAJ_GRID)

    # ── Results table ─────────────────────────────────────────────────────────
    results = {
        "A — Oracle (t_norm)":      res_a,
        "B — Réaliste (t_raw)":     res_b,
        "C — Steelman (mean traj)": res_c,
    }

    print(f"\n{'═' * 78}")
    print(f"  BASELINE TEMPORELLE — Poly Degree {DEGREE}  |  Grid={MEAN_TRAJ_GRID}")
    print(f"{'═' * 78}")

    # --- ALL rows ---
    print(f"\n  Evaluation on ALL rows (online penicillin)")
    print(f"{'─' * 78}")
    print(f"  {'Model':<28s} {'Train RMSE':>12s} {'Test RMSE':>12s} {'n_test':>8s}")
    print(f"{'─' * 78}")
    for name, r in results.items():
        print(f"  {name:<28s} {r['rmse_train_all']:>10.4f} g/L"
              f" {r['rmse_test_all']:>10.4f} g/L"
              f" {r['n_test_all']:>8,}")
    print(f"{'─' * 78}")

    # --- SPARSE rows only (β-VAE-LSTM protocol) ---
    print(f"\n  Evaluation on SPARSE points only (offline lab measurements)")
    print(f"  ⚠  This is the protocol used by β-VAE-LSTM for fair comparison")
    print(f"{'─' * 78}")
    print(f"  {'Model':<28s} {'Train RMSE':>12s} {'Test RMSE':>12s} {'n_test':>8s}")
    print(f"{'─' * 78}")
    for name, r in results.items():
        print(f"  {name:<28s} {r['rmse_train_sparse']:>10.4f} g/L"
              f" {r['rmse_test_sparse']:>10.4f} g/L"
              f" {r['n_test_sparse']:>8,}")
    print(f"{'─' * 78}")

    # --- Index alignment warning ---
    n_all = res_a["n_test_all"]
    n_sp  = res_a["n_test_sparse"]
    print(f"\n  INDEX ALIGNMENT CHECK")
    print(f"  Baseline evaluated on : {n_all:,} rows (ALL) / {n_sp} rows (SPARSE)")
    print(f"  β-VAE-LSTM evaluates on ~{n_sp} sparse points (same protocol)")
    if n_all != n_sp:
        print(f"  ⚠  Factor {n_all / max(n_sp, 1):.0f}× difference — "
              f"use SPARSE RMSE for fair comparison with β-VAE-LSTM")

    # ── Poly coefficients ─────────────────────────────────────────────────────
    print(f"\n{'─' * 78}")
    print("  Model A coefficients (Oracle, time_norm ∈ [0,1]):")
    _print_coeffs(res_a["model"], scaled=False)
    print(f"\n  Model B coefficients (Réaliste, StandardScaled time):")
    _print_coeffs(res_b["model"], scaled=True)
    print()


def _print_coeffs(pipeline, scaled: bool):
    """Print polynomial coefficients from a fitted pipeline."""
    lr   = pipeline[-1]
    poly = pipeline[1] if scaled else pipeline[0]
    names = poly.get_feature_names_out(["t"])
    for name, coef in zip(names, lr.coef_):
        print(f"    {name:>6s}: {coef:+.6f}")
    print(f"    {'bias':>6s}: {lr.intercept_:+.6f}")


if __name__ == "__main__":
    main()
