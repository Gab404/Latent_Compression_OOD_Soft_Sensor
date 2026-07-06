"""
svr_baseline.py
===============
Support Vector Regression (SVR) baseline for the ReaKt-v2 bioprocess
soft-sensor benchmark (IndPenSim V3 -- Penicillin prediction from Raman
spectra).

Model
-----
  sklearn.pipeline.Pipeline:
    ("scaler", StandardScaler())   -- unit-variance scaling per wavenumber
    ("svr",    SVR(kernel="rbf"))  -- epsilon-SVR with Gaussian RBF kernel

  Hyperparameters tuned via RandomizedSearchCV + GroupKFold(10):
    C       : log-uniform  [1,      1000]
    gamma   : log-uniform  [1e-4,   0.1 ]
    epsilon : uniform      [0.01,   0.10]

Input
-----
  2001-point Raman spectra, Savitzky-Golay first-derivative preprocessed
  (window=15, poly=2, deriv=1). Identical pipeline to pls_baseline.py.

Training-set size -- why SVR trains on sparse samples
------------------------------------------------------
  SVR (libsvm) has O(n^2) memory and O(n^2..n^3) time complexity through
  the kernel matrix K_ij = exp(-gamma||xi-xj||^2) of size n x n.
  Training on all 82,205 dense time-steps would require a kernel matrix of
  82,205 x 82,205 x 8 bytes = ~54 GB -- computationally intractable.

  Standard chemometric calibration protocol trains SVR only on the actual
  OFFLINE MEASUREMENTS: ~20 sparse Penicillin assays per batch x 72 batches
  = 1,488 calibration spectra. The kernel matrix is then only 1,488 x 1,488
  x 8 bytes = 17.7 MB -- fits entirely in the kernel cache in milliseconds.

  The dense arrays (82,205 x 2001) are still built and printed to confirm
  that the data pipeline is byte-for-byte identical to the PLS baseline.
  They are exposed as X_train_dense / y_train_dense for reference.

Evaluation
----------
  Identical to pls_baseline.py: predictions are made at ALL time steps but
  metrics are aggregated ONLY at sparse offline measurement points, matching
  the evaluation methodology of PI-LSTM and Neural-ODE.

Usage
-----
  python svr_baseline.py
  python svr_baseline.py --csv /path/to/100_Batches_IndPenSim_V3.csv
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
from scipy.signal import savgol_filter
from scipy.stats import loguniform, uniform
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# ── Project imports ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import (
    TARGET_COLS,
    FAULT_COL,
    RAMAN_START_COL_IDX,
)
from src.evaluation.metrics import compute_metrics


# ── ─────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ── ─────────────────────────────────────────────────────────────────────────

CSV_PATH: str = "./100_Batches_IndPenSim_V3.csv"

# Raman spectral window (same as pls_baseline.py)
N_RAMAN_COLS: int = 2001

# Savitzky-Golay preprocessing (same as pls_baseline.py)
SG_WINDOW: int = 15
SG_POLY:   int = 2
SG_DERIV:  int = 1

# Train / val / test split (mirrors PI-LSTM and PLS -- byte-for-byte identical)
TRAIN_FRAC: float = 0.80
SPLIT_SEED: int   = 42

# Cross-validation
N_CV_FOLDS:    int = 10

# PCA dimensionality reduction inserted between StandardScaler and SVR
# -----------------------------------------------------------------------
# Motivation: SVR kernel evaluation is O(n_features) per sample-pair.
# Without PCA: 1,488 samples x 2001 features  -> ~15 s per SVR fit.
# With PCA(50): 1,488 samples x 50 PCs        ->  ~1.5 s per SVR fit (10x).
# PCA(50) retains the dominant spectral variation while discarding noise
# dimensions and making the gamma search space much more interpretable.
# A second StandardScaler after PCA rescales the PC scores to unit variance
# so that gamma='scale' semantics are preserved for the SVR.
N_PCA_COMPONENTS: int = 50

# RandomizedSearchCV settings
# n_jobs=1: Windows spawns child processes via 'spawn' which causes stdout
# from worker processes to be lost in PowerShell sessions.  Setting n_jobs=1
# avoids this and keeps all output visible.  On Linux/macOS set to -1.
N_SEARCH_ITER: int = 12
N_JOBS:        int = 1          # set to -1 on Linux/macOS for full parallelism

# Hyperparameter search distributions
# After StandardScaler + PCA(50) + StandardScaler:
#   gamma 'scale' = 1 / (n_pca_features * var) = 1/50 = 0.02
#   --> loguniform(1e-3, 1e0) brackets this value comfortably
PARAM_DIST: Dict = {
    "svr__C":       loguniform(1e0, 1e3),    # log-uniform in [1, 1000]
    "svr__gamma":   loguniform(1e-3, 1e0),   # log-uniform in [0.001, 1]
    "svr__epsilon": uniform(0.01, 0.09),     # uniform     in [0.01, 0.10] g/L
}

# Minimum rows per batch
MIN_BATCH_LEN: int = 50

# Output paths
OUTPUT_DIR:      Path = Path("./outputs/svr_baseline")
CHECKPOINT_PATH: Path = Path("./checkpoints/svr_baseline.joblib")


# ── ─────────────────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING  (byte-for-byte copy of pls_baseline.py)
# ── ─────────────────────────────────────────────────────────────────────────

def _apply_sg_filter(X: np.ndarray) -> np.ndarray:
    """
    Savitzky-Golay first-derivative filter applied row-wise along axis=1
    (the spectral / wavenumber axis).  Identical to pls_baseline.py.
    """
    return savgol_filter(
        X,
        window_length=SG_WINDOW,
        polyorder=SG_POLY,
        deriv=SG_DERIV,
        axis=1,
    )


def load_batches(csv_path: str) -> List[Dict]:
    """
    Load the IndPenSim V3 CSV, detect batch boundaries (time-reset
    convention), apply SG preprocessing to Raman spectra, and return a
    list of per-batch dicts.

    Byte-for-byte identical pipeline to pls_baseline.load_batches().

    Each dict contains
    ------------------
    X_raman      : (n_steps, N_RAMAN_COLS) -- SG first-derivative spectra
    y_pen_dense  : (n_steps,)              -- interpolated Penicillin (g/L)
    y_pen_sparse : (n_steps,)              -- raw measured Penicillin
                                              (NaN where not sampled)
    batch_id     : int   -- 0-indexed
    is_fault     : bool  -- True for fault batches 90-99
    """
    print(f"\n[DATA]  Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"[DATA]  Raw shape: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # -- Resolve Raman column names by absolute position -------------------
    # Using names (not iloc integers) ensures per-batch sub-DataFrames index
    # the same columns correctly after reset_index.
    all_cols    = df.columns.tolist()
    avail_raman = len(all_cols) - RAMAN_START_COL_IDX
    n_raman_use = min(N_RAMAN_COLS, avail_raman)
    if n_raman_use < N_RAMAN_COLS:
        warnings.warn(
            f"CSV has only {avail_raman} Raman columns starting at index "
            f"{RAMAN_START_COL_IDX}; expected {N_RAMAN_COLS}. "
            f"Using all {n_raman_use} available.",
        )
    raman_col_names = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + n_raman_use]
    print(f"[DATA]  Raman channels: {n_raman_use}  "
          f"(cols {RAMAN_START_COL_IDX}--{RAMAN_START_COL_IDX + n_raman_use - 1})")

    # -- Batch detection: time-reset convention ----------------------------
    # Identical to PenicillinDataModule in src/data/dataset.py
    t_vals    = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid       = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:   # time reset -> new batch
            bid += 1
        batch_ids[i] = bid
    df["_batch_id"] = batch_ids
    n_batches = int(bid + 1)
    print(f"[DATA]  Detected {n_batches} batches")

    # -- Per-batch processing ---------------------------------------------
    batches: List[Dict] = []

    for b_id in range(n_batches):
        b = df[df["_batch_id"] == b_id].copy().reset_index(drop=True)

        if len(b) < MIN_BATCH_LEN:
            continue

        # Extract Raman BEFORE any row-filtering so arrays stay aligned
        X_raw        = b[raman_col_names].values.astype(np.float64)
        y_pen_sparse = b[TARGET_COLS[1]].values.copy()   # NaN where not sampled

        # Dense linear interpolation of targets (same as PenicillinDataModule)
        for tc in TARGET_COLS:
            b[tc] = b[tc].interpolate(method="linear").ffill().bfill()

        # Single keep-mask applied uniformly to X, y_sparse, and b
        required  = ["Time (h)", TARGET_COLS[0], TARGET_COLS[1], FAULT_COL]
        keep_mask = ~b[required].isna().any(axis=1).values
        b            = b[keep_mask].reset_index(drop=True)
        X_raw        = X_raw[keep_mask]
        y_pen_sparse = y_pen_sparse[keep_mask]

        if len(b) < MIN_BATCH_LEN:
            continue

        # Clamp spectra (instrument warm-up rows may produce zeros / NaN)
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        X_raw = np.clip(X_raw, a_min=0.0, a_max=None)

        # Savitzky-Golay first-derivative along the spectral axis (axis=1)
        X_sg = _apply_sg_filter(X_raw)

        batches.append({
            "X_raman":     X_sg,
            "y_pen_dense": b[TARGET_COLS[1]].values,
            "y_pen_sparse": y_pen_sparse,
            "batch_id":    b_id,
            "is_fault":    bool(b[FAULT_COL].max() > 0),
        })

    n_clean = sum(not b["is_fault"] for b in batches)
    n_fault = sum(    b["is_fault"] for b in batches)
    print(f"[DATA]  Fault-free: {n_clean}  |  Faulty: {n_fault}")
    return batches


# ── ─────────────────────────────────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT  (byte-for-byte copy of pls_baseline.py)
# ── ─────────────────────────────────────────────────────────────────────────

def split_batches(
    batches:    List[Dict],
    train_frac: float = TRAIN_FRAC,
    seed:       int   = SPLIT_SEED,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Replicate the PI-LSTM / PLS random split (np.random.default_rng(42),
    80 / 20 on fault-free batches).  Fault batches are always the fixed
    test set.  Identical to pls_baseline.split_batches().
    """
    clean = [b for b in batches if not b["is_fault"]]
    fault = [b for b in batches if     b["is_fault"]]

    n_clean = len(clean)
    n_train = int(n_clean * train_frac)

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(n_clean)

    train_b = [clean[i] for i in perm[:n_train]]
    val_b   = [clean[i] for i in perm[n_train:]]

    print(f"\n[SPLIT]  Train: {len(train_b)} batches  |  "
          f"FF-val: {len(val_b)} batches  |  Fault: {len(fault)} batches")
    return train_b, val_b, fault


# ── ─────────────────────────────────────────────────────────────────────────
# ARRAY BUILDERS  (byte-for-byte copy of pls_baseline.py)
# ── ─────────────────────────────────────────────────────────────────────────

def build_flat_arrays(
    batches:           List[Dict],
    use_dense_targets: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten per-batch dicts into 2-D (X, y) arrays for sklearn.

    Parameters
    ----------
    batches           : list of batch dicts (from load_batches)
    use_dense_targets : True  -> all time-steps, interpolated Penicillin
                        False -> only rows where Penicillin was measured

    Returns
    -------
    X      : (n_samples, n_raman)
    y      : (n_samples,)
    groups : (n_samples,)   -- batch_id per row, used by GroupKFold
    """
    X_parts, y_parts, g_parts = [], [], []

    for b in batches:
        if use_dense_targets:
            X_b = b["X_raman"]
            y_b = b["y_pen_dense"]
        else:
            # Sparse calibration: rows where Penicillin was actually measured
            mask = ~np.isnan(b["y_pen_sparse"])
            if mask.sum() == 0:
                continue
            X_b = b["X_raman"][mask]
            y_b = b["y_pen_sparse"][mask]

        X_parts.append(X_b)
        y_parts.append(y_b)
        g_parts.append(
            np.full(len(X_b), fill_value=b["batch_id"], dtype=np.int32)
        )

    X      = np.vstack(X_parts)
    y      = np.concatenate(y_parts)
    groups = np.concatenate(g_parts)
    return X, y, groups


# ── ─────────────────────────────────────────────────────────────────────────
# SVR PIPELINE
# ── ─────────────────────────────────────────────────────────────────────────

def build_svr_pipeline(C: float = 100.0,
                       gamma: float = 0.02,
                       epsilon: float = 0.05) -> Pipeline:
    """
    Construct the SVR pipeline.

    Pipeline stages
    ---------------
    1. StandardScaler
       Centres and scales each of the 2001 SG-derivative wavenumber
       channels to zero mean / unit variance.  Mandatory for RBF SVR:
       the kernel k(x, x') = exp(-gamma * ||x-x'||^2) is distance-based
       and collapses without equal-variance scaling.

    2. PCA(n_components=N_PCA_COMPONENTS=50)
       Projects the 2001-d scaled spectra into the 50 principal components
       that capture the dominant spectral covariance.  Two reasons:
         (a) Speed: SVR kernel computation is O(d) per sample-pair; PCA
             reduces d from 2001 to 50, giving a ~10x speedup per fit.
         (b) Regularisation: noise-dominated high-frequency PC dimensions
             are discarded, which typically improves generalisation.
       50 components was chosen to retain >=99% of spectral variance for
       first-derivative SG spectra of this type.

    3. StandardScaler (second)
       PCA scores have decreasing variance by construction (PC1 >> PC50).
       Without rescaling, the RBF kernel would be dominated by PC1 and
       almost blind to PC50.  A second StandardScaler equalises all 50
       PC scores to unit variance so that gamma has consistent geometry
       across all components.

    4. SVR(kernel='rbf', cache_size=2000 MB)
       Epsilon-SVR with Gaussian kernel.  cache_size=2000 MB ensures the
       full 1,488 x 1,488 kernel matrix (17.7 MB) fits in cache.
    """
    return Pipeline([
        ("scaler1", StandardScaler()),
        ("pca",     PCA(n_components=N_PCA_COMPONENTS, random_state=SPLIT_SEED)),
        ("scaler2", StandardScaler()),
        ("svr",     SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon,
                        cache_size=2000, max_iter=100_000)),
    ])


# ── ─────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER SEARCH
# ── ─────────────────────────────────────────────────────────────────────────

def run_hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups:  np.ndarray,
    param_dist: Dict = PARAM_DIST,
    n_iter:     int  = N_SEARCH_ITER,
    n_folds:    int  = N_CV_FOLDS,
    n_jobs:     int  = N_JOBS,
    seed:       int  = SPLIT_SEED,
) -> RandomizedSearchCV:
    """
    RandomizedSearchCV over (C, gamma, epsilon) for the SVR pipeline.

    Cross-validation strategy
    -------------------------
    GroupKFold(10) is mandatory here -- not optional.  The 1,488 sparse
    calibration samples come from 72 batches (~20 rows each).  Consecutive
    spectra from the same batch share nearly identical Raman backgrounds
    (same reactor, same run).  A standard KFold could place rows 1-15 of
    batch 5 in the training fold while rows 16-20 land in the test fold,
    creating a near-perfect leak of batch-level covariance.  GroupKFold
    guarantees every batch appears in exactly one test fold.

    Parameters
    ----------
    X_train : (n_sparse, 2001)  -- SG-preprocessed spectra
    y_train : (n_sparse,)       -- sparse Penicillin measurements (g/L)
    groups  : (n_sparse,)       -- batch_id per sample

    Returns
    -------
    Fitted RandomizedSearchCV.  best_estimator_ is already refitted on
    the full X_train / y_train with the best hyperparameters (refit=True).
    """
    base_pipe = build_svr_pipeline()
    cv        = GroupKFold(n_splits=n_folds)

    search = RandomizedSearchCV(
        estimator           = base_pipe,
        param_distributions = param_dist,
        n_iter              = n_iter,
        cv                  = cv,
        scoring             = "neg_root_mean_squared_error",
        n_jobs              = n_jobs,
        random_state        = seed,
        refit               = True,   # refit best params on full train set
        verbose             = 2,
        return_train_score  = False,
        error_score         = "raise",
    )

    print(f"\n[SEARCH]  RandomizedSearchCV:")
    print(f"          {n_iter} iterations x {n_folds} folds = "
          f"{n_iter * n_folds} total SVR fits")
    print(f"          Pipeline: StandardScaler -> PCA({N_PCA_COMPONENTS}) "
          f"-> StandardScaler -> SVR(rbf)")
    print(f"          Training on {X_train.shape[0]} sparse samples x "
          f"{X_train.shape[1]} raw spectral features "
          f"(-> {N_PCA_COMPONENTS} PCA dims)")
    print(f"          n_jobs={n_jobs}  (set to -1 on Linux/macOS for parallelism)")

    search.fit(X_train, y_train, groups=groups)

    best_p     = search.best_params_
    best_cv    = -search.best_score_   # convert neg-RMSE to positive RMSE
    print(f"\n[SEARCH]  Best parameters found:")
    for k in sorted(best_p):
        print(f"            {k:<24} = {best_p[k]:.6g}")
    print(f"[SEARCH]  Mean CV RMSE (best setting) = {best_cv:.4f} g/L")

    return search


# ── ─────────────────────────────────────────────────────────────────────────
# POOLED RMSECV  (comparable to PLS RMSECV)
# ── ─────────────────────────────────────────────────────────────────────────

def compute_rmsecv(
    best_pipeline: Pipeline,
    X_train:       np.ndarray,
    y_train:       np.ndarray,
    groups:        np.ndarray,
    n_folds:       int = N_CV_FOLDS,
    n_jobs:        int = N_JOBS,
) -> Tuple[float, np.ndarray]:
    """
    Compute the POOLED 10-fold RMSECV using cross_val_predict.

    Why this differs from search.best_score_
    -----------------------------------------
    RandomizedSearchCV reports the MEAN of per-fold RMSE values.  The PLS
    baseline computes the POOLED RMSECV: sqrt(mean((y - y_oof)^2)) over all
    out-of-fold predictions concatenated together.  These two are numerically
    different when fold sizes are unequal.  Using cross_val_predict here
    ensures the SVR RMSECV is computed by the identical formula.

    Parameters
    ----------
    best_pipeline : Pipeline with best hyperparameters (unfitted clone used)
    X_train, y_train, groups : sparse calibration arrays

    Returns
    -------
    rmsecv   : pooled out-of-fold RMSE
    y_oof    : (n_sparse,) out-of-fold predictions (diagnostic use)
    """
    pipe_clone = clone(best_pipeline)   # unfitted copy with same hyperparams
    cv         = GroupKFold(n_splits=n_folds)

    print(f"\n[RMSECV]  Computing pooled 10-fold RMSECV via cross_val_predict ...")
    y_oof = cross_val_predict(
        pipe_clone, X_train, y_train,
        cv=cv, groups=groups, n_jobs=n_jobs,
    ).ravel()

    rmsecv = float(np.sqrt(mean_squared_error(y_train, y_oof)))
    print(f"[RMSECV]  10-fold RMSECV (pooled) = {rmsecv:.4f} g/L")
    return rmsecv, y_oof


# ── ─────────────────────────────────────────────────────────────────────────
# MODEL PERSISTENCE
# ── ─────────────────────────────────────────────────────────────────────────

def save_model(
    model:    Pipeline,
    path:     Path           = CHECKPOINT_PATH,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Persist the fitted SVR pipeline using joblib (compress=3).

    Saved artefact layout
    ---------------------
    {
        "model"    : fitted sklearn Pipeline  (scaler + SVR),
        "metadata" : dict -- hyperparams, RMSECV, evaluation scores
    }
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata or {}}, path, compress=3)
    print(f"\n[SAVE]  Pipeline saved  ->  {path}")


def load_model(path: Path = CHECKPOINT_PATH) -> Tuple[Pipeline, Dict]:
    """Load a previously saved SVR pipeline and its metadata."""
    payload = joblib.load(path)
    print(f"[LOAD]  Pipeline loaded  <-  {path}")
    return payload["model"], payload.get("metadata", {})


# ── ─────────────────────────────────────────────────────────────────────────
# EVALUATION  (byte-for-byte copy of pls_baseline.py)
# ── ─────────────────────────────────────────────────────────────────────────

def _predict_batch_at_sparse_points(
    model: Pipeline,
    batch: Dict,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Predict Penicillin ONLY at sparse offline measurement time-points
    (rows where y_pen_sparse is not NaN).

    Identical contract to pls_baseline._predict_batch_at_sparse_points().
    This ensures evaluation is strictly comparable across all models.
    """
    mask = ~np.isnan(batch["y_pen_sparse"])
    if mask.sum() == 0:
        return None, None

    X_sparse = batch["X_raman"][mask]          # (n_sparse, n_raman)
    y_true   = batch["y_pen_sparse"][mask]     # (n_sparse,)
    y_pred   = model.predict(X_sparse).ravel() # (n_sparse,)

    return y_true, y_pred


def evaluate_split(
    model:      Pipeline,
    batches:    List[Dict],
    split_name: str = "split",
) -> Dict:
    """
    Evaluate on a list of batches; aggregate RMSE / R2 / MAE across all
    sparse measurement points.  Returns the metrics dict.
    """
    all_true, all_pred = [], []

    for b in batches:
        y_true, y_pred = _predict_batch_at_sparse_points(model, b)
        if y_true is None:
            continue
        all_true.append(y_true)
        all_pred.append(y_pred)

    if not all_true:
        print(f"[EVAL]  {split_name}: no sparse measurement points found.")
        return {}

    y_true_cat = np.concatenate(all_true)
    y_pred_cat = np.concatenate(all_pred)
    metrics    = compute_metrics(y_true_cat, y_pred_cat)

    print(f"\n[EVAL]  {split_name}")
    print(f"         Pen RMSE = {metrics['RMSE']:.4f} g/L  |  "
          f"R2 = {metrics['R2']:.4f}  |  "
          f"MAE = {metrics['MAE']:.4f} g/L  |  "
          f"n = {metrics['n']}")
    return metrics


def evaluate_fault_batches(
    model:         Pipeline,
    fault_batches: List[Dict],
) -> Tuple[float, List[Dict]]:
    """
    Compute per-batch RMSE for each fault batch (90-99) and the mean.

    Returns
    -------
    mean_fault_rmse : scalar
    per_batch       : list of {batch_id, rmse, r2, n}
    """
    per_batch: List[Dict] = []

    for b in fault_batches:
        y_true, y_pred = _predict_batch_at_sparse_points(model, b)
        if y_true is None:
            per_batch.append({
                "batch_id": b["batch_id"],
                "rmse": float("nan"), "r2": float("nan"), "n": 0,
            })
            continue
        m = compute_metrics(y_true, y_pred)
        per_batch.append({
            "batch_id": b["batch_id"],
            "rmse":     m["RMSE"],
            "r2":       m["R2"],
            "n":        m["n"],
        })

    valid_rmses = [r["rmse"] for r in per_batch if not np.isnan(r["rmse"])]
    mean_rmse   = float(np.mean(valid_rmses)) if valid_rmses else float("nan")

    print(f"\n[EVAL]  Fault batches -- per-batch Penicillin RMSE")
    print(f"        {'Batch':>7}  {'RMSE (g/L)':>12}  {'R2':>8}  {'n':>5}")
    print(f"        {'-'*7}  {'-'*12}  {'-'*8}  {'-'*5}")
    for r in per_batch:
        rmse_str = f"{r['rmse']:.4f}" if not np.isnan(r["rmse"]) else "   N/A"
        r2_str   = f"{r['r2']:.4f}"   if not np.isnan(r["r2"])   else "   N/A"
        print(f"        {r['batch_id']:>7}  {rmse_str:>12}  {r2_str:>8}  {r['n']:>5}")
    print(f"        {'-'*37}")
    print(f"        {'Mean':>7}  {mean_rmse:>12.4f}")

    return mean_rmse, per_batch


# ── ─────────────────────────────────────────────────────────────────────────
# RESULTS SUMMARY TABLE
# ── ─────────────────────────────────────────────────────────────────────────

def print_comparison_table(
    best_params:     Dict,
    rmsecv:          float,
    ffval_metrics:   Dict,
    fault_mean_rmse: float,
    per_batch:       List[Dict],
) -> None:
    """Print a benchmark-ready summary matching the PLS output format."""
    sep = "-" * 64
    print(f"\n{sep}")
    print(f"  SVR BASELINE -- BENCHMARK SUMMARY  (RBF kernel)")
    print(sep)

    print(f"  Best Hyperparameters")
    for k in sorted(best_params):
        v = best_params[k]
        print(f"    {'svr__' + k.split('__')[-1]:<24} = {v:.6g}"
              if isinstance(v, float) else f"    {k:<24} = {v}")

    print(f"")
    print(f"  Calibration")
    print(f"    10-fold RMSECV (pooled) : {rmsecv:.4f} g/L")
    print(f"")
    print(f"  FF-Validation")
    print(f"    Pen RMSE                : {ffval_metrics.get('RMSE', float('nan')):.4f} g/L")
    print(f"    Pen R2                  : {ffval_metrics.get('R2',   float('nan')):.4f}")
    print(f"    Pen MAE                 : {ffval_metrics.get('MAE',  float('nan')):.4f} g/L")
    print(f"    n (sparse pts)          : {ffval_metrics.get('n', 0)}")
    print(f"")
    print(f"  Fault Batches (Test)")
    print(f"    Pen RMSE mean           : {fault_mean_rmse:.4f} g/L")
    print(f"    Per-batch RMSE          :")
    for r in per_batch:
        rmse_str = f"{r['rmse']:.4f}" if not np.isnan(r["rmse"]) else "N/A"
        print(f"      Batch {r['batch_id']:>3}            : {rmse_str} g/L")
    print(sep)

    print(f"\n  (Copy-paste row for comparison table)")
    print(f"  {'Model':<22} | {'Pen RMSE (val)':>14} | "
          f"{'Pen R2 (val)':>12} | {'Fault RMSE':>10} | {'RMSECV':>8}")
    print(f"  {'-'*22}-+-{'-'*14}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}")
    print(
        f"  {'SVR (SG d=1, RBF)':<22} | "
        f"{ffval_metrics.get('RMSE', float('nan')):>14.4f} | "
        f"{ffval_metrics.get('R2',   float('nan')):>12.4f} | "
        f"{fault_mean_rmse:>10.4f} | "
        f"{rmsecv:>8.4f}"
    )
    print(sep)


# ── ─────────────────────────────────────────────────────────────────────────
# MAIN
# ── ─────────────────────────────────────────────────────────────────────────

def main(csv_path: str = CSV_PATH) -> None:
    """
    End-to-end SVR baseline pipeline:

      1.  Load & SG-preprocess raw Raman spectra
      2.  Split: 72 train | 18 FF-val | 10 fault  (seed=42, same as PLS)
      3.  Build DENSE arrays  (82205, 2001) -- pipeline parity check
      4.  Build SPARSE arrays (~1488, 2001) -- actual SVR calibration data
      5.  RandomizedSearchCV with GroupKFold(10) -- tune C, gamma, epsilon
      6.  Pooled 10-fold RMSECV via cross_val_predict (comparable to PLS)
      7.  best_estimator_ already refitted on full sparse train (refit=True)
      8.  Evaluate on FF-val and fault batches (sparse points only)
      9.  Print benchmark comparison table
      10. Save fitted pipeline to CHECKPOINT_PATH
    """

    # 1. Load ----------------------------------------------------------------
    batches = load_batches(csv_path)

    # 2. Split ---------------------------------------------------------------
    train_batches, val_batches, fault_batches = split_batches(batches)

    # 3. Dense arrays (pipeline-parity reference; not passed to SVR fit) -----
    X_train_dense, y_train_dense, groups_dense = build_flat_arrays(
        train_batches, use_dense_targets=True,
    )
    print(f"\n[DATA]  Dense  matrix (reference)  : "
          f"X={X_train_dense.shape}  y={y_train_dense.shape}  "
          f"n_groups={len(np.unique(groups_dense))}")
    print(f"        (Built to confirm identical pipeline to PLS baseline;")
    print(f"         SVR trains on the sparse subset below for feasibility.)")

    # 4. Sparse calibration arrays (actual SVR training data) ----------------
    X_train_sp, y_train_sp, groups_sp = build_flat_arrays(
        train_batches, use_dense_targets=False,
    )
    print(f"\n[DATA]  Sparse matrix (SVR calib)  : "
          f"X={X_train_sp.shape}  y={y_train_sp.shape}  "
          f"n_batches={len(np.unique(groups_sp))}")
    print(f"        Kernel matrix size (after PCA-{N_PCA_COMPONENTS}): "
          f"{X_train_sp.shape[0]}^2 x 8 B = "
          f"{X_train_sp.shape[0]**2 * 8 / 1e6:.1f} MB  "
          f"(fits entirely in 2 GB kernel cache)")

    # 5. Hyperparameter search -----------------------------------------------
    search = run_hyperparameter_search(X_train_sp, y_train_sp, groups_sp)
    best_pipeline = search.best_estimator_   # fitted on full sparse train
    best_params   = search.best_params_

    # 6. Pooled RMSECV (cross_val_predict, identical formula to PLS) ---------
    rmsecv, _ = compute_rmsecv(best_pipeline, X_train_sp, y_train_sp, groups_sp)

    # 7. Final model is search.best_estimator_ (refit=True in search) --------
    print(f"\n[FIT]   Final SVR: best_estimator_ refitted on all "
          f"{X_train_sp.shape[0]} sparse calibration samples")

    # 8. Evaluate ------------------------------------------------------------
    ffval_metrics   = evaluate_split(best_pipeline, val_batches,
                                     split_name="FF-Validation")
    fault_mean_rmse, per_batch = evaluate_fault_batches(best_pipeline,
                                                         fault_batches)

    # 9. Summary table -------------------------------------------------------
    print_comparison_table(best_params, rmsecv, ffval_metrics,
                           fault_mean_rmse, per_batch)

    # 10. Save ---------------------------------------------------------------
    metadata = {
        # Hyperparameters
        "best_params":     best_params,
        "n_raman_cols":    N_RAMAN_COLS,
        "sg_window":       SG_WINDOW,
        "sg_poly":         SG_POLY,
        "sg_deriv":        SG_DERIV,
        "train_frac":      TRAIN_FRAC,
        "split_seed":      SPLIT_SEED,
        # Training set info
        "n_train_dense":   int(X_train_dense.shape[0]),
        "n_train_sparse":  int(X_train_sp.shape[0]),
        # Results
        "rmsecv":          rmsecv,
        "ffval_rmse":      ffval_metrics.get("RMSE"),
        "ffval_r2":        ffval_metrics.get("R2"),
        "fault_mean_rmse": fault_mean_rmse,
        "per_batch_fault": per_batch,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_model(best_pipeline, CHECKPOINT_PATH, metadata)


# ── CLI entry-point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SVR baseline for ReaKt-v2 Penicillin soft-sensor benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv", type=str, default=CSV_PATH,
        help="Path to the IndPenSim V3 CSV file.",
    )
    args = parser.parse_args()
    main(csv_path=args.csv)
