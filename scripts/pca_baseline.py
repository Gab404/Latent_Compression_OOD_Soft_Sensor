"""
pca_baseline.py
===============
Static Principal Component Regression (PCR) baseline for the ReaKt-v2
bioprocess soft-sensor benchmark (IndPenSim V3 -- Penicillin prediction
from Raman spectra).

Model
-----
  sklearn.pipeline.Pipeline:
    ("scaler", StandardScaler())                    -- unit-variance per channel
    ("pca",    PCA(n_components=4, random_state=42)) -- unsupervised reduction
    ("reg",    LinearRegression())                  -- ordinary least squares

  This is the textbook "PCA baseline" in chemometrics (Principal Component
  Regression), i.e. the direct counterpart of PLS where the regression is
  performed in an UNSUPERVISED PCA basis instead of PLS's supervised latent
  directions.  Reporting PCR alongside PLS isolates the value of having
  PLS supervise the latent directions for the prediction target.

Input
-----
  2001-point Raman spectra, Savitzky-Golay first-derivative preprocessed
  (SG d=1, window=15, poly=2).  Each time step is treated as an independent
  sample  ->  X shape: [n_samples, N_RAMAN_COLS].

  This pipeline is byte-for-byte identical to pls_baseline.py and
  svr_baseline.py, ensuring strictly apples-to-apples comparison.

Target
------
  Interpolated Penicillin concentration (g/L).

Evaluation
----------
  Metrics are computed at the sparse offline measurement time-points only
  (wherever ``_penicillin_sparse`` is non-NaN), which exactly matches the
  evaluation methodology used for PLS, SVR, CDAE-(PI-)LSTM in this benchmark.

  Reported metrics
  ~~~~~~~~~~~~~~~~
  * 10-fold RMSECV  (GroupKFold by batch_id -- simulates Leave-One-Batch-Out)
  * FF-val  RMSE  (g/L)
  * FF-val  R2
  * Fault-batch  RMSE  mean  (g/L)
  * Per-batch RMSE  for fault batches 90-99

Side effect
-----------
  In addition to the static PCR Pipeline (saved to ``./checkpoints/
  pca_baseline.joblib``), this script also persists the fitted StandardScaler
  and PCA as standalone artefacts so they can be reused as a frozen Raman
  encoder by PCA-PI-LSTM (mirroring the cdae_best.pt + cdae_scaler.joblib
  pattern used by CDAERamanEncoderV2):

      ./checkpoints/pca_best.joblib    -- fitted sklearn.decomposition.PCA
      ./checkpoints/pca_scaler.joblib  -- fitted StandardScaler

  These are loaded by ``src/data/pca_encoder.PCARamanEncoderV1``.

Usage
-----
  python pca_baseline.py                          # uses defaults below
  python pca_baseline.py --csv path/to/data.csv
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -- Project imports (add repo root to sys.path when running as a script) -----
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import (
    TARGET_COLS,
    FAULT_COL,
    RAMAN_START_COL_IDX,
    RAMAN_N_COLS,
)
from src.evaluation.metrics import compute_metrics

# -- ------------------------------------------------------------------------
# CONFIGURATION  -- edit these constants to reproduce or modify the experiment
# -- ------------------------------------------------------------------------

CSV_PATH: str = "./100_Batches_IndPenSim_V3.csv"

# PCR hyperparameters (matched to PLS for direct comparison)
N_COMPONENTS: int = 4       # principal components (same K as PLS baseline)

# Raman spectral window
# The benchmark uses the first 2001 wavenumber channels (400-2400 cm^-1 @1 cm^-1
# resolution).  RAMAN_N_COLS in dataset.py is 2200; we deliberately cap at 2001.
N_RAMAN_COLS: int = 2001    # spectral dimensionality fed to PCA

# Savitzky-Golay preprocessing parameters
SG_WINDOW: int = 15         # must be odd and > SG_POLY
SG_POLY:   int = 2          # polynomial order
SG_DERIV:  int = 1          # derivative order  (d=1 -> first-derivative spectra)

# Train / val split (mirrors the PI-LSTM / PLS random split for fairness)
TRAIN_FRAC: float = 0.80
SPLIT_SEED: int   = 42

# Cross-validation
N_CV_FOLDS: int = 10        # GroupKFold folds (or standard KFold as fallback)

# Minimum batch length to include
MIN_BATCH_LEN: int = 50

# Output paths
OUTPUT_DIR:      Path = Path("./outputs/pca_baseline")
CHECKPOINT_PATH: Path = Path("./checkpoints/pca_baseline.joblib")

# Encoder artefacts -- consumed by src/data/pca_encoder.PCARamanEncoderV1
PCA_MODEL_PATH:  Path = Path("./checkpoints/pca_best.joblib")
PCA_SCALER_PATH: Path = Path("./checkpoints/pca_scaler.joblib")


# -- ------------------------------------------------------------------------
# DATA LOADING & PREPROCESSING  (byte-for-byte copy of pls_baseline.py)
# -- ------------------------------------------------------------------------

def _apply_sg_filter(X: np.ndarray) -> np.ndarray:
    """
    Apply a Savitzky-Golay first-derivative filter row-wise to a 2-D spectra
    matrix  (rows = samples, cols = wavenumber channels).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_wavenumbers)

    Returns
    -------
    X_sg : ndarray of same shape -- first-derivative spectra
    """
    return savgol_filter(
        X,
        window_length=SG_WINDOW,
        polyorder=SG_POLY,
        deriv=SG_DERIV,
        axis=1,   # filter along the spectral axis
    )


def load_batches(csv_path: str) -> List[Dict]:
    """
    Load the IndPenSim V3 CSV, detect batch boundaries via the time-reset
    convention, apply SG preprocessing to Raman spectra, and return a list
    of per-batch dicts.

    Byte-for-byte identical pipeline to pls_baseline.load_batches() and
    svr_baseline.load_batches().

    Each dict contains
    ------------------
    X_raman    : ndarray (n_steps, N_RAMAN_COLS)  -- SG-filtered spectra
    y_pen_dense: ndarray (n_steps,)               -- interpolated Penicillin (g/L)
    y_pen_sparse: ndarray (n_steps,)              -- raw sparse Penicillin (NaN where not measured)
    batch_id   : int                              -- 0-indexed batch number
    is_fault   : bool                             -- True for faulty batches
    """
    print(f"\n[DATA]  Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"[DATA]  Raw shape: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # -- Raman column names ------------------------------------------------
    # Resolve the first N_RAMAN_COLS channel column *names* by absolute position
    # in the DataFrame.  Using column names (not integer iloc positions) ensures
    # that subsequent per-batch sub-DataFrames (which share the same column
    # schema) can be indexed consistently via df[raman_col_names].
    all_cols    = df.columns.tolist()
    avail_raman = len(all_cols) - RAMAN_START_COL_IDX
    n_raman_use = min(N_RAMAN_COLS, avail_raman)
    if n_raman_use < N_RAMAN_COLS:
        warnings.warn(
            f"CSV has only {avail_raman} Raman columns after index "
            f"{RAMAN_START_COL_IDX}; expected {N_RAMAN_COLS}. "
            f"Using all {n_raman_use} available.",
        )
    raman_col_names = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + n_raman_use]
    print(f"[DATA]  Raman channels: {n_raman_use}  (cols {RAMAN_START_COL_IDX}-"
          f"{RAMAN_START_COL_IDX + n_raman_use - 1})")

    # -- Batch detection (time-reset convention, same as PenicillinDataModule)
    t_vals    = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid       = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:   # time went backwards -> new batch
            bid += 1
        batch_ids[i] = bid
    df["_batch_id"] = batch_ids
    n_batches = int(bid + 1)
    print(f"[DATA]  Detected {n_batches} batches")

    # -- Per-batch processing ----------------------------------------------
    batches: List[Dict] = []

    for b_id in range(n_batches):
        b = df[df["_batch_id"] == b_id].copy().reset_index(drop=True)

        if len(b) < MIN_BATCH_LEN:
            continue

        # -- Extract Raman BEFORE any row filtering so indices stay aligned -
        # b shares the same column schema as df; indexing by name is safe here.
        X_raw = b[raman_col_names].values.astype(np.float64)

        # Preserve sparse Penicillin measurements before interpolation
        y_pen_sparse = b[TARGET_COLS[1]].values.copy()   # NaN where not sampled

        # Dense linear interpolation (same as PenicillinDataModule)
        for tc in TARGET_COLS:
            b[tc] = b[tc].interpolate(method="linear").ffill().bfill()

        # Build a keep-mask from required columns (avoids re-indexing mismatch)
        required = ["Time (h)", TARGET_COLS[0], TARGET_COLS[1], FAULT_COL]
        keep_mask = ~b[required].isna().any(axis=1).values  # shape: (n_steps,)

        # Apply the same mask to Raman and sparse targets to stay in sync
        b            = b[keep_mask].reset_index(drop=True)
        X_raw        = X_raw[keep_mask]
        y_pen_sparse = y_pen_sparse[keep_mask]

        if len(b) < MIN_BATCH_LEN:
            continue

        # Clamp any negative/NaN values in spectra before SG filtering
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        X_raw = np.clip(X_raw, a_min=0.0, a_max=None)

        # Savitzky-Golay first-derivative filter (row-wise along spectral axis)
        X_sg = _apply_sg_filter(X_raw)   # shape: (n_steps, n_raman_use)

        y_pen_dense    = b[TARGET_COLS[1]].values
        y_pen_sparse_b = y_pen_sparse

        is_fault = bool(b[FAULT_COL].max() > 0)

        batches.append({
            "X_raman":     X_sg,
            "y_pen_dense": y_pen_dense,
            "y_pen_sparse": y_pen_sparse_b,
            "batch_id":    b_id,
            "is_fault":    is_fault,
        })

    n_clean = sum(not b["is_fault"] for b in batches)
    n_fault = sum(b["is_fault"]     for b in batches)
    print(f"[DATA]  Fault-free: {n_clean}  |  Faulty: {n_fault}")
    return batches


# -- ------------------------------------------------------------------------
# TRAIN / VAL / TEST SPLIT  (byte-for-byte copy of pls_baseline.py)
# -- ------------------------------------------------------------------------

def split_batches(
    batches: List[Dict],
    train_frac: float = TRAIN_FRAC,
    seed:       int   = SPLIT_SEED,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Replicate the PI-LSTM / PLS / SVR random split (seed=42) for a strict
    apples-to-apples comparison.

    Returns
    -------
    train_batches, val_batches (FF), fault_batches (test)
    """
    clean = [b for b in batches if not b["is_fault"]]
    fault = [b for b in batches if b["is_fault"]]

    n_clean = len(clean)
    n_train = int(n_clean * train_frac)

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(n_clean)

    train_b = [clean[i] for i in perm[:n_train]]
    val_b   = [clean[i] for i in perm[n_train:]]

    print(f"\n[SPLIT]  Train: {len(train_b)} batches  |  "
          f"FF-val: {len(val_b)} batches  |  Fault: {len(fault)} batches")
    return train_b, val_b, fault


# -- ------------------------------------------------------------------------
# ARRAY BUILDERS  (byte-for-byte copy of pls_baseline.py)
# -- ------------------------------------------------------------------------

def build_flat_arrays(
    batches: List[Dict],
    use_dense_targets: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten a list of batch dicts into 2-D (X, y) arrays suitable for
    sklearn.  Also returns a group vector for GroupKFold CV.

    Parameters
    ----------
    batches           : list of batch dicts (output of load_batches)
    use_dense_targets : if True (default) use interpolated Penicillin for y,
                        matching the dense supervision used by neural models;
                        if False use only sparse measurement time-points.

    Returns
    -------
    X      : (n_samples, n_raman)
    y      : (n_samples,)
    groups : (n_samples,)  -- batch_id per sample, for GroupKFold
    """
    X_parts, y_parts, g_parts = [], [], []

    for b in batches:
        if use_dense_targets:
            X_b = b["X_raman"]
            y_b = b["y_pen_dense"]
        else:
            # Sparse only: keep rows where Penicillin was actually measured
            mask = ~np.isnan(b["y_pen_sparse"])
            if mask.sum() == 0:
                continue
            X_b = b["X_raman"][mask]
            y_b = b["y_pen_sparse"][mask]

        X_parts.append(X_b)
        y_parts.append(y_b)
        g_parts.append(np.full(len(X_b), fill_value=b["batch_id"], dtype=np.int32))

    X      = np.vstack(X_parts)
    y      = np.concatenate(y_parts)
    groups = np.concatenate(g_parts)
    return X, y, groups


# -- ------------------------------------------------------------------------
# PCR PIPELINE
# -- ------------------------------------------------------------------------

def build_pcr_pipeline(n_components: int = N_COMPONENTS) -> Pipeline:
    """
    Construct the Principal Component Regression pipeline.

    Pipeline stages
    ---------------
    1. StandardScaler
       Centres and scales each of the 2001 SG-derivative wavenumber channels
       to zero mean / unit variance.  Required because PCA is variance-based
       and would otherwise be dominated by high-magnitude channels.

    2. PCA(n_components=N_COMPONENTS, random_state=SPLIT_SEED)
       Projects the 2001-d scaled spectra into the K=4 principal components
       that capture the dominant spectral covariance.  This is the
       UNSUPERVISED counterpart of PLSRegression: the components maximise
       variance explained on the spectra alone, with no knowledge of the
       Penicillin target -- a deliberate contrast with PLS.

    3. LinearRegression()
       Ordinary least squares on the K=4 PCA scores.  No regularisation, no
       intercept manipulation (sklearn handles intercept automatically).
       This is the standard PCR formulation.

    Parameters
    ----------
    n_components : number of principal components to retain (default 4)
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_components, random_state=SPLIT_SEED)),
        ("reg",    LinearRegression()),
    ])


# -- ------------------------------------------------------------------------
# CROSS-VALIDATION
# -- ------------------------------------------------------------------------

def run_cross_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups:  np.ndarray,
    n_components: int = N_COMPONENTS,
    n_folds:      int = N_CV_FOLDS,
) -> Tuple[float, np.ndarray]:
    """
    10-fold cross-validation, grouped by batch_id (GroupKFold), to simulate
    Leave-One-Batch-Out (LOBO) CV.  Falls back to standard KFold if the number
    of unique batches is fewer than n_folds.

    Identical strategy to pls_baseline.run_cross_validation() so the RMSECV
    values are directly comparable.

    Parameters
    ----------
    X_train, y_train, groups : flat training arrays (from build_flat_arrays)
    n_components             : PCA components
    n_folds                  : number of CV folds

    Returns
    -------
    rmsecv   : scalar  -- root-mean-squared error of cross-validation
    y_cv_pred: (n_samples,)  -- out-of-fold predictions for diagnostic use
    """
    pcr = build_pcr_pipeline(n_components=n_components)

    n_groups = len(np.unique(groups))
    if n_groups >= n_folds:
        cv = GroupKFold(n_splits=n_folds)
        cv_kwargs: Dict = {"groups": groups}
        cv_label = f"GroupKFold({n_folds}) by batch_id"
    else:
        warnings.warn(
            f"Only {n_groups} unique batch groups found in training data; "
            f"falling back to KFold({n_folds}).",
        )
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=SPLIT_SEED)
        cv_kwargs = {}
        cv_label = f"KFold({n_folds})"

    print(f"\n[CV]  Running {cv_label} -- this may take a moment ...")
    y_cv_pred = cross_val_predict(pcr, X_train, y_train, cv=cv, **cv_kwargs)
    y_cv_pred = y_cv_pred.ravel()

    rmsecv = float(np.sqrt(mean_squared_error(y_train, y_cv_pred)))
    print(f"[CV]  10-fold RMSECV = {rmsecv:.4f} g/L")
    return rmsecv, y_cv_pred


# -- ------------------------------------------------------------------------
# MODEL FITTING
# -- ------------------------------------------------------------------------

def fit_pcr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_components: int = N_COMPONENTS,
) -> Pipeline:
    """
    Fit the PCR pipeline (StandardScaler -> PCA -> LinearRegression) on the
    full training/calibration set.

    Parameters
    ----------
    X_train      : (n_samples, n_raman) -- SG-preprocessed spectra
    y_train      : (n_samples,)         -- Penicillin concentration (g/L)
    n_components : number of principal components

    Returns
    -------
    Fitted sklearn Pipeline
    """
    print(f"\n[FIT]  Fitting PCR pipeline "
          f"(StandardScaler -> PCA(n_components={n_components}) -> LinearRegression) "
          f"on {X_train.shape[0]:,} samples ({X_train.shape[1]} spectral features) ...")
    pcr = build_pcr_pipeline(n_components=n_components)
    pcr.fit(X_train, y_train)

    # Diagnostic: print explained variance ratio
    evr = pcr.named_steps["pca"].explained_variance_ratio_
    print(f"[FIT]  PCA explained variance ratio per component: "
          f"{np.array2string(evr, precision=4)}")
    print(f"[FIT]  PCA cumulative explained variance: {evr.sum():.4f}")
    print("[FIT]  Done.")
    return pcr


# -- ------------------------------------------------------------------------
# MODEL PERSISTENCE
# -- ------------------------------------------------------------------------

def save_model(
    model: Pipeline,
    path: Path = CHECKPOINT_PATH,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Persist the fitted PCR pipeline (and optional metadata) to disk using
    joblib.

    The saved artefact is a dict with keys:
      "model"    : the fitted sklearn Pipeline object
      "metadata" : dict with hyperparameters, RMSECV, and metric results

    Parameters
    ----------
    model    : fitted PCR Pipeline
    path     : output file path (extension .joblib recommended)
    metadata : any extra information to store alongside the model
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "metadata": metadata or {},
    }
    joblib.dump(payload, path, compress=3)
    print(f"\n[SAVE]  PCR pipeline saved to {path}")


def load_model(path: Path = CHECKPOINT_PATH) -> Tuple[Pipeline, Dict]:
    """
    Load a previously saved PCR pipeline.

    Returns
    -------
    model    : sklearn Pipeline
    metadata : dict stored alongside the model
    """
    payload  = joblib.load(path)
    model    = payload["model"]
    metadata = payload.get("metadata", {})
    print(f"[LOAD]  PCR pipeline loaded from {path}")
    return model, metadata


def save_encoder_artefacts(
    pcr_pipeline: Pipeline,
    pca_path:     Path = PCA_MODEL_PATH,
    scaler_path:  Path = PCA_SCALER_PATH,
) -> None:
    """
    Persist the fitted StandardScaler and PCA components as standalone
    artefacts so they can be loaded by ``PCARamanEncoderV1`` and reused as
    a frozen Raman encoder by the PCA-PI-LSTM downstream model.

    File layout (mirrors cdae_best.pt + cdae_scaler.joblib convention)
    -------------------------------------------------------------------
      ``./checkpoints/pca_best.joblib``    -- sklearn.decomposition.PCA
      ``./checkpoints/pca_scaler.joblib``  -- sklearn.preprocessing.StandardScaler

    Parameters
    ----------
    pcr_pipeline : the fitted PCR sklearn Pipeline (must contain steps
                   named "scaler" and "pca")
    pca_path     : output path for the fitted PCA object
    scaler_path  : output path for the fitted StandardScaler object
    """
    pca_path    = Path(pca_path)
    scaler_path = Path(scaler_path)
    pca_path.parent.mkdir(parents=True,    exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    pca_obj    = pcr_pipeline.named_steps["pca"]
    scaler_obj = pcr_pipeline.named_steps["scaler"]

    joblib.dump(pca_obj,    pca_path,    compress=3)
    joblib.dump(scaler_obj, scaler_path, compress=3)

    print(f"[SAVE]  PCA encoder artefact saved to    {pca_path}")
    print(f"[SAVE]  PCA encoder scaler saved to       {scaler_path}")
    print(f"        (these are loaded by PCARamanEncoderV1 for PCA-PI-LSTM)")


# -- ------------------------------------------------------------------------
# EVALUATION HELPERS  (byte-for-byte copy of pls_baseline.py)
# -- ------------------------------------------------------------------------

def _predict_batch_at_sparse_points(
    model: Pipeline,
    batch: Dict,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run the PCR pipeline on a single batch and collect predictions only at
    the sparse offline measurement time-points  (i.e., where  y_pen_sparse
    is not NaN).

    This exactly mirrors the evaluation methodology for PLS / SVR /
    CDAE-PI-LSTM.

    Returns
    -------
    y_true : (n_sparse,) -- measured Penicillin values
    y_pred : (n_sparse,) -- PCR predictions at those time-points
    Returns (None, None) if no sparse measurements exist in this batch.
    """
    mask = ~np.isnan(batch["y_pen_sparse"])
    if mask.sum() == 0:
        return None, None

    X_sparse = batch["X_raman"][mask]          # (n_sparse, n_raman)
    y_true   = batch["y_pen_sparse"][mask]     # (n_sparse,)
    y_pred   = model.predict(X_sparse).ravel() # (n_sparse,)

    return y_true, y_pred


def evaluate_split(
    model: Pipeline,
    batches: List[Dict],
    split_name: str = "split",
) -> Dict:
    """
    Evaluate the PCR pipeline on a list of batches (FF-val or fault batches).
    Returns the aggregated metrics dict.
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
    model: Pipeline,
    fault_batches: List[Dict],
) -> Tuple[float, List[Dict]]:
    """
    Compute per-batch RMSE for each fault batch and report the mean.

    Returns
    -------
    mean_fault_rmse : scalar
    per_batch_results : list of dicts  {batch_id, rmse, r2, n}
    """
    per_batch: List[Dict] = []

    for b in fault_batches:
        y_true, y_pred = _predict_batch_at_sparse_points(model, b)
        if y_true is None:
            per_batch.append({"batch_id": b["batch_id"],
                               "rmse": float("nan"), "r2": float("nan"), "n": 0})
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


# -- ------------------------------------------------------------------------
# RESULTS SUMMARY TABLE
# -- ------------------------------------------------------------------------

def print_comparison_table(
    rmsecv:          float,
    ffval_metrics:   Dict,
    fault_mean_rmse: float,
    per_batch:       List[Dict],
    explained_var:   np.ndarray,
) -> None:
    """
    Print a compact results summary formatted for copy-paste into the
    benchmark comparison table.
    """
    sep = "-" * 64
    print(f"\n{sep}")
    print(f"  PCA-PCR BASELINE -- BENCHMARK SUMMARY  (n_components={N_COMPONENTS})")
    print(sep)
    print(f"  PCA explained variance")
    print(f"    Per component           : "
          f"{np.array2string(explained_var, precision=4)}")
    print(f"    Cumulative              : {float(explained_var.sum()):.4f}")
    print(f"")
    print(f"  Calibration")
    print(f"    10-fold RMSECV          : {rmsecv:.4f} g/L")
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
    print(f"  {'Model':<22} | {'Pen RMSE (val)':>14} | {'Pen R2 (val)':>12} | "
          f"{'Fault RMSE':>10} | {'RMSECV':>8}")
    print(f"  {'-'*22}-+-{'-'*14}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}")
    print(
        f"  {'PCA-PCR (SG d=1, K=4)':<22} | "
        f"{ffval_metrics.get('RMSE', float('nan')):>14.4f} | "
        f"{ffval_metrics.get('R2',   float('nan')):>12.4f} | "
        f"{fault_mean_rmse:>10.4f} | "
        f"{rmsecv:>8.4f}"
    )
    print(sep)


# -- ------------------------------------------------------------------------
# MAIN
# -- ------------------------------------------------------------------------

def main(csv_path: str = CSV_PATH) -> None:
    """
    End-to-end PCA-PCR baseline pipeline:
      1.  Load & SG-preprocess raw Raman data
      2.  Split into train / FF-val / fault-test (random split, seed=42)
      3.  10-fold GroupKFold cross-validation on calibration set  -> RMSECV
      4.  Fit final PCR pipeline on all calibration batches
      5.  Evaluate on FF-val and fault batches
      6.  Print benchmark comparison table
      7.  Save fitted Pipeline to CHECKPOINT_PATH
      8.  Save standalone StandardScaler + PCA as encoder artefacts (for
          downstream PCA-PI-LSTM training via src/data/pca_encoder.py)
    """
    # -- 1. Load data ------------------------------------------------------
    batches = load_batches(csv_path)

    # -- 2. Split ----------------------------------------------------------
    train_batches, val_batches, fault_batches = split_batches(batches)

    # -- 3. Build flat training arrays ------------------------------------
    # use_dense_targets=True: supervise on all time-steps with interpolated P,
    # consistent with how PLS and CDAE-PI-LSTM are trained.
    X_train, y_train, groups = build_flat_arrays(train_batches, use_dense_targets=True)
    print(f"\n[DATA]  Training set: X={X_train.shape}  y={y_train.shape}  "
          f"n_groups={len(np.unique(groups))}")

    # -- 4. Cross-validation -----------------------------------------------
    rmsecv, _ = run_cross_validation(X_train, y_train, groups)

    # -- 5. Fit final model ------------------------------------------------
    pcr = fit_pcr(X_train, y_train)

    # -- 6. Evaluate -- FF-val ---------------------------------------------
    ffval_metrics = evaluate_split(pcr, val_batches, split_name="FF-Validation")

    # -- 7. Evaluate -- fault batches --------------------------------------
    fault_mean_rmse, per_batch = evaluate_fault_batches(pcr, fault_batches)

    # -- 8. Benchmark summary ----------------------------------------------
    evr = pcr.named_steps["pca"].explained_variance_ratio_
    print_comparison_table(rmsecv, ffval_metrics, fault_mean_rmse, per_batch, evr)

    # -- 9. Save model -----------------------------------------------------
    metadata = {
        # Hyperparameters
        "n_components":  N_COMPONENTS,
        "n_raman_cols":  N_RAMAN_COLS,
        "sg_window":     SG_WINDOW,
        "sg_poly":       SG_POLY,
        "sg_deriv":      SG_DERIV,
        "train_frac":    TRAIN_FRAC,
        "split_seed":    SPLIT_SEED,
        # Results
        "rmsecv":        rmsecv,
        "ffval_rmse":    ffval_metrics.get("RMSE"),
        "ffval_r2":      ffval_metrics.get("R2"),
        "fault_mean_rmse": fault_mean_rmse,
        "per_batch_fault": per_batch,
        "pca_explained_variance_ratio": evr.tolist(),
        "pca_cumulative_explained_variance": float(evr.sum()),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_model(pcr, CHECKPOINT_PATH, metadata)

    # -- 10. Save encoder artefacts (consumed by PCA-PI-LSTM) -------------
    save_encoder_artefacts(pcr, PCA_MODEL_PATH, PCA_SCALER_PATH)


# -- CLI entry-point --------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PCA-PCR baseline for ReaKt-v2 Penicillin soft-sensor benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=CSV_PATH,
        help="Path to the IndPenSim V3 CSV file.",
    )
    args = parser.parse_args()
    main(csv_path=args.csv)
