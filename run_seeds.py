#!/usr/bin/env python3
"""
run_seeds.py
============
Multi-seed, dual-protocol training and evaluation sweep for all benchmark
algorithms.  Produces mean RMSE ± std (variance) per model per protocol,
plus NRMSE% and Δ% columns matching the paper's Table format.

Protocols
---------
  Protocol 1 (default):
      Train  : R1 + R2 + R3  (batches 1–90, 80/20 random → train/val)
      Test   : R4 / Fault    (batches 91–100)

  Protocol 2:
      Train  : R1 only       (batches 1–30, 80/20 random → train/val)
      Test   : R4 / Fault    (batches 91–100)
      Also reports generalisation on R2 (batches 31–60) and R3 (61–90).

Model categories
----------------
  Static baselines (inline sklearn, no checkpoint):
      cubic_poly, mean_traj, pca_pcr, pls, svr

  Sequence/temporal models (via train.py):
      pi_lstm, neural_ode, cdae_pi_lstm, cvae_pi_lstm, beta_vae_pi_lstm,
      pca_pi_lstm, pls_pi_lstm, beta_vae_lstm, beta_vae_lp_pi_lstm,
      beta_vae_latent_lp_pi_lstm, beta_vae_mamba, beta_vae_liquid,
      delta_cdae_pi_lstm, cdae_process_pi_lstm, process_pi_lstm

  On-the-fly adaptation (subprocess calls to scripts/):
      beta_vae_jit_ft, cvae_jit_ft, vae_jitl

Usage
-----
    # Single model, both protocols, 5 seeds
    python run_seeds.py --model beta_vae_pi_lstm

    # All models, both protocols
    python run_seeds.py --model all --n-seeds 4 --device cuda

    # Protocol 1 only, custom seeds
    python run_seeds.py --model cvae_pi_lstm --protocol 1 --seeds 42 123 456

    # Skip training; aggregate from existing checkpoints / caches
    python run_seeds.py --model all --eval-only

    # Resume an interrupted run
    python run_seeds.py --model all --skip-existing

Output
------
    outputs/seed_sweep/<model>/
        p1/checkpoints/<model>_s<N>.pt
        p1/seed_<N>.log
        p1/results.csv          per-seed rows + mean/std footer
        p2/checkpoints/<model>_s<N>.pt
        p2/seed_<N>.log
        p2/results.csv
        summary.csv             mean ± std / NRMSE / Δ for both protocols

    outputs/seed_sweep/summary.csv   combined table (--model all)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SEEDS   = [123, 456, 789, 1024]
DEFAULT_N_SEEDS = 4
OUT_ROOT        = _ROOT / "outputs" / "seed_sweep"
CSV_PATH        = str(_ROOT / "100_Batches_IndPenSim_V3.csv")

# NRMSE denominator: mean penicillin g/L at sparse test (fault) samples.
# Back-calculated from paper table: cubic_poly test=8.617 → NRMSE=101.6%
# ⟹ denominator = 8.617 / 1.016 ≈ 8.481 g/L.
PEN_NORM = 8.481

# ── Model categories ─────────────────────────────────────────────────────────

# ── Model categories — restricted to the paper's benchmark table ──────────────
# Time-only baselines + static chemometric baselines
STATIC_MODELS = ["cubic_poly", "mean_traj", "pca_pcr", "pls", "svr"]

# Sequence models trained via train.py — only the ones in the paper table
TRAINPY_MODELS = [
    "pi_lstm",            # LSTM (end-to-end, process features only)
    "neural_ode",         # Neural ODE
    "pca_pi_lstm",        # PCA-LSTM
    "pls_pi_lstm",        # PLS-LSTM
    "cdae_pi_lstm",       # CDAE-LSTM
    "cvae_pi_lstm",       # CVAE-LSTM
    "beta_vae_pi_lstm",   # β-VAE-LSTM (proposed)
]

# Adaptation models requiring pre-trained base checkpoints.
ADAPTATION_MODELS = ["beta_vae_jit_ft", "cvae_jit_ft", "vae_jitl"]

ALL_MODELS = STATIC_MODELS + TRAINPY_MODELS + ADAPTATION_MODELS

# ── Encoder configuration per LSTM model ─────────────────────────────────────
# Each entry maps an LSTM model name to the script and arguments needed to
# retrain its Raman encoder from scratch for a given seed / protocol.
#
# "type"          : key passed as --model to the encoder training script
# "script"        : path relative to _ROOT
# "ckpt_ext"      : file extension of the encoder checkpoint (.pt or .joblib)
# "extra_args"    : extra CLI args forwarded verbatim (e.g. β-VAE schedule)
#
# β-VAE: match the hyperparameters used for the reference checkpoint
#   (max_beta=0.005, anneal_schedule=linear, anneal_epochs=100, anneal_delay=20)

ENCODER_CONFIG: Dict[str, Dict] = {
    "cdae_pi_lstm": {
        "type":      "cdae",
        "script":    "scripts/train_autoencoder.py",
        "ckpt_ext":  ".pt",
        "extra_args": [],
    },
    "cvae_pi_lstm": {
        "type":      "cvae",
        "script":    "scripts/train_autoencoder.py",
        "ckpt_ext":  ".pt",
        "extra_args": [],
    },
    "beta_vae_pi_lstm": {
        "type":      "beta_vae",
        "script":    "scripts/train_autoencoder.py",
        "ckpt_ext":  ".pt",
        "extra_args": [
            "--max-beta",        "0.005",
            "--anneal-schedule", "linear",
            "--anneal-epochs",   "100",
            "--anneal-delay",    "20",
        ],
    },
    "pca_pi_lstm": {
        "type":      "pca",
        "script":    "scripts/pca_baseline.py",
        "ckpt_ext":  ".joblib",
        "extra_args": [],
    },
    "pls_pi_lstm": {
        "type":      "pls",
        "script":    "scripts/pls_baseline.py",
        "ckpt_ext":  ".joblib",
        "extra_args": [],
    },
    # pi_lstm and neural_ode use no Raman encoder → no entry here
}

# ── JIT-FT / JITL configuration ──────────────────────────────────────────────

JITFT_CONFIG: Dict[str, Dict] = {
    "beta_vae_jit_ft": {
        "script":       "scripts/cvae_jit_ft.py",
        "script_model": "beta_vae",
        "base_model":   "beta_vae_pi_lstm",
        "K":            150,
    },
    "cvae_jit_ft": {
        "script":       "scripts/cvae_jit_ft.py",
        "script_model": "cvae",
        "base_model":   "cvae_pi_lstm",
        "K":            200,
    },
    # vae_jitl uses the CVAE encoder (not the LSTM checkpoint) and does not
    # support --regime-ablation; Protocol 2 results will always be NaN.
    "vae_jitl": {
        "script":       "scripts/vae_jitl.py",
        "base_model":   "cvae_pi_lstm",   # only used for val-RMSE lookup
        "K":            500,
    },
}

# Fallback checkpoint paths (used when seeded checkpoints are missing)
_DEFAULT_P1_CKPTS = {
    "cvae_pi_lstm":     "./checkpoints/cvae_pi_lstm.pt",
    "beta_vae_pi_lstm": "./checkpoints/beta_vae_pi_lstm.pt",
}
_DEFAULT_P2_CKPTS = {
    "cvae_pi_lstm":     "./checkpoints/cvae_pi_lstm_regime_ablation.pt",
    "beta_vae_pi_lstm": "./checkpoints/beta_vae_pi_lstm_regime_ablation.pt",
}


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ckpt_path(out_dir: Path, model: str, protocol: int, seed: int) -> Path:
    return out_dir / model / f"p{protocol}" / "checkpoints" / f"{model}_s{seed}.pt"


def _log_path(out_dir: Path, model: str, protocol: int, seed: int) -> Path:
    return out_dir / model / f"p{protocol}" / f"seed_{seed}.log"


def _static_cache_path(out_dir: Path, protocol: int, seed: int) -> Path:
    return out_dir / "_static_cache" / f"p{protocol}_s{seed}.json"


def _adapt_cache_path(out_dir: Path, model: str, protocol: int, seed: int) -> Path:
    return out_dir / model / f"p{protocol}" / f"adapt_s{seed}.json"


def _enc_ckpt_path(out_dir: Path, model: str, protocol: int, seed: int) -> Path:
    """Return the per-seed encoder checkpoint path."""
    ext = ENCODER_CONFIG[model]["ckpt_ext"]
    enc_type = ENCODER_CONFIG[model]["type"]
    return out_dir / model / f"p{protocol}" / "encoders" / f"{enc_type}_s{seed}{ext}"


def _enc_scaler_path(out_dir: Path, model: str, protocol: int, seed: int) -> Path:
    """Return the per-seed encoder scaler path (always .joblib)."""
    enc_type = ENCODER_CONFIG[model]["type"]
    return out_dir / model / f"p{protocol}" / "encoders" / f"{enc_type}_scaler_s{seed}.joblib"


# ─────────────────────────────────────────────────────────────────────────────
# Encoder training helper
# ─────────────────────────────────────────────────────────────────────────────

def _run_encoder_train(
    model:    str,
    protocol: int,
    seed:     int,
    out_dir:  Path,
    device:   str,
) -> int:
    """
    Train the Raman encoder for a given LSTM model from scratch.

    Dispatches to:
      - scripts/train_autoencoder.py  (cdae / cvae / beta_vae)
      - scripts/pca_baseline.py       (pca)
      - scripts/pls_baseline.py       (pls)

    Returns the subprocess exit code.
    """
    cfg      = ENCODER_CONFIG[model]
    script   = str(_ROOT / cfg["script"])
    enc_type = cfg["type"]
    extra    = cfg.get("extra_args", [])

    enc_ckpt   = _enc_ckpt_path(out_dir, model, protocol, seed)
    enc_scaler = _enc_scaler_path(out_dir, model, protocol, seed)
    enc_ckpt.parent.mkdir(parents=True, exist_ok=True)

    log = out_dir / model / f"p{protocol}" / f"encoder_s{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    if enc_type in ("cdae", "cvae", "beta_vae"):
        cmd = [
            sys.executable, script,
            "--model",  enc_type,
            "--seed",   str(seed),
            "--device", device or "cuda",
            "--ckpt",   str(enc_ckpt),
            "--scaler", str(enc_scaler),
        ] + extra
    else:  # pca / pls
        cmd = [
            sys.executable, script,
            "--seed",   str(seed),
            "--ckpt",   str(enc_ckpt),
            "--scaler", str(enc_scaler),
        ]

    if protocol == 2:
        cmd.append("--regime-ablation")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT) + ":" + env.get("PYTHONPATH", "")

    print(f"  [P{protocol},s={seed}] Training {enc_type} encoder → {enc_ckpt.name}  "
          f"(log: {log.name})")
    with open(log, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                cwd=str(_ROOT), env=env)
    return result.returncode


# ─────────────────────────────────────────────────────────────────────────────
# train.py model helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_train(
    model: str,
    protocol: int,
    seed: int,
    out_dir: Path,
    epochs: Optional[int],
    device: str,
    extra_args: List[str],
    raman_ckpt: Optional[Path] = None,
    raman_scaler: Optional[Path] = None,
) -> int:
    """Launch train.py for one model/protocol/seed.  Returns exit code."""
    ckpt = _ckpt_path(out_dir, model, protocol, seed)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    log  = _log_path(out_dir, model, protocol, seed)
    log.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(_ROOT / "train.py"),
        "--model", model,
        "--seed",  str(seed),
        "--ckpt",  str(ckpt),
        "--no-plots",
    ]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if device:
        cmd += ["--device", device]
    if protocol == 2:
        cmd += ["--regime-ablation"]
    # Per-seed encoder checkpoint overrides
    if raman_ckpt is not None:
        cmd += ["--raman-ckpt", str(raman_ckpt)]
    if raman_scaler is not None:
        cmd += ["--raman-scaler", str(raman_scaler)]
    cmd += extra_args

    print(f"  [P{protocol},s={seed}] Training {model} → {ckpt.name}  (log: {log.name})")
    with open(log, "w") as f:
        result = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(_ROOT),
        )
    return result.returncode


def _extract_pen(m: dict) -> Dict:
    """Extract penicillin metrics from a nested metrics dict."""
    pen = (m or {}).get("pen", {}) or {}
    return {
        "RMSE": pen.get("RMSE", float("nan")),
        "MAE":  pen.get("MAE",  float("nan")),
        "R2":   pen.get("R2",   float("nan")),
        "n":    pen.get("n",    0),
    }


def _eval_from_ckpt(
    model: str,
    protocol: int,
    seed: int,
    out_dir: Path,
) -> Optional[Dict]:
    """
    Load a checkpoint and extract per-split penicillin metrics.

    Protocol 1  (no --regime-ablation)
        val_metrics   → validation  (R1-R3 holdout 20 %)
        test_metrics  → Test R4/Fault  ← primary test column

    Protocol 2  (--regime-ablation in train.py)
        val_metrics   → validation  (R1 holdout 20 %)
        test_metrics  → R2 (operator-controlled, batches 31-60)
        r3_metrics    → R3 (APC closed-loop, batches 61-90)
        fault_metrics → R4/Fault  ← primary test column  *** THE KEY FIX ***
    """
    import torch

    ckpt = _ckpt_path(out_dir, model, protocol, seed)
    if not ckpt.exists():
        print(f"  [P{protocol},s={seed}] {model}: checkpoint not found ({ckpt})")
        return None

    state   = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    val_m   = _extract_pen(state.get("val_metrics")   or {})
    test_m  = _extract_pen(state.get("test_metrics")  or {})
    r3_m    = _extract_pen(state.get("r3_metrics")    or {})
    fault_m = _extract_pen(state.get("fault_metrics") or {})

    row: Dict = {
        "seed":         seed,
        "val_pen_rmse": val_m["RMSE"],
        "val_pen_mae":  val_m["MAE"],
        "val_pen_r2":   val_m["R2"],
    }

    if protocol == 1:
        row.update({
            "test_pen_rmse": test_m["RMSE"],  # Fault/R4
            "test_pen_mae":  test_m["MAE"],
            "test_pen_r2":   test_m["R2"],
        })
    else:
        # Under --regime-ablation: test_metrics == R2, fault_metrics == R4
        row.update({
            "r2_pen_rmse":   test_m["RMSE"],    # R2
            "r3_pen_rmse":   r3_m["RMSE"],      # R3
            "test_pen_rmse": fault_m["RMSE"],   # R4/Fault  ← primary
            "test_pen_mae":  fault_m["MAE"],
            "test_pen_r2":   fault_m["R2"],
        })

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Static baseline helpers
# ─────────────────────────────────────────────────────────────────────────────

def _preload_batches(csv_path: str) -> List[Dict]:
    """
    Load the IndPenSim CSV **once**, extract per-batch numpy arrays, then
    immediately free the full ~2 GB DataFrame.

    Returns a list of lightweight dicts (read-only, shared across threads):
        {"id": int, "time": (T,), "pen_sparse": (T,),
         "raman": (T, 2001), "fault": bool}
    """
    import gc
    import pandas as pd

    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from src.data.dataset import TARGET_COLS, RAMAN_START_COL_IDX, FAULT_COL

    print(f"  [preload] Reading {csv_path} ...", flush=True)
    df = pd.read_csv(csv_path)

    t_vals    = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid

    N_RAMAN = 2001

    # Extract needed arrays as numpy ONCE before the per-batch loop.
    # Use .iloc for the Raman block (positional slice is much faster than
    # selecting 2001 columns by name with df[raman_col_names]).
    raman_arr = df.iloc[:, RAMAN_START_COL_IDX:RAMAN_START_COL_IDX + N_RAMAN
                        ].to_numpy(dtype=np.float32, copy=False)  # (N, 2001)
    time_arr  = df["Time (h)"].to_numpy(dtype=np.float64)
    pen_arr   = df[TARGET_COLS[1]].to_numpy(dtype=np.float64)
    fault_arr = df[FAULT_COL].to_numpy()

    # DataFrame no longer needed — free 2 GB now, before the batch loop
    del df
    gc.collect()

    batches: List[Dict] = []
    for b_id in range(int(batch_ids[-1]) + 1):
        mask = batch_ids == b_id
        if mask.sum() < 50:
            continue
        batches.append({
            "id":         b_id,
            "time":       time_arr[mask].copy(),
            "pen_sparse": pen_arr[mask].copy(),
            "raman":      raman_arr[mask].copy(),
            "fault":      bool(fault_arr[mask].max() > 0),
        })

    del time_arr, pen_arr, fault_arr, raman_arr
    gc.collect()

    import ctypes
    try:
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    raman_mb = sum(b["raman"].nbytes for b in batches) / 1e6
    print(f"  [preload] {len(batches)} batches ready  (Raman: {raman_mb:.0f} MB)", flush=True)
    return batches


def _train_eval_static_baselines(
    all_batches: List[Dict],
    protocol: int,
    seed: int,
) -> Dict[str, Dict]:
    """
    Train and evaluate static baselines using pre-loaded compact batch dicts.
    No CSV is read here — all data comes from _preload_batches().

    Protocol 1: train R1+R2+R3 (ids 0-89), test Fault (90-99)
    Protocol 2: train R1       (ids  0-29), test Fault (90-99)
    """
    import gc
    import random as _random

    from scipy.interpolate import interp1d
    from scipy.signal import savgol_filter
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.svm import SVR

    _random.seed(seed)
    np.random.seed(seed)

    train_ids = set(range(0, 90)) if protocol == 1 else set(range(0, 30))
    fault_ids = set(range(90, 100))

    train_batches = [b for b in all_batches if b["id"] in train_ids and not b["fault"]]
    fault_batches = [b for b in all_batches if b["id"] in fault_ids or b["fault"]]

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(len(train_batches))
    n_tr = int(len(train_batches) * 0.80)
    r_train = [train_batches[i] for i in perm[:n_tr]]
    r_val   = [train_batches[i] for i in perm[n_tr:]]

    def _raman_pen(batches):
        xs, ys = [], []
        for b in batches:
            X      = b["raman"]
            energy = np.abs(X).sum(axis=1) > 0
            sp     = b["pen_sparse"]
            valid  = energy & np.isfinite(sp)
            if not valid.any():
                continue
            X_v  = np.clip(np.nan_to_num(X[valid].astype(np.float64), nan=0.0), 0.0, None)
            X_sg = savgol_filter(X_v, 15, 2, deriv=1, axis=1).astype(np.float32)
            xs.append(X_sg)
            ys.append(sp[valid])
        if not xs:
            return np.empty((0, 2001), dtype=np.float32), np.empty(0)
        return np.concatenate(xs), np.concatenate(ys)

    def _time_pen(batches):
        ts, ys = [], []
        for b in batches:
            t  = b["time"]
            sp = b["pen_sparse"]
            ok = np.isfinite(sp)
            if ok.sum() < 2:
                continue
            t_v = t[ok]
            t_n = (t_v - t_v.min()) / (t_v.max() - t_v.min() + 1e-12)
            ts.append(t_n)
            ys.append(sp[ok])
        if not ts:
            return np.empty(0), np.empty(0)
        return np.concatenate(ts), np.concatenate(ys)

    def _rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
    def _mae(a, b):  return float(np.mean(np.abs(a - b)))
    nan = float("nan")

    def _row(val_r, test_r, val_a, test_a):
        base: Dict = {
            "val_pen_rmse": val_r, "val_pen_mae": val_a, "val_pen_r2": nan,
            "test_pen_rmse": test_r, "test_pen_mae": test_a, "test_pen_r2": nan,
        }
        if protocol == 2:
            base["r2_pen_rmse"] = nan
            base["r3_pen_rmse"] = nan
        return base

    results: Dict[str, Dict] = {}

    # ── Time-only baselines ───────────────────────────────────────────────────
    t_tr, y_tr = _time_pen(r_train)
    t_va, y_va = _time_pen(r_val)
    t_te, y_te = _time_pen(fault_batches)

    if len(t_tr) > 0:
        poly = Pipeline([("poly", PolynomialFeatures(3, include_bias=True)),
                         ("lr",   LinearRegression())])
        poly.fit(t_tr.reshape(-1, 1), y_tr)
        results["cubic_poly"] = _row(
            _rmse(y_va, poly.predict(t_va.reshape(-1, 1))),
            _rmse(y_te, poly.predict(t_te.reshape(-1, 1))),
            _mae(y_va,  poly.predict(t_va.reshape(-1, 1))),
            _mae(y_te,  poly.predict(t_te.reshape(-1, 1))),
        )
        n_bins  = 200
        bins    = np.linspace(0, 1, n_bins + 1)
        bin_idx = np.clip(np.digitize(t_tr, bins) - 1, 0, n_bins - 1)
        mt      = np.array([y_tr[bin_idx == i].mean()
                             if (bin_idx == i).any() else 0.0
                             for i in range(n_bins)])
        centers = 0.5 * (bins[:-1] + bins[1:])
        valid_b = mt > 0
        if valid_b.sum() >= 2:
            mt = interp1d(centers[valid_b], mt[valid_b],
                          kind="linear", fill_value="extrapolate")(centers)
        def _pred_mt(t_n):
            return mt[np.clip(np.digitize(t_n, bins) - 1, 0, n_bins - 1)]
        results["mean_traj"] = _row(_rmse(y_va, _pred_mt(t_va)),
                                    _rmse(y_te, _pred_mt(t_te)),
                                    _mae(y_va,  _pred_mt(t_va)),
                                    _mae(y_te,  _pred_mt(t_te)))
        del t_tr, t_va, t_te, y_tr, y_va, y_te
        gc.collect()
    else:
        results["cubic_poly"] = _row(nan, nan, nan, nan)
        results["mean_traj"]  = _row(nan, nan, nan, nan)

    # ── Spectral baselines ────────────────────────────────────────────────────
    X_tr, y_tr_s = _raman_pen(r_train)
    X_va, y_va_s = _raman_pen(r_val)
    X_te, y_te_s = _raman_pen(fault_batches)

    if len(X_tr) == 0:
        for name in ("pca_pcr", "pls", "svr"):
            results[name] = _row(nan, nan, nan, nan)
    else:
        sc    = StandardScaler()
        Xtr_s = sc.fit_transform(X_tr)
        Xva_s = sc.transform(X_va)
        Xte_s = sc.transform(X_te)
        del X_tr, X_va, X_te          # free un-scaled copies
        gc.collect()

        pca  = PCA(n_components=4)
        Z_tr = pca.fit_transform(Xtr_s)
        lr   = LinearRegression().fit(Z_tr, y_tr_s)
        results["pca_pcr"] = _row(
            _rmse(y_va_s, lr.predict(pca.transform(Xva_s))),
            _rmse(y_te_s, lr.predict(pca.transform(Xte_s))),
            _mae(y_va_s,  lr.predict(pca.transform(Xva_s))),
            _mae(y_te_s,  lr.predict(pca.transform(Xte_s))),
        )

        pls = PLSRegression(n_components=4, scale=False).fit(Xtr_s, y_tr_s)
        results["pls"] = _row(
            _rmse(y_va_s, pls.predict(Xva_s).ravel()),
            _rmse(y_te_s, pls.predict(Xte_s).ravel()),
            _mae(y_va_s,  pls.predict(Xva_s).ravel()),
            _mae(y_te_s,  pls.predict(Xte_s).ravel()),
        )

        # SVR: n_jobs=1 — avoid forking worker processes that each copy the
        # Raman arrays (each worker would add ~500 MB of RAM per process).
        svr_grid = RandomizedSearchCV(
            SVR(kernel="rbf"),
            param_distributions={
                "C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.001, 0.01],
                "epsilon": [0.01, 0.05, 0.1, 0.5],
            },
            n_iter=16, cv=3, scoring="neg_root_mean_squared_error",
            random_state=seed, n_jobs=1,
        )
        idx = rng.choice(len(Xtr_s), min(3000, len(Xtr_s)), replace=False)
        svr_grid.fit(Xtr_s[idx], y_tr_s[idx])
        best = svr_grid.best_estimator_
        results["svr"] = _row(
            _rmse(y_va_s, best.predict(Xva_s)),
            _rmse(y_te_s, best.predict(Xte_s)),
            _mae(y_va_s,  best.predict(Xva_s)),
            _mae(y_te_s,  best.predict(Xte_s)),
        )

        del Xtr_s, Xva_s, Xte_s, y_tr_s, y_va_s, y_te_s
        gc.collect()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Adaptation model helpers (JIT-FT / VAE-JITL)
# ─────────────────────────────────────────────────────────────────────────────

def _find_base_ckpt(base_model: str, protocol: int, seed: int, out_dir: Path) -> Optional[Path]:
    """
    Locate the seeded base-model checkpoint.
    Falls back to the default (non-seeded) checkpoint if the seeded one is absent.
    """
    # 1. Seeded checkpoint produced by this run
    seeded = _ckpt_path(out_dir, base_model, protocol, seed)
    if seeded.exists():
        return seeded
    # 2. Default pre-trained checkpoint
    defaults = _DEFAULT_P2_CKPTS if protocol == 2 else _DEFAULT_P1_CKPTS
    if base_model in defaults:
        p = _ROOT / defaults[base_model]
        if p.exists():
            return p
    return None


def _val_rmse_from_ckpt(base_ckpt: Optional[Path]) -> float:
    """Read val_metrics pen RMSE from a checkpoint file."""
    if base_ckpt is None or not base_ckpt.exists():
        return float("nan")
    import torch
    state = torch.load(str(base_ckpt), map_location="cpu", weights_only=False)
    return _extract_pen(state.get("val_metrics") or {})["RMSE"]


def _run_cvae_jit_ft(
    model: str,
    protocol: int,
    seed: int,
    out_dir: Path,
    device: str,
) -> Optional[Dict]:
    """
    Run scripts/cvae_jit_ft.py and parse the RMSE from stdout.

    Standard mode  (P1): parses the "JIT-FT (local)  X.XXXX" line.
    Regime-ablation (P2): parses the Fault / R2 / R3 rows in the summary table.
    Val RMSE is taken from the base model's checkpoint.
    """
    cfg          = JITFT_CONFIG[model]
    script_model = cfg["script_model"]
    K            = int(cfg.get("K", 150))
    base_model   = cfg["base_model"]

    base_ckpt = _find_base_ckpt(base_model, protocol, seed, out_dir)
    if base_ckpt is None:
        print(f"  [P{protocol},s={seed}] {model}: base checkpoint not found — skipping")
        return None

    val_rmse  = _val_rmse_from_ckpt(base_ckpt)
    adapt_dir = out_dir / model / f"p{protocol}"
    adapt_dir.mkdir(parents=True, exist_ok=True)
    db_cache  = str(adapt_dir / f"jitft_db_s{seed}.npz")
    log_path  = adapt_dir / f"jitft_s{seed}.log"

    cmd = [
        sys.executable, str(_ROOT / "scripts" / "cvae_jit_ft.py"),
        "--model",       script_model,
        "--global-ckpt", str(base_ckpt),
        "--K",           str(K),
        "--device",      device,
        "--seed",        str(seed),
        "--db-cache",    db_cache,
    ]
    if protocol == 2:
        cmd += ["--regime-ablation"]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT) + ":" + env.get("PYTHONPATH", "")

    print(f"  [P{protocol},s={seed}] Running {model} (K={K}) → {log_path.name}")
    with open(log_path, "w") as f:
        ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                             cwd=str(_ROOT), env=env)

    if ret.returncode != 0:
        print(f"  [P{protocol},s={seed}] {model} FAILED (rc={ret.returncode}) "
              f"— see {log_path}")
        return None

    text = log_path.read_text(errors="replace")
    nan  = float("nan")

    if protocol == 2:
        # Regime-ablation summary table rows (format: "  R2  0.1234  ...")
        def _parse_regime(label: str) -> float:
            m = re.search(rf"^\s+{label}\s+([\d.]+)", text, re.MULTILINE)
            return float(m.group(1)) if m else nan

        return {
            "seed":          seed,
            "val_pen_rmse":  val_rmse,
            "val_pen_mae":   nan,
            "val_pen_r2":    nan,
            "r2_pen_rmse":   _parse_regime("R2"),
            "r3_pen_rmse":   _parse_regime("R3"),
            "test_pen_rmse": _parse_regime("Fault"),   # R4
            "test_pen_mae":  nan,
            "test_pen_r2":   nan,
        }
    else:
        # Standard mode: "  CVAE-PI-LSTM JIT-FT (local)  0.1234  0.9999"
        m = re.search(r"JIT-FT \(local\)\s+([\d.]+)", text)
        return {
            "seed":          seed,
            "val_pen_rmse":  val_rmse,
            "val_pen_mae":   nan,
            "val_pen_r2":    nan,
            "test_pen_rmse": float(m.group(1)) if m else nan,
            "test_pen_mae":  nan,
            "test_pen_r2":   nan,
        }


def _run_vae_jitl(
    model: str,
    protocol: int,
    seed: int,
    out_dir: Path,
    device: str,
) -> Dict:
    """
    Run scripts/vae_jitl.py for Protocol 1.

    vae_jitl.py does NOT support --regime-ablation (Protocol 2).
    Protocol 2 results are returned as NaN without running the script.
    """
    nan       = float("nan")
    base_ckpt = _find_base_ckpt(JITFT_CONFIG[model]["base_model"], protocol, seed, out_dir)
    val_rmse  = _val_rmse_from_ckpt(base_ckpt)
    p2_extra  = {"r2_pen_rmse": nan, "r3_pen_rmse": nan} if protocol == 2 else {}

    if protocol == 2:
        print(f"  [P{protocol},s={seed}] vae_jitl: "
              "Protocol 2 not supported — returning NaN")
        return {
            "seed":          seed,
            "val_pen_rmse":  nan,
            "val_pen_mae":   nan,
            "val_pen_r2":    nan,
            "test_pen_rmse": nan,
            "test_pen_mae":  nan,
            "test_pen_r2":   nan,
            **p2_extra,
        }

    K         = int(JITFT_CONFIG[model].get("K", 500))
    adapt_dir = out_dir / model / f"p{protocol}"
    adapt_dir.mkdir(parents=True, exist_ok=True)
    db_cache  = str(adapt_dir / f"vae_jitl_db_s{seed}.npz")
    log_path  = adapt_dir / f"vae_jitl_s{seed}.log"

    cmd = [
        sys.executable, str(_ROOT / "scripts" / "vae_jitl.py"),
        "--K",        str(K),
        "--device",   device,
        "--seed",     str(seed),
        "--db-cache", db_cache,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT) + ":" + env.get("PYTHONPATH", "")

    print(f"  [P{protocol},s={seed}] Running vae_jitl (K={K}) → {log_path.name}")
    with open(log_path, "w") as f:
        ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                             cwd=str(_ROOT), env=env)

    if ret.returncode != 0:
        print(f"  [P{protocol},s={seed}] vae_jitl FAILED (rc={ret.returncode}) "
              f"— see {log_path}")
        return {"seed": seed,
                "val_pen_rmse": nan, "val_pen_mae": nan, "val_pen_r2": nan,
                "test_pen_rmse": nan, "test_pen_mae": nan, "test_pen_r2": nan}

    text = log_path.read_text(errors="replace")
    m    = re.search(r"Penicillin\s+RMSE\s*=\s*([\d.]+)", text)
    return {
        "seed":          seed,
        "val_pen_rmse":  val_rmse,
        "val_pen_mae":   nan,
        "val_pen_r2":    nan,
        "test_pen_rmse": float(m.group(1)) if m else nan,
        "test_pen_mae":  nan,
        "test_pen_r2":   nan,
    }


def _run_adaptation(
    model: str,
    protocol: int,
    seed: int,
    out_dir: Path,
    device: str,
) -> Optional[Dict]:
    """Dispatch to the correct adaptation runner."""
    if model == "vae_jitl":
        return _run_vae_jitl(model, protocol, seed, out_dir, device)
    return _run_cvae_jit_ft(model, protocol, seed, out_dir, device)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate(results: List[Dict], pen_norm: float) -> Dict:
    """
    Compute summary statistics across seeds for one (protocol, model) pair.

    Returns mean, std, variance, NRMSE%, Δ% for val and test RMSE,
    plus mean R2/R3 RMSE when present (Protocol 2).
    """
    def _v(key: str) -> List[float]:
        return [
            r[key] for r in results
            if isinstance(r.get(key), (int, float)) and not np.isnan(float(r[key]))
        ]

    val_v  = _v("val_pen_rmse")
    test_v = _v("test_pen_rmse")

    m_val  = float(np.mean(val_v))  if val_v  else float("nan")
    s_val  = float(np.std(val_v))   if val_v  else float("nan")
    m_test = float(np.mean(test_v)) if test_v else float("nan")
    s_test = float(np.std(test_v))  if test_v else float("nan")

    nrmse = m_test / pen_norm * 100 if not np.isnan(m_test) else float("nan")
    delta = (
        (m_test - m_val) / m_val * 100
        if (not np.isnan(m_val) and not np.isnan(m_test) and m_val > 0)
        else float("nan")
    )

    agg: Dict = {
        "n_seeds":        len(test_v),
        "mean_val_rmse":  m_val,
        "std_val_rmse":   s_val,
        "var_val_rmse":   s_val ** 2 if not np.isnan(s_val) else float("nan"),
        "mean_test_rmse": m_test,
        "std_test_rmse":  s_test,
        "var_test_rmse":  s_test ** 2 if not np.isnan(s_test) else float("nan"),
        "nrmse_pct":      nrmse,
        "delta_pct":      delta,
    }

    # R2 / R3 generalisation (Protocol 2 only)
    for src_key, dst_key in [("r2_pen_rmse", "mean_r2_rmse"),
                              ("r3_pen_rmse", "mean_r3_rmse")]:
        vs = _v(src_key)
        if vs:
            agg[dst_key]             = float(np.mean(vs))
            agg[dst_key + "_std"]    = float(np.std(vs))
            agg[dst_key + "_var"]    = float(np.std(vs)) ** 2

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_protocol_csv(results: List[Dict], path: Path) -> None:
    """Write per-seed rows + mean/std summary rows."""
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_cols = [k for k in results[0].keys() if k != "seed"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed"] + metric_cols)
        writer.writeheader()
        for r in results:
            writer.writerow({
                k: (f"{v:.6f}" if isinstance(v, float) else v)
                for k, v in r.items()
            })
        # Summary rows
        for label, fn in [("mean", np.mean), ("std", np.std)]:
            row = {"seed": label}
            for col in metric_cols:
                vs = [r[col] for r in results
                      if isinstance(r.get(col), float) and not np.isnan(r[col])]
                row[col] = f"{fn(vs):.6f}" if vs else "nan"   # type: ignore[arg-type]
            writer.writerow(row)


def _save_summary_csv(
    summary: Dict[Tuple, Dict],
    models: List[str],
    protocols: List[int],
    path: Path,
) -> None:
    """Write the combined summary CSV (one row per protocol×model)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for protocol in protocols:
        for model in models:
            key = (protocol, model)
            if key not in summary:
                continue
            row: Dict = {"protocol": protocol, "model": model}
            row.update(summary[key])
            rows.append(row)
    if not rows:
        return
    cols = ["protocol", "model"] + [
        k for k in rows[0] if k not in ("protocol", "model")
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: (f"{v:.6f}" if isinstance(v, float) else v)
                for k, v in row.items()
            })


def _fmt(v: float, digits: int = 3) -> str:
    return "?" if (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}"


def _print_paper_table(
    summary: Dict[Tuple, Dict],
    models: List[str],
    seeds: List[int],
    protocols: List[int],
) -> None:
    """Print a paper-style benchmark table (mean ± std, NRMSE%, Δ%)."""
    W = 128
    print(f"\n{'═' * W}")
    print(f"  BENCHMARK RESULTS — Multi-Seed Summary  "
          f"(seeds={seeds},  n={len(seeds)})")
    print(f"{'═' * W}")

    for protocol in protocols:
        p_label = ("Train R1–R3, Test R4 (Fault)" if protocol == 1
                   else "Train R1 only, Test R4 (Fault) + R2/R3 generalisation")
        print(f"\n  Protocol {protocol}: {p_label}")

        hdr_r2r3 = "   R2 RMSE   R3 RMSE" if protocol == 2 else ""
        print(
            f"  {'Model':<32}  {'Val RMSE (mean±std)':>20}  "
            f"{'Test RMSE (mean±std)':>20}  {'NRMSE%':>8}  {'Δ%':>9}"
            f"{hdr_r2r3}"
        )
        print(f"  {'─' * (W - 2)}")

        for model in models:
            key = (protocol, model)
            if key not in summary:
                print(f"  {model:<32}  {'—':>20}  {'—':>20}  {'—':>8}  {'—':>9}")
                continue

            agg   = summary[key]
            m_val = agg.get("mean_val_rmse",  float("nan"))
            s_val = agg.get("std_val_rmse",   float("nan"))
            m_tst = agg.get("mean_test_rmse", float("nan"))
            s_tst = agg.get("std_test_rmse",  float("nan"))
            nrmse = agg.get("nrmse_pct",      float("nan"))
            delta = agg.get("delta_pct",      float("nan"))
            n     = agg.get("n_seeds", 0)

            # val_str: "X.XXX ± Y.YYY" or "X.XXX" if std is nan
            nan_v = isinstance(s_val, float) and np.isnan(s_val)
            nan_t = isinstance(s_tst, float) and np.isnan(s_tst)
            val_str  = (f"{_fmt(m_val)} ± {_fmt(s_val)}" if not nan_v
                        else _fmt(m_val))
            test_str = (f"{_fmt(m_tst)} ± {_fmt(s_tst)}" if not nan_t
                        else _fmt(m_tst))
            nm_str  = f"{nrmse:.1f}%" if not (isinstance(nrmse, float) and np.isnan(nrmse)) else "?"
            dp_str  = (f"{delta:+.1f}%"
                       if not (isinstance(delta, float) and np.isnan(delta)) else "?")

            r2r3 = ""
            if protocol == 2:
                r2 = agg.get("mean_r2_rmse", float("nan"))
                r3 = agg.get("mean_r3_rmse", float("nan"))
                r2r3 = f"   {_fmt(r2):>8}   {_fmt(r3):>8}"

            print(
                f"  {model:<32}  {val_str:>20}  {test_str:>20}  "
                f"{nm_str:>8}  {dp_str:>9}{r2r3}  (n={n})"
            )

        print(f"  {'─' * (W - 2)}")

    print(f"\n{'═' * W}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-seed, dual-protocol benchmark sweep.\n\n"
            "Trains and evaluates algorithms under Protocol 1 (R1-R3 train, R4 test)\n"
            "and Protocol 2 (R1 train, R4 test), then reports mean RMSE ± std,\n"
            "variance, NRMSE%, and Δ% in a paper-style table."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=ALL_MODELS + ["all"],
        help=(
            "Model to run, or 'all' to sweep every algorithm. "
            f"train.py models: {TRAINPY_MODELS}. "
            f"Static: {STATIC_MODELS}. "
            f"Adaptation: {ADAPTATION_MODELS}."
        ),
    )
    parser.add_argument(
        "--protocol",
        default="both",
        choices=["1", "2", "both"],
        help=(
            "Protocol(s) to run (default: both). "
            "1 = Train R1-R3, Test R4.  "
            "2 = Train R1, Test R4 (also reports R2/R3)."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int, nargs="+", default=None,
        help="Explicit seed list (e.g. --seeds 0 1 2 3 4).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int, default=DEFAULT_N_SEEDS,
        help=f"Number of seeds when --seeds is not given (default: {DEFAULT_N_SEEDS}).",
    )
    parser.add_argument(
        "--epochs",
        type=int, default=None,
        help="Override n_epochs from config (applies to train.py models only).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force device (cpu / cuda / cuda:0). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip seeds whose checkpoint / cache files already exist.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; aggregate from existing checkpoints and cache files.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Root output directory (default: outputs/seed_sweep/<model> for a "
            "single model, or outputs/seed_sweep for --model all)."
        ),
    )
    parser.add_argument(
        "--pen-norm",
        type=float, default=PEN_NORM,
        help=(
            f"NRMSE denominator in g/L (default: {PEN_NORM} g/L). "
            "Equal to the mean penicillin concentration at sparse test samples."
        ),
    )
    parser.add_argument(
        "--parallel",
        type=int, default=4,
        help=(
            "Number of train.py jobs to run simultaneously on the GPU "
            "(default: 4).  Each job uses ~200-400 MB of VRAM; tune this "
            "to fit your GPU memory budget (e.g. --parallel 3 for 6 GB)."
        ),
    )
    args = parser.parse_args()

    # ── Resolve models ────────────────────────────────────────────────────────
    if args.model is None or args.model == "all":
        models = list(ALL_MODELS)
    else:
        models = [args.model]

    # ── Resolve protocols ─────────────────────────────────────────────────────
    protocols = [1, 2] if args.protocol == "both" else [int(args.protocol)]

    # ── Resolve seeds ─────────────────────────────────────────────────────────
    if args.seeds:
        seeds = list(args.seeds)
    else:
        pool  = DEFAULT_SEEDS + list(range(2000, 2100))
        seeds = pool[:args.n_seeds]

    # ── Resolve output directory ──────────────────────────────────────────────
    # All paths are nested under out_dir/model/p{N}/... so out_dir itself
    # should NOT include the model name (avoids double-nesting like
    # outputs/seed_sweep/pi_lstm/pi_lstm/p1/...).
    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    device   = args.device or ""
    parallel = max(1, args.parallel)

    print(f"\n{'═' * 70}")
    print(f"  MULTI-SEED BENCHMARK SWEEP")
    print(f"  Models    : {models}")
    print(f"  Protocols : {protocols}")
    print(f"  Seeds     : {seeds}")
    print(f"  Parallel  : {parallel} concurrent GPU jobs")
    print(f"  Output    : {out_dir}")
    print(f"  PEN_NORM  : {args.pen_norm} g/L  (NRMSE denominator)")
    print(f"{'═' * 70}\n")

    # ── Thread-safe print helper ──────────────────────────────────────────────
    _print_lock = threading.Lock()

    def tprint(*msg_args, **kwargs):
        with _print_lock:
            print(*msg_args, **kwargs)

    # all_results[(protocol, model, seed)] = metrics_row_dict
    all_results: Dict[Tuple, Dict] = {}
    _result_lock = threading.Lock()

    def _store(key, row):
        with _result_lock:
            all_results[key] = row

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1 — Static baselines
    # CSV is loaded ONCE here, then freed.  Each (protocol, seed) call reuses
    # the same compact numpy arrays — no re-reading of the 2 GB file.
    # Run sequentially to avoid having multiple Raman feature matrices (each
    # ~500 MB) alive at the same time.
    # ─────────────────────────────────────────────────────────────────────────
    static_needed = [m for m in models if m in STATIC_MODELS]

    # Pre-load batch data only if static baselines are actually needed
    _batch_cache: List[Dict] = []
    if static_needed and not args.eval_only:
        _batch_cache = _preload_batches(CSV_PATH)

    def _handle_static(protocol: int, seed: int) -> None:
        cache = _static_cache_path(out_dir, protocol, seed)

        if (args.eval_only or args.skip_existing) and cache.exists():
            with open(cache) as fp:
                cached = json.load(fp)
            for name, row in cached.items():
                if name in models:
                    row["seed"] = seed
                    _store((protocol, name, seed), row)
            tprint(f"  [P{protocol},s={seed}] Static baselines: loaded from cache")
            return

        if args.eval_only:
            tprint(f"  [P{protocol},s={seed}] Static baselines: no cache "
                   f"(run without --eval-only to generate)")
            return

        tprint(f"  [P{protocol},s={seed}] Training static baselines ...")
        static_res = _train_eval_static_baselines(_batch_cache, protocol, seed)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w") as fp:
            json.dump(static_res, fp, indent=2)
        for name, row in static_res.items():
            if name in models:
                row["seed"] = seed
                _store((protocol, name, seed), row)
                v = row.get("val_pen_rmse",  float("nan"))
                t = row.get("test_pen_rmse", float("nan"))
                tprint(f"    [P{protocol},s={seed}] {name:<20}  "
                       f"val={_fmt(v)}  test={_fmt(t)}")

    if static_needed:
        combos = [(p, s) for p in protocols for s in seeds]
        # Sequential (max_workers=1): prevents two 500 MB Raman feature
        # matrices from being live simultaneously.
        with ThreadPoolExecutor(max_workers=1) as ex:
            futs = {ex.submit(_handle_static, p, s): (p, s) for p, s in combos}
            for fut in as_completed(futs):
                exc = fut.exception()
                if exc:
                    p, s = futs[fut]
                    tprint(f"  [P{p},s={s}] Static baselines ERROR: {exc}")

    # Free the shared batch cache before GPU training to reclaim RAM
    del _batch_cache
    import gc as _gc
    import ctypes as _ct
    _gc.collect()
    try:
        _ct.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1.5 — Encoder training (one encoder per LSTM model that needs one)
    # Encoders are trained sequentially (they are CPU+GPU intensive and not
    # easily parallelised without OOM).  LSTM training (Phase 2) is blocked
    # until the encoder for each (model, protocol, seed) is ready.
    # ─────────────────────────────────────────────────────────────────────────
    encoder_needed = [m for m in models if m in ENCODER_CONFIG]

    if encoder_needed and not args.eval_only:
        enc_tasks = [
            (m, p, s)
            for p in protocols
            for s in seeds
            for m in encoder_needed
        ]
        tprint(f"\n{'━' * 70}")
        tprint(f"  PHASE 1.5 — Encoder training  "
               f"({len(enc_tasks)} encoders, 1 at a time)")
        tprint(f"{'━' * 70}")

        for m, p, s in enc_tasks:
            enc_ckpt   = _enc_ckpt_path(out_dir, m, p, s)
            enc_scaler = _enc_scaler_path(out_dir, m, p, s)

            if args.skip_existing and enc_ckpt.exists() and enc_scaler.exists():
                tprint(f"  [P{p},s={s}] {m} encoder: exists — skipping")
                continue

            rc = _run_encoder_train(m, p, s, out_dir, device)
            if rc != 0:
                enc_log = out_dir / m / f"p{p}" / f"encoder_s{s}.log"
                tprint(f"  [P{p},s={s}] {m} encoder: FAILED (rc={rc}) "
                       f"— see {enc_log}")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2 — train.py models
    # Collect ALL (model, protocol, seed) tasks, then fire `parallel` at a time.
    # Each subprocess runs train.py independently with its own CUDA context —
    # multiple processes share GPU VRAM and compute concurrently.
    # ─────────────────────────────────────────────────────────────────────────
    trainpy_needed = [m for m in models if m in TRAINPY_MODELS]

    def _handle_trainpy(model: str, protocol: int, seed: int) -> None:
        ckpt = _ckpt_path(out_dir, model, protocol, seed)

        # Resolve per-seed encoder paths (None for models without an encoder)
        raman_ckpt_arg   = None
        raman_scaler_arg = None
        if model in ENCODER_CONFIG:
            enc_ckpt   = _enc_ckpt_path(out_dir, model, protocol, seed)
            enc_scaler = _enc_scaler_path(out_dir, model, protocol, seed)
            if enc_ckpt.exists() and enc_scaler.exists():
                raman_ckpt_arg   = enc_ckpt
                raman_scaler_arg = enc_scaler
            else:
                tprint(f"  [P{protocol},s={seed}] {model}: per-seed encoder not found, "
                       "falling back to default checkpoint")

        if not args.eval_only:
            if args.skip_existing and ckpt.exists():
                tprint(f"  [P{protocol},s={seed}] {model}: "
                       "checkpoint exists — skipping")
            else:
                rc = _run_train(model, protocol, seed, out_dir,
                                args.epochs, device, [],
                                raman_ckpt=raman_ckpt_arg,
                                raman_scaler=raman_scaler_arg)
                if rc != 0:
                    tprint(f"  [P{protocol},s={seed}] {model}: "
                           f"FAILED (rc={rc}) — "
                           f"see {_log_path(out_dir, model, protocol, seed)}")
                    return

        row = _eval_from_ckpt(model, protocol, seed, out_dir)
        if row is not None:
            _store((protocol, model, seed), row)
            v = row.get("val_pen_rmse",  float("nan"))
            t = row.get("test_pen_rmse", float("nan"))
            tprint(f"  [P{protocol},s={seed}] {model:<30}  "
                   f"val={_fmt(v)}  test={_fmt(t)}")

    if trainpy_needed:
        tasks = [
            (m, p, s)
            for p in protocols
            for s in seeds
            for m in trainpy_needed
        ]
        n_tasks = len(tasks)
        tprint(f"\n{'━' * 70}")
        tprint(f"  PHASE 2 — train.py models  "
               f"({n_tasks} jobs, {parallel} in parallel)")
        tprint(f"{'━' * 70}")

        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futs = {
                ex.submit(_handle_trainpy, m, p, s): (m, p, s)
                for m, p, s in tasks
            }
            done = 0
            for fut in as_completed(futs):
                done += 1
                exc = fut.exception()
                if exc:
                    m, p, s = futs[fut]
                    tprint(f"  [P{p},s={s}] {m}: EXCEPTION: {exc}")
                tprint(f"  [{done}/{n_tasks} done]", end="\r")
        tprint()  # newline after progress counter

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3 — Adaptation models (JIT-FT / VAE-JITL)
    # Run after Phase 2 so base model checkpoints exist.
    # ─────────────────────────────────────────────────────────────────────────
    adapt_needed = [m for m in models if m in ADAPTATION_MODELS]

    def _handle_adaptation(model: str, protocol: int, seed: int) -> None:
        cache = _adapt_cache_path(out_dir, model, protocol, seed)

        if (args.eval_only or args.skip_existing) and cache.exists():
            with open(cache) as fp:
                row = json.load(fp)
            _store((protocol, model, seed), row)
            v = row.get("val_pen_rmse",  float("nan"))
            t = row.get("test_pen_rmse", float("nan"))
            tprint(f"  [P{protocol},s={seed}] {model:<30}  "
                   f"val={_fmt(v)}  test={_fmt(t)}  (cached)")
            return

        if args.eval_only:
            tprint(f"  [P{protocol},s={seed}] {model}: no cache — "
                   "run without --eval-only to generate")
            return

        row = _run_adaptation(model, protocol, seed, out_dir, device or "cuda")
        if row is not None:
            _store((protocol, model, seed), row)
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "w") as fp:
                json.dump(row, fp, indent=2)
            v = row.get("val_pen_rmse",  float("nan"))
            t = row.get("test_pen_rmse", float("nan"))
            tprint(f"  [P{protocol},s={seed}] {model:<30}  "
                   f"val={_fmt(v)}  test={_fmt(t)}")

    if adapt_needed:
        tasks = [
            (m, p, s)
            for p in protocols
            for s in seeds
            for m in adapt_needed
        ]
        tprint(f"\n{'━' * 70}")
        tprint(f"  PHASE 3 — adaptation models  "
               f"({len(tasks)} jobs, {parallel} in parallel)")
        tprint(f"{'━' * 70}")

        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futs = {
                ex.submit(_handle_adaptation, m, p, s): (m, p, s)
                for m, p, s in tasks
            }
            for fut in as_completed(futs):
                exc = fut.exception()
                if exc:
                    m, p, s = futs[fut]
                    tprint(f"  [P{p},s={s}] {m}: EXCEPTION: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # Aggregate per (protocol, model) and print
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  Aggregating results …")

    summary: Dict[Tuple, Dict] = {}
    for protocol in protocols:
        for model in models:
            rows = [
                all_results[(protocol, model, s)]
                for s in seeds
                if (protocol, model, s) in all_results
            ]
            if not rows:
                continue

            # Per-protocol CSV — always nested under model/p{N}/
            csv_out = out_dir / model / f"p{protocol}" / "results.csv"
            _save_protocol_csv(rows, csv_out)

            summary[(protocol, model)] = _aggregate(rows, args.pen_norm)

    _print_paper_table(summary, models, seeds, protocols)

    # Combined summary CSV — at model level for single model, top level for all
    summary_path = (out_dir / models[0] / "summary.csv"
                    if len(models) == 1
                    else out_dir / "summary.csv")
    _save_summary_csv(summary, models, protocols, summary_path)
    print(f"  Summary CSV → {summary_path}\n")


if __name__ == "__main__":
    main()
