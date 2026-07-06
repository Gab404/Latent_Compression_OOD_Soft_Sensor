"""
src/evaluation/metrics.py
=========================
Metric computation utilities shared across all algorithms.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute RMSE, MAE, and R² between two 1-D arrays.

    Returns
    -------
    dict with keys: RMSE, MAE, R2, n
    """
    if len(y_true) == 0:
        return {"RMSE": float("nan"), "MAE": float("nan"),
                "R2": float("nan"), "n": 0}

    err  = y_pred - y_true
    mse  = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(err)))
    ss_r = float(np.sum(err ** 2))
    ss_t = float(np.sum((y_true - y_true.mean()) ** 2))
    r2   = float(1.0 - ss_r / ss_t) if ss_t > 0 else 0.0

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "n": int(len(y_true))}


def aggregate_metrics(
    bio_true: List[np.ndarray],
    bio_pred: List[np.ndarray],
    pen_true: List[np.ndarray],
    pen_pred: List[np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Concatenate per-batch arrays and return aggregated metrics for both targets.

    Returns
    -------
    {"bio": {RMSE, MAE, R2, n}, "pen": {RMSE, MAE, R2, n}}
    """
    bio_t = np.concatenate(bio_true) if bio_true else np.array([])
    bio_p = np.concatenate(bio_pred) if bio_pred else np.array([])
    pen_t = np.concatenate(pen_true) if pen_true else np.array([])
    pen_p = np.concatenate(pen_pred) if pen_pred else np.array([])

    return {
        "bio": compute_metrics(bio_t, bio_p),
        "pen": compute_metrics(pen_t, pen_p),
    }


def print_metrics_table(
    metrics_dict: Dict[str, Dict],
    label: str = "",
):
    """
    Pretty-print a metrics dict of the form:
        {"bio": {"RMSE": ..., "MAE": ..., "R2": ..., "n": ...},
         "pen": {...}}
    """
    prefix = f"  {label:20s}" if label else "  "
    for tgt, tname in [("bio", "Biomass   "), ("pen", "Penicillin")]:
        m = metrics_dict.get(tgt, {})
        if m:
            print(f"{prefix}  [{tname}]  "
                  f"MSE={m.get('MSE', float('nan')):.4f} (g/L)²  "
                  f"RMSE={m.get('RMSE', float('nan')):.3f} g/L  "
                  f"MAE={m.get('MAE', float('nan')):.3f} g/L  "
                  f"R²={m.get('R2', float('nan')):.4f}  "
                  f"(n={m.get('n', '?')})")
