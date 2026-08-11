#!/usr/bin/env python
"""
plot_bvae_lstm_fault.py
=======================
Load the β=0.05 PI-LSTM (seq_len=24, best model) and run inference on all
10 Fault/R4 batches (IDs 90–99).  For each batch, saves a PNG showing:
  - Dashed grey  : dense interpolated ground-truth (full time axis)
  - Filled circles: sparse offline lab measurements
  - Solid blue   : model prediction

Usage
-----
    PYTHONPATH=. .venv/bin/python scripts/plot_bvae_lstm_fault.py

Outputs
-------
    outputs/fault_predictions/fault_batch_{i:02d}.png  (10 files)
    outputs/fault_predictions/summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.algorithms  # noqa: F401 — populate REGISTRY
from src.algorithms.base import BaseAlgorithm
from src.data.beta_vae_encoder import BetaVAERamanEncoderV2
from src.data.dataset import TARGET_COLS

# ── paths ─────────────────────────────────────────────────────────────────────
CSV_PATH     = str(_ROOT / "100_Batches_IndPenSim_V3.csv")
VAE_CKPT     = str(_ROOT / "outputs/beta_sweep_e2e/checkpoints/b0.05/vae.pt")
VAE_SCALER   = str(_ROOT / "outputs/beta_sweep_e2e/checkpoints/b0.05/vae_scaler.joblib")
LSTM_CKPT    = str(_ROOT / "outputs/beta_sweep_e2e/checkpoints/b0.05/lstm.pt")
OUT_DIR      = _ROOT / "outputs/fault_predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAULT_IDS    = set(range(90, 100))   # batch IDs 90–99
LATENT_DIM   = 64


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load CSV and extract Fault batches
# ─────────────────────────────────────────────────────────────────────────────

def load_fault_batches(csv_path: str) -> List[pd.DataFrame]:
    print(f"\nLoading CSV: {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Identify batch boundaries by time reset
    t_vals    = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid
    df["_batch_id"] = batch_ids

    # Preserve sparse measurements before interpolation
    df["_penicillin_sparse"] = df[TARGET_COLS[1]].copy()
    df["_biomass_sparse"]    = df[TARGET_COLS[0]].copy()

    # Dense interpolation for model targets
    for tc in TARGET_COLS:
        df[tc] = df[tc].interpolate(method="linear").ffill().bfill()

    fault_batches = []
    for b_id in sorted(FAULT_IDS):
        sub = df[df["_batch_id"] == b_id].reset_index(drop=True)
        if len(sub) >= 50:
            fault_batches.append(sub)
        else:
            print(f"  [WARNING] Batch {b_id} has only {len(sub)} rows — skipped")

    print(f"  Loaded {len(fault_batches)} Fault batches")
    return fault_batches


# ─────────────────────────────────────────────────────────────────────────────
# 2. Encode batches with frozen β-VAE
# ─────────────────────────────────────────────────────────────────────────────

def encode_batches(
    batches: List[pd.DataFrame],
    encoder: BetaVAERamanEncoderV2,
) -> List[pd.DataFrame]:
    encoded = []
    for b in batches:
        b2      = b.copy()
        latents = encoder.encode_dataframe(b2)
        for d in range(latents.shape[1]):
            b2[f"_raman_{d}"] = latents[:, d]
        encoded.append(b2)
    return encoded


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-batch plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_batch(
    time:      np.ndarray,
    pen_dense: np.ndarray,
    pen_sparse: np.ndarray,
    pen_pred:  np.ndarray,
    batch_global_id: int,
    batch_idx: int,
    rmse: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))

    # Ground-truth dense interpolation (reference curve)
    ax.plot(time, pen_dense, color="silver", lw=1.2, linestyle="--",
            label="Ground truth (dense interp.)", zorder=1)

    # Sparse offline lab measurements
    sp_mask = ~np.isnan(pen_sparse)
    ax.scatter(time[sp_mask], pen_sparse[sp_mask],
               color="black", s=40, zorder=3, label="Offline lab measurements")

    # Model prediction
    pred_mask = ~np.isnan(pen_pred)
    ax.plot(time[pred_mask], pen_pred[pred_mask],
            color="#1f77b4", lw=1.8, label=f"β-VAE PI-LSTM (RMSE={rmse:.3f} g/L)",
            zorder=2)

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Penicillin (g/L)")
    ax.set_title(
        f"Fault/R4 — Batch {batch_global_id + 1}  (ID={batch_global_id}, "
        f"batch_idx={batch_idx + 1}/10)"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}  (RMSE={rmse:.4f} g/L)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Fault batches
    fault_batches = load_fault_batches(CSV_PATH)

    # Load encoder
    print("\nLoading β-VAE encoder ...")
    encoder = BetaVAERamanEncoderV2(
        ckpt_path=VAE_CKPT,
        scaler_path=VAE_SCALER,
        device=device,
    )

    # Encode Fault batches
    print("\nEncoding Fault batches ...")
    fault_enc = encode_batches(fault_batches, encoder)

    # Load PI-LSTM
    print(f"\nLoading PI-LSTM from {LSTM_CKPT} ...")
    alg = BaseAlgorithm.load(LSTM_CKPT, device)
    print(f"  Loaded algorithm: {alg.name}")

    # Evaluate
    print("\nRunning evaluation on Fault batches ...")
    results = alg.evaluate(fault_enc)

    y_preds_pen   = results["y_preds_pen"]    # list of (T,) arrays, g/L
    y_sparses_pen = results["y_sparses_pen"]  # list of (T,) sparse arrays

    overall_rmse = results["pen"]["RMSE"]
    print(f"\n  Overall Fault RMSE: {overall_rmse:.4f} g/L")

    # Per-batch plots
    print("\nGenerating per-batch plots ...")
    summary: Dict[str, object] = {
        "overall_rmse": overall_rmse,
        "overall_mae":  results["pen"]["MAE"],
        "overall_r2":   results["pen"]["R2"],
        "batches":      [],
    }

    sorted_fault_ids = sorted(FAULT_IDS)

    for idx, (batch_df, pred, sparse) in enumerate(
        zip(fault_enc, y_preds_pen, y_sparses_pen)
    ):
        global_id = sorted_fault_ids[idx]
        time      = batch_df["Time (h)"].values
        pen_dense = batch_df[TARGET_COLS[1]].values  # dense interpolated GT

        # Per-batch RMSE at sparse points
        sp_mask   = ~np.isnan(sparse) & ~np.isnan(pred)
        if sp_mask.sum() > 0:
            rmse_b = float(np.sqrt(np.mean((sparse[sp_mask] - pred[sp_mask]) ** 2)))
        else:
            rmse_b = float("nan")

        out_path = OUT_DIR / f"fault_batch_{global_id:02d}.png"
        plot_batch(
            time=time,
            pen_dense=pen_dense,
            pen_sparse=sparse,
            pen_pred=pred,
            batch_global_id=global_id,
            batch_idx=idx,
            rmse=rmse_b,
            out_path=out_path,
        )

        summary["batches"].append({
            "batch_id":       global_id,
            "batch_label":    f"Batch {global_id + 1}",
            "n_timepoints":   int(len(time)),
            "n_sparse_meas":  int(sp_mask.sum()),
            "rmse_g_per_L":   rmse_b,
            "plot":           str(out_path),
        })

    # Save summary JSON
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written -> {summary_path}")

    # Print per-batch RMSE table
    print("\n── Per-batch Fault RMSE ──────────────────────────────────────")
    print(f"{'Batch ID':>10}  {'Label':>12}  {'RMSE (g/L)':>12}")
    print("-" * 40)
    for b in summary["batches"]:
        rmse_str = f"{b['rmse_g_per_L']:.4f}" if not np.isnan(b['rmse_g_per_L']) else "  NaN"
        print(f"{b['batch_id']:>10}  {b['batch_label']:>12}  {rmse_str:>12}")
    print("-" * 40)
    print(f"{'Overall':>10}  {'':>12}  {overall_rmse:>12.4f}")
    print()


if __name__ == "__main__":
    main()
