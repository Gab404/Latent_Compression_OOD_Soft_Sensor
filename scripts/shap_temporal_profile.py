#!/usr/bin/env python
"""
scripts/shap_temporal_profile.py
=================================
Temporal SHAP importance profile averaged over all available test samples,
with a ±1 sigma variance band.

For each model (CVAE-PI-LSTM and beta-VAE-PI-LSTM), this script:
  1. Recomputes SHAP values over all available test windows (up to --n-test)
     using GradientExplainer on faulty-batch windows at sparse measurement points.
  2. For each time step t in the 24-step window, aggregates |SHAP| over all
     2001 wavenumbers -> one scalar per (sample, time_step).
  3. Computes mean and std across samples to produce a temporal importance
     profile with a ±1sigma band.
  4. Plots both models on a two-panel figure for direct visual comparison.

Why this matters
----------------
The single-sample heatmap (shap_analysis.py --sample 0) shows one window's
attribution pattern, which may be atypical.  The averaged profile reveals the
systematic temporal structure: are recent spectra more informative than older
ones?  Do the two encoders exploit the temporal context differently?

Outputs
-------
  outputs/shap_temporal_profile/temporal_profile_cvae.npy     (198, 24)
  outputs/shap_temporal_profile/temporal_profile_beta_vae.npy (198, 24)
  outputs/shap_temporal_profile/temporal_profile.png          (300 dpi)
  outputs/shap_temporal_profile/temporal_profile.pdf          (vector)

Run from project root:
    PYTHONPATH=. python scripts/shap_temporal_profile.py [--n-test N] [--n-bg N]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Reuse all loading / data-prep / SHAP machinery from shap_analysis.py
from scripts.shap_analysis import (
    load_end_to_end_model,
    prepare_data,
    compute_shap_values,
    MODEL_LABELS,
    MODEL_DEFAULTS,
)

OUTPUT_DIR = Path("./outputs/shap_temporal_profile")

# ── Visual constants ─────────────────────────────────────────────────────────
STYLE = {
    "cvae":     {"color": "#2980b9", "label": "CVAE-PI-LSTM"},
    "beta_vae": {"color": "#27ae60", "label": "\u03b2-VAE-PI-LSTM"},
}

RC = {
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.linewidth":    0.9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.size":  4,
    "ytick.major.size":  4,
    "legend.frameon":    False,
    "figure.dpi":        150,
}


# ─────────────────────────────────────────────────────────────────────────────
# Core: temporal aggregation
# ─────────────────────────────────────────────────────────────────────────────

def temporal_importance(shap_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce (N, T, L) SHAP array to a per-time-step importance profile.

    For each sample s and time step t we compute the mean absolute SHAP
    over all 2001 wavenumbers:

        importance[s, t] = mean_l |SHAP[s, t, l]|

    Then we return the mean and std across the N samples.

    Parameters
    ----------
    shap_values : (N, T, L) float32 — raw GradientExplainer output

    Returns
    -------
    mean_profile : (T,) float64  — mean importance per time step
    std_profile  : (T,) float64  — std  importance per time step
    """
    # Per-sample, per-timestep mean |SHAP| over wavenumbers
    per_sample = np.abs(shap_values).mean(axis=2)   # (N, T)
    return per_sample.mean(axis=0), per_sample.std(axis=0)  # (T,), (T,)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_temporal_profiles(
    profiles: dict,          # {model_key: {"mean": (T,), "std": (T,), "n": int}}
    seq_len:  int,
    out_dir:  Path,
) -> None:
    """
    Two-panel figure: one panel per model, temporal importance ± 1sigma.

    Parameters
    ----------
    profiles : dict mapping model_key -> {"mean", "std", "n", "shap_shape"}
    seq_len  : length of the LSTM window (24)
    out_dir  : destination directory
    """
    time_labels = [f"t\u2212{seq_len - 1 - i}" if i < seq_len - 1 else "t"
                   for i in range(seq_len)]
    x = np.arange(seq_len)

    model_keys = list(profiles.keys())   # preserve insertion order

    with plt.rc_context(RC):
        fig, axes = plt.subplots(
            1, len(model_keys),
            figsize=(6.5 * len(model_keys), 4.8),
            sharey=True,
        )
        if len(model_keys) == 1:
            axes = [axes]

        # Use the largest y-range across all models so both panels share the
        # same scale (driven by whichever model has the highest |SHAP| values).
        global_ymax = max(
            (profiles[k]["mean"] + profiles[k]["std"]).max()
            for k in model_keys
        ) * 1.05   # 5 % headroom

        for ax, key in zip(axes, model_keys):
            style   = STYLE[key]
            p       = profiles[key]
            mean    = p["mean"]
            std     = p["std"]
            n       = p["n"]
            color   = style["color"]
            label   = style["label"]

            ax.fill_between(x, mean - std, mean + std,
                            color=color, alpha=0.20, zorder=2,
                            label=r"$\pm 1\sigma$")
            ax.plot(x, mean,
                    color=color, linewidth=2.0, zorder=3,
                    label="Mean |SHAP|")

            ax.set_xticks(x[::2])                        # every other label
            ax.set_xticklabels(time_labels[::2], rotation=45, ha="right", fontsize=9)
            ax.set_xlabel("Time step in sequence window", fontsize=11)
            ax.set_ylabel("Mean |SHAP value|  (pen_norm units)", fontsize=11)
            ax.set_xlim(-0.5, seq_len - 0.5)
            ax.set_ylim(0.0, global_ymax)
            ax.set_title(
                f"{label}\n"
                f"n = {n} test windows  |  aggregated over 2 001 wavenumbers",
                fontsize=11,
            )

            # Highlight the most recent time step
            ax.axvline(seq_len - 1, color="grey", linewidth=0.8,
                       linestyle="--", zorder=1, alpha=0.7)

            handles, labels_leg = ax.get_legend_handles_labels()
            # Reverse so mean line is first in legend
            ax.legend(handles[::-1], labels_leg[::-1], fontsize=10, loc="upper left")

        fig.suptitle(
            "Temporal SHAP Importance Profile  —  Faulty Batch Windows",
            fontsize=13, y=1.02,
        )
        plt.tight_layout()

        png_path = out_dir / "temporal_profile.png"
        pdf_path = out_dir / "temporal_profile.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path,           bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved PNG  ->  {png_path}")
    print(f"  Saved PDF  ->  {pdf_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Temporal SHAP profile averaged over all test samples (CVAE + beta-VAE)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",      default="./100_Batches_IndPenSim_V3.csv")
    p.add_argument("--n-bg",     type=int, default=200,
                   help="Background samples for GradientExplainer")
    p.add_argument("--n-test",   type=int, default=9999,
                   help="Test windows to explain (capped at pool size; use 9999 for all)")
    p.add_argument("--shap-batch", type=int, default=20,
                   help="Test samples per SHAP call (reduce if OOM)")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--device",   default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--models",   nargs="+", default=["cvae", "beta_vae"],
                   choices=["cvae", "beta_vae"],
                   help="Which models to include (default: both)")
    p.add_argument("--skip-compute", action="store_true",
                   help="Skip SHAP recomputation and load existing .npy files "
                        "(must have been saved by a previous run)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n  Device : {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles:    dict = {}
    raw_shap:    dict = {}

    # ── Data (shared across models) ───────────────────────────────────────────
    if not args.skip_compute:
        print("\n── Preparing data ───────────────────────────────────────────────")
        X_background, X_test, wavenumber_labels = prepare_data(
            csv_path = args.csv,
            n_bg     = args.n_bg,
            n_test   = args.n_test,
            seed     = args.seed,
        )
        seq_len = X_test.shape[1]
        print(f"  Using {len(X_test)} test windows  (seq_len={seq_len})")
    else:
        # Infer seq_len from any saved file
        seq_len = None
        for key in args.models:
            npy = out_dir / f"temporal_profile_{key}.npy"
            if npy.exists():
                arr = np.load(npy)
                seq_len = arr.shape[1]
                break
        if seq_len is None:
            raise FileNotFoundError(
                "No saved .npy files found.  Run without --skip-compute first."
            )

    # ── Per-model SHAP computation ────────────────────────────────────────────
    for key in args.models:
        label = MODEL_LABELS[key]
        npy_path = out_dir / f"temporal_profile_{key}.npy"   # (N, T, L) raw SHAP

        if args.skip_compute and npy_path.exists():
            print(f"\n  Loading cached SHAP values for {label} ...")
            shap_values = np.load(npy_path)
            print(f"    shape = {shap_values.shape}")
        else:
            defaults = MODEL_DEFAULTS[key]
            print(f"\n── {label} ───────────────────────────────────────────────")

            model, _ = load_end_to_end_model(
                cvae_ckpt   = defaults["encoder_ckpt"],
                cvae_scaler = defaults["encoder_scaler"],
                lstm_ckpt   = defaults["lstm_ckpt"],
                device      = device,
                model_type  = key,
            )

            print(f"\n  Computing SHAP over {len(X_test)} windows ...")
            shap_values = compute_shap_values(
                model        = model,
                X_background = X_background,
                X_test       = X_test,
                device       = device,
                batch_size   = args.shap_batch,
            )                                     # (N, T, L)

            np.save(npy_path, shap_values)
            print(f"  Saved raw SHAP  ->  {npy_path}")

            # Free GPU memory before next model
            del model
            torch.cuda.empty_cache()

        raw_shap[key] = shap_values

        # ── Aggregate ─────────────────────────────────────────────────────────
        mean_p, std_p = temporal_importance(shap_values)
        profiles[key] = {
            "mean":       mean_p,
            "std":        std_p,
            "n":          shap_values.shape[0],
            "shap_shape": shap_values.shape,
        }

        print(f"  {label}: mean |SHAP| range "
              f"[{mean_p.min():.5f}, {mean_p.max():.5f}]  "
              f"(std range [{std_p.min():.5f}, {std_p.max():.5f}])")

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\n── Plotting ─────────────────────────────────────────────────────")
    plot_temporal_profiles(profiles, seq_len, out_dir)

    # Save aggregated arrays (mean + std per model)
    agg_path = out_dir / "temporal_profile_aggregated.npz"
    save_dict = {}
    for key, p in profiles.items():
        save_dict[f"{key}_mean"] = p["mean"]
        save_dict[f"{key}_std"]  = p["std"]
        save_dict[f"{key}_n"]    = np.array(p["n"])
    np.savez(agg_path, **save_dict)
    print(f"  Saved aggregated stats  ->  {agg_path}")

    print(f"\n  All outputs saved to {out_dir}/")
    print("  Done.\n")


if __name__ == "__main__":
    main()
