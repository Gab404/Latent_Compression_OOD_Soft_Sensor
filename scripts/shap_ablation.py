#!/usr/bin/env python
"""
shap_ablation.py
================
Publication-quality SHAP ablation study for the CVAE-based penicillin
soft-sensor: Physics-Informed CVAE-PI-LSTM  vs.  Vanilla CVAE-LSTM baseline.

The central question: does the physics (mass-balance) constraint steer the
model towards chemically meaningful Raman bands, or does data alone suffice?

──────────────────────────────────────────────────────────────────────────────
Assumed to be in memory before calling run_ablation() / plot_shap_comparison()
──────────────────────────────────────────────────────────────────────────────
  model_pi_wrapper    nn.Module  cuDNN-safe SHAP wrapper for PI model
  model_base_wrapper  nn.Module  cuDNN-safe SHAP wrapper for baseline model
  X_background        ndarray    (N_bg,  T, 2001) float32 – SHAP background
  X_test              ndarray    (N_test, T, 2001) float32 – samples to explain
  mean_raman_spectrum ndarray    (2001,) float32  – mean raw Raman spectrum
  wavenumbers         ndarray    (2001,) float64  – wavenumber axis (cm⁻¹)

See "Loading reference" near the bottom of this file for how to build them
from the project's checkpoint files.

──────────────────────────────────────────────────────────────────────────────
CLI usage (standalone)
──────────────────────────────────────────────────────────────────────────────
  # Train the vanilla baseline first (one-time):
  python train.py --model cvae_pi_lstm --config configs/cvae_lstm_vanilla.yaml

  # Run the full ablation:
  python shap_ablation.py [--device cuda] [--n-bg 100] [--n-test 50]
                          [--smooth 2.0]  [--shap-batch 10]
                          [--output-dir ./outputs/shap_ablation]
                          [--pi-ckpt   ./checkpoints/cvae_pi_lstm.pt]
                          [--base-ckpt ./checkpoints/cvae_lstm_vanilla.pt]
                          [--csv ./100_Batches_IndPenSim_V3.csv]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn
import shap
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shap_analysis import EndToEndCVAEPILSTM, load_end_to_end_model, prepare_data


# ═════════════════════════════════════════════════════════════════════════════
# § 1  Visual style constants
# ═════════════════════════════════════════════════════════════════════════════

# Matplotlib rcParams applied via rc_context — does not pollute global state
_RC = {
    "font.family":         "DejaVu Sans",
    "font.size":           11,
    "axes.labelsize":      12,
    "axes.titlesize":      12,
    "axes.linewidth":      0.8,
    "xtick.labelsize":     10,
    "ytick.labelsize":     10,
    "xtick.major.width":   0.8,
    "ytick.major.width":   0.8,
    "xtick.minor.width":   0.5,
    "xtick.direction":     "out",
    "ytick.direction":     "out",
    "legend.fontsize":     9,
    "legend.framealpha":   0.90,
    "legend.edgecolor":    "0.75",
    "legend.handlelength": 1.6,
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
}

# Colorblind-safe palette (Paul Tol "Bright" subset)
_C_SPECTRUM  = "#BBBBBB"   # neutral gray  – Raman spectrum line
_C_BASELINE  = "#4477AA"   # blue          – vanilla CVAE-LSTM
_C_PI        = "#EE7733"   # orange        – physics-informed model
_C_ANNOT     = "#555555"   # dark gray     – spectral band markers
_C_FILL_ALPHA = 0.40       # SHAP area transparency
_C_LINE_ALPHA = 0.90       # SHAP edge-line alpha

# Known penicillin-G / fermentation Raman bands (approximate, cm⁻¹)
# Override or extend via the `band_annotations` argument to plot_shap_comparison()
DEFAULT_BAND_ANNOTATIONS: Dict[float, str] = {
     618: "Phenyl\ndeform.",
    1000: "Phe ring\nbreathing",
    1163: "C–O\nstretch",
    1606: "C=C\naromatic",
    1825: "β-lactam\nC=O",
}


# ═════════════════════════════════════════════════════════════════════════════
# § 2  cuDNN-safe SHAP wrapper
# ═════════════════════════════════════════════════════════════════════════════

def make_shap_wrapper(model: EndToEndCVAEPILSTM) -> nn.Module:
    """
    Wrap an EndToEndCVAEPILSTM for use with shap.GradientExplainer.

    Two technical issues are resolved:

    1. Output shape:  GradientExplainer requires (B, n_outputs).
       Our model returns (B,); the wrapper adds the trailing dim → (B, 1).

    2. cuDNN LSTM backward mode:  GradientExplainer.__init__ calls
       ``model.eval()``, which causes "cudnn RNN backward can only be called
       in training mode".  We override ``train()`` / ``eval()`` so the LSTM
       sub-module is always kept in training mode while everything else
       (CVAE BatchNorm layers) stays in eval.  Safe because lstm_dropout=0.0.

    Parameters
    ----------
    model : EndToEndCVAEPILSTM — assembled from load_end_to_end_model()

    Returns
    -------
    wrapper : nn.Module  ready for shap.GradientExplainer
    """
    _lstm = model.lstm_model.lstm   # direct reference for mode control

    class _SHAPWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def train(self, mode: bool = True) -> "_SHAPWrapper":
            super().train(mode)
            _lstm.train(True)      # always in train — required for cuDNN backward
            return self

        def eval(self) -> "_SHAPWrapper":
            return self.train(mode=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.m(x).unsqueeze(-1)   # (B,) → (B, 1)

    return _SHAPWrapper()


# ═════════════════════════════════════════════════════════════════════════════
# § 3  SHAP computation
# ═════════════════════════════════════════════════════════════════════════════

def compute_shap(
    wrapper:      nn.Module,
    X_background: np.ndarray,
    X_test:       np.ndarray,
    device:       torch.device,
    batch_size:   int = 10,
    label:        str = "model",
) -> np.ndarray:
    """
    Compute gradient SHAP values for one model wrapper.

    Parameters
    ----------
    wrapper      : cuDNN-safe wrapper (from make_shap_wrapper)
    X_background : (N_bg,  T, W) float32 — reference distribution
    X_test       : (N_test, T, W) float32 — samples to explain
    device       : torch device
    batch_size   : test samples per GradientExplainer call (reduce if OOM)
    label        : short string used in progress messages

    Returns
    -------
    shap_values : (N_test, T, W) float32
        Positive → feature pushes pen_norm prediction upward.
    """
    bg = torch.tensor(X_background, dtype=torch.float32, device=device)
    explainer = shap.GradientExplainer(wrapper, bg)
    print(f"    [{label}] GradientExplainer ready  "
          f"(background N={len(X_background)}, test N={len(X_test)})")

    parts = []
    n = len(X_test)
    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        chunk = torch.tensor(
            X_test[start:end], dtype=torch.float32, device=device
        )
        sv = explainer.shap_values(chunk)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.array(sv, dtype=np.float32)
        if sv.ndim == 4 and sv.shape[-1] == 1:
            sv = sv.squeeze(-1)          # (B, T, W, 1) → (B, T, W)
        parts.append(sv)
        print(f"    [{label}]   {end:>3}/{n} done")

    result = np.concatenate(parts, axis=0)   # (N_test, T, W)
    print(f"    [{label}] complete — shape {result.shape}")
    return result


def aggregate_shap(shap_values: np.ndarray) -> np.ndarray:
    """
    Collapse (N, T, W) SHAP array to global per-wavenumber importance (W,).

    Aggregation: mean of absolute values over samples (N) and time steps (T).
    This gives an unsigned global importance score for each wavenumber.

    Parameters
    ----------
    shap_values : (N, T, W) float32

    Returns
    -------
    importance : (W,) float32
    """
    return np.abs(shap_values).mean(axis=(0, 1))


# ═════════════════════════════════════════════════════════════════════════════
# § 4  Publication-quality comparison figure
# ═════════════════════════════════════════════════════════════════════════════

def plot_shap_comparison(
    wavenumbers:       np.ndarray,
    mean_raman:        np.ndarray,
    shap_base:         np.ndarray,
    shap_pi:           np.ndarray,
    output_path:       str | Path,
    n_bg:              int = 100,
    n_test:            int = 50,
    smooth_sigma:      float = 2.0,
    band_annotations:  Optional[Dict[float, str]] = DEFAULT_BAND_ANNOTATIONS,
    figsize:           Tuple[float, float] = (11.5, 7.5),
) -> plt.Figure:
    """
    Two-panel figure comparing global SHAP importance for baseline vs. PI model.

    Layout
    ------
    (a)  Top panel   : Vanilla CVAE-LSTM baseline
    (b)  Bottom panel: Physics-Informed CVAE-PI-LSTM
    Each panel has a dual Y-axis:
      Left  (gray)  — normalised mean Raman spectrum
      Right (color) — mean |SHAP| per wavenumber (same scale on both panels)

    Parameters
    ----------
    wavenumbers      : (W,) float   — wavenumber axis in cm⁻¹
    mean_raman       : (W,) float   — mean raw spectrum (any scale)
    shap_base        : (W,) float   — aggregated |SHAP| for baseline
    shap_pi          : (W,) float   — aggregated |SHAP| for PI model
    output_path      : save path (.png / .pdf / .svg)
    n_bg             : number of background samples (for figure caption)
    n_test           : number of test samples (for figure caption)
    smooth_sigma     : Gaussian σ in wavenumber channels applied to SHAP curves
                       (default 2.0 ≈ 2–4 cm⁻¹; set 0 to disable)
    band_annotations : dict {wavenumber_cm: label_str} or None to suppress
    figsize          : (width, height) in inches

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    with plt.rc_context(_RC):

        # ── Pre-process ────────────────────────────────────────────────────────
        raman_norm = mean_raman / (mean_raman.max() + 1e-12)

        if smooth_sigma > 0:
            s_base = gaussian_filter1d(shap_base, sigma=smooth_sigma)
            s_pi   = gaussian_filter1d(shap_pi,   sigma=smooth_sigma)
        else:
            s_base = shap_base.copy()
            s_pi   = shap_pi.copy()

        # Shared SHAP Y-axis max (identical on both panels for direct comparison)
        shap_ymax = max(s_base.max(), s_pi.max()) * 1.12

        # ── Figure skeleton ────────────────────────────────────────────────────
        fig, (ax_a, ax_b) = plt.subplots(
            2, 1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"hspace": 0.08},
        )

        # ── Inner helper: render one panel ────────────────────────────────────
        def _panel(
            ax_spec:     plt.Axes,
            shap_curve:  np.ndarray,
            shap_color:  str,
            model_label: str,
            panel_id:    str,
        ) -> plt.Axes:
            """
            Draw a single subplot.

            Returns the secondary (SHAP) axes so the caller can sync limits.
            """
            ax_shap = ax_spec.twinx()

            # ── 1. Raman spectrum (left axis, gray) ──────────────────────────
            ax_spec.plot(
                wavenumbers, raman_norm,
                color=_C_SPECTRUM, linewidth=1.0, alpha=0.80,
                label="Mean Raman spectrum (normalised)",
                zorder=2,
            )
            # Allow extra headroom above 1.0 for band annotation text
            ax_spec.set_ylim(-0.05, 1.55)
            ax_spec.set_ylabel("Normalised\nRaman Intensity", fontsize=11,
                               labelpad=4)
            ax_spec.tick_params(axis="y", labelcolor=_C_SPECTRUM)
            ax_spec.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
            ax_spec.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v:.1f}")
            )

            # Spine cleanup (left/bottom only; twinx provides right spine)
            for sp in ("top", "right"):
                ax_spec.spines[sp].set_visible(False)

            # ── 2. SHAP importance (right axis, colored) ─────────────────────
            ax_shap.fill_between(
                wavenumbers, shap_curve,
                alpha=_C_FILL_ALPHA,
                color=shap_color,
                linewidth=0,
                zorder=3,
                label=f"Mean $|\\phi_k|$  [{model_label}]",
            )
            ax_shap.plot(
                wavenumbers, shap_curve,
                color=shap_color, linewidth=0.9,
                alpha=_C_LINE_ALPHA, zorder=4,
            )
            ax_shap.set_ylim(0, shap_ymax)
            ax_shap.set_ylabel(
                r"Mean $\left|\phi_k\right|$" + "\n" + r"(pen$_\mathrm{norm}$ units)",
                fontsize=11, labelpad=6,
            )
            ax_shap.tick_params(axis="y", labelcolor=shap_color)
            # Scientific notation with consistent exponent
            _exp = int(np.floor(np.log10(shap_ymax))) if shap_ymax > 0 else 0
            ax_shap.yaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda v, _: rf"${v / 10**_exp:.1f}\!\times\!10^{{{_exp}}}$"
                    if v > 0 else "0"
                )
            )
            ax_shap.yaxis.set_major_locator(
                mticker.MaxNLocator(nbins=4, min_n_ticks=3)
            )
            for sp in ("top", "left"):
                ax_shap.spines[sp].set_visible(False)

            # ── 3. Band annotations (vertical dashed markers) ─────────────────
            if band_annotations:
                wn_lo, wn_hi = float(wavenumbers.min()), float(wavenumbers.max())
                for wn_c, lbl in band_annotations.items():
                    if wn_lo <= wn_c <= wn_hi:
                        ax_spec.axvline(
                            wn_c, ymin=0.0, ymax=0.72,   # up to ~1.1 in data coords
                            color=_C_ANNOT, linewidth=0.7,
                            linestyle="--", alpha=0.55, zorder=1,
                        )
                        ax_spec.text(
                            wn_c, 1.16, lbl,
                            fontsize=6.5, ha="center", va="bottom",
                            color=_C_ANNOT, linespacing=1.35,
                            rotation=0,
                        )

            # ── 4. Panel label + title ─────────────────────────────────────────
            ax_spec.text(
                -0.075, 1.0, panel_id,
                transform=ax_spec.transAxes,
                fontsize=13, fontweight="bold", va="top",
            )
            ax_spec.set_title(
                model_label,
                fontsize=11, fontweight="normal",
                pad=3, loc="left",
                color="#333333",
            )

            # ── 5. Legend (combine both axes into one legend box) ─────────────
            h_spec, l_spec = ax_spec.get_legend_handles_labels()
            h_shap, l_shap = ax_shap.get_legend_handles_labels()
            ax_spec.legend(
                h_spec + h_shap,
                l_spec + l_shap,
                loc="upper right",
                frameon=True,
                fontsize=9,
                ncol=1,
            )

            return ax_shap

        # ── Draw panels ────────────────────────────────────────────────────────
        ax_a_shap = _panel(
            ax_spec      = ax_a,
            shap_curve   = s_base,
            shap_color   = _C_BASELINE,
            model_label  = "Baseline CVAE-LSTM  —  data-driven, no physics constraint  (λ = 0)",
            panel_id     = "(a)",
        )
        ax_b_shap = _panel(
            ax_spec      = ax_b,
            shap_curve   = s_pi,
            shap_color   = _C_PI,
            model_label  = "Physics-Informed CVAE-PI-LSTM  —  mass-balance regularised  (λ > 0)",
            panel_id     = "(b)",
        )

        # Enforce identical SHAP Y-axis limits (already set inside _panel,
        # but make explicit to guard against floating-point rounding)
        ax_a_shap.set_ylim(0, shap_ymax)
        ax_b_shap.set_ylim(0, shap_ymax)

        # ── Shared X-axis cosmetics ────────────────────────────────────────────
        ax_b.set_xlabel(r"Raman Wavenumber  (cm$^{-1}$)", fontsize=12)
        ax_b.xaxis.set_major_locator(mticker.MultipleLocator(200))
        ax_b.xaxis.set_minor_locator(mticker.MultipleLocator(50))
        ax_b.tick_params(axis="x", which="both")
        for sp in ("top", "right"):
            ax_b.spines[sp].set_visible(False)

        # ── Figure-level caption / suptitle ───────────────────────────────────
        _smooth_note = (
            rf"Gaussian-smoothed ($\sigma = {smooth_sigma:.0f}$ channels)"
            if smooth_sigma > 0 else "unsmoothed"
        )
        fig.suptitle(
            r"Global Raman Feature Importance — SHAP Ablation Study"
            "\n"
            rf"GradientExplainer · $N_\mathrm{{bg}}={n_bg}$ training windows · "
            rf"$N_\mathrm{{test}}={n_test}$ fault-test windows · "
            rf"Mean $|\phi_k|$ {_smooth_note}",
            fontsize=10,
            color="#444444",
            y=1.01,
        )

        # ── Layout & save ──────────────────────────────────────────────────────
        fig.tight_layout()

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved → {out}")

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# § 5  Top-band summary table
# ═════════════════════════════════════════════════════════════════════════════

def print_top_bands(
    wavenumbers:  np.ndarray,
    importance:   np.ndarray,
    label:        str,
    n:            int = 15,
) -> None:
    """Print the top-N most important wavenumbers for one model."""
    top_idx = np.argsort(importance)[::-1][:n]
    print(f"\n  ── Top-{n} wavenumbers  [{label}] ──")
    print(f"  {'Rank':>4}  {'Wavenumber (cm⁻¹)':>18}  {'Mean |SHAP|':>12}")
    print(f"  {'────':>4}  {'──────────────────':>18}  {'───────────':>12}")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:>4}  {wavenumbers[int(idx)]:>18.1f}  "
              f"{importance[int(idx)]:>12.4e}")


# ═════════════════════════════════════════════════════════════════════════════
# § 6  Main ablation orchestrator
# ═════════════════════════════════════════════════════════════════════════════

def run_ablation(
    model_pi_wrapper:    nn.Module,
    model_base_wrapper:  nn.Module,
    X_background:        np.ndarray,
    X_test:              np.ndarray,
    mean_raman_spectrum: np.ndarray,
    wavenumbers:         np.ndarray,
    device:              torch.device,
    output_dir:          str | Path = "./outputs/shap_ablation",
    shap_batch:          int   = 10,
    smooth_sigma:        float = 2.0,
) -> Dict[str, np.ndarray]:
    """
    Full ablation pipeline: compute SHAP for both models and render the figure.

    Parameters
    ----------
    model_pi_wrapper    : SHAP wrapper for physics-informed model
    model_base_wrapper  : SHAP wrapper for vanilla baseline model
    X_background        : (N_bg,  T, 2001) float32 — SHAP reference samples
    X_test              : (N_test, T, 2001) float32 — samples to explain
    mean_raman_spectrum : (2001,) float32  — mean Raman spectrum for overlay
    wavenumbers         : (2001,) float64  — wavenumber axis (cm⁻¹)
    device              : torch device
    output_dir          : where to save .npy arrays and figures
    shap_batch          : samples per GradientExplainer call
    smooth_sigma        : Gaussian σ for the plot (does not affect saved .npy)

    Returns
    -------
    dict with keys:
        shap_base       (N_test, T, 2001) float32 — raw SHAP, baseline model
        shap_pi         (N_test, T, 2001) float32 — raw SHAP, PI model
        importance_base (2001,)           float32 — aggregated importance
        importance_pi   (2001,)           float32 — aggregated importance
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_bg   = len(X_background)
    n_test = len(X_test)

    # ── Compute SHAP ───────────────────────────────────────────────────────────
    print("\n  ── SHAP: Baseline CVAE-LSTM ──────────────────────────────────────")
    sv_base = compute_shap(
        model_base_wrapper, X_background, X_test, device,
        batch_size=shap_batch, label="BASE",
    )

    print("\n  ── SHAP: Physics-Informed CVAE-PI-LSTM ───────────────────────────")
    sv_pi = compute_shap(
        model_pi_wrapper, X_background, X_test, device,
        batch_size=shap_batch, label="PI  ",
    )

    # ── Aggregate ──────────────────────────────────────────────────────────────
    imp_base = aggregate_shap(sv_base)
    imp_pi   = aggregate_shap(sv_pi)

    # ── Persist raw SHAP arrays ────────────────────────────────────────────────
    np.save(out / "shap_values_base.npy", sv_base)
    np.save(out / "shap_values_pi.npy",   sv_pi)
    np.save(out / "importance_base.npy",  imp_base)
    np.save(out / "importance_pi.npy",    imp_pi)
    print(f"\n  Saved shap_values_base.npy  {sv_base.shape}")
    print(f"  Saved shap_values_pi.npy    {sv_pi.shape}")

    # ── Console summary ────────────────────────────────────────────────────────
    print_top_bands(wavenumbers, imp_base, label="Baseline CVAE-LSTM")
    print_top_bands(wavenumbers, imp_pi,   label="CVAE-PI-LSTM")

    # ── Publication figure (PNG + PDF) ─────────────────────────────────────────
    print("\n  ── Generating publication figures ────────────────────────────────")
    for ext in ("png", "pdf"):
        plot_shap_comparison(
            wavenumbers        = wavenumbers,
            mean_raman         = mean_raman_spectrum,
            shap_base          = imp_base,
            shap_pi            = imp_pi,
            output_path        = out / f"shap_ablation_comparison.{ext}",
            n_bg               = n_bg,
            n_test             = n_test,
            smooth_sigma       = smooth_sigma,
        )

    print(f"\n  All outputs in  {out}/")
    return {
        "shap_base":       sv_base,
        "shap_pi":         sv_pi,
        "importance_base": imp_base,
        "importance_pi":   imp_pi,
    }


# ═════════════════════════════════════════════════════════════════════════════
# § 7  CLI entry point  (standalone execution)
# ═════════════════════════════════════════════════════════════════════════════

# ── Loading reference ──────────────────────────────────────────────────────────
# When running as a script the variables are built from disk.
# When imported as a module they can be passed in directly (e.g. from a Jupyter
# notebook that already ran shap_analysis.py).
#
# model_pi_wrapper, model_base_wrapper:
#
#   from shap_analysis import load_end_to_end_model
#   from shap_ablation import make_shap_wrapper
#
#   model_pi, _   = load_end_to_end_model(
#       cvae_ckpt   = "./checkpoints/cvae_best.pt",
#       cvae_scaler = "./checkpoints/cvae_scaler.joblib",
#       lstm_ckpt   = "./checkpoints/cvae_pi_lstm.pt",
#       device      = device,
#   )
#   model_base, _ = load_end_to_end_model(
#       cvae_ckpt   = "./checkpoints/cvae_best.pt",
#       cvae_scaler = "./checkpoints/cvae_scaler.joblib",
#       lstm_ckpt   = "./checkpoints/cvae_lstm_vanilla.pt",
#       device      = device,
#   )
#   model_pi_wrapper   = make_shap_wrapper(model_pi)
#   model_base_wrapper = make_shap_wrapper(model_base)
#
# X_background, X_test, wavenumber_labels:
#
#   from shap_analysis import prepare_data
#   X_background, X_test, wn_labels = prepare_data(
#       csv_path = "./100_Batches_IndPenSim_V3.csv",
#       n_bg=100, n_test=50,
#   )
#   wavenumbers         = np.array([float(w) for w in wn_labels])
#   mean_raman_spectrum = X_test[:, -1, :].mean(axis=0)  # last time step
# ──────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SHAP ablation: Physics-Informed vs. Vanilla CVAE-LSTM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pi-ckpt",    default="./checkpoints/cvae_pi_lstm.pt")
    p.add_argument("--base-ckpt",  default="./checkpoints/cvae_lstm_vanilla.pt")
    p.add_argument("--csv",        default="./100_Batches_IndPenSim_V3.csv")
    p.add_argument("--device",     default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--n-bg",       type=int,   default=100)
    p.add_argument("--n-test",     type=int,   default=50)
    p.add_argument("--shap-batch", type=int,   default=10)
    p.add_argument("--smooth",     type=float, default=2.0,
                   help="Gaussian σ (channels) for SHAP smoothing in the plot")
    p.add_argument("--output-dir", default="./outputs/shap_ablation")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"\n  Device : {device}")

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n── Loading models ────────────────────────────────────────────────")
    _CVAE_CKPT   = "./checkpoints/cvae_best.pt"
    _CVAE_SCALER = "./checkpoints/cvae_scaler.joblib"

    print("  [PI   ]", end=" ")
    model_pi, _ = load_end_to_end_model(
        cvae_ckpt    = _CVAE_CKPT,
        cvae_scaler  = _CVAE_SCALER,
        lstm_ckpt    = args.pi_ckpt,
        device       = device,
    )

    print("  [BASE ]", end=" ")
    model_base, _ = load_end_to_end_model(
        cvae_ckpt    = _CVAE_CKPT,
        cvae_scaler  = _CVAE_SCALER,
        lstm_ckpt    = args.base_ckpt,
        device       = device,
    )

    model_pi_wrapper   = make_shap_wrapper(model_pi)
    model_base_wrapper = make_shap_wrapper(model_base)
    print("  Both models wrapped for GradientExplainer.")

    # ── Prepare data ──────────────────────────────────────────────────────────
    print("\n── Preparing data ────────────────────────────────────────────────")
    X_background, X_test, wn_labels = prepare_data(
        csv_path  = args.csv,
        n_bg      = args.n_bg,
        n_test    = args.n_test,
    )
    wavenumbers         = np.array([float(w) for w in wn_labels], dtype=np.float64)
    mean_raman_spectrum = X_test[:, -1, :].mean(axis=0)    # (2001,) last time step

    # ── Run ablation ──────────────────────────────────────────────────────────
    run_ablation(
        model_pi_wrapper    = model_pi_wrapper,
        model_base_wrapper  = model_base_wrapper,
        X_background        = X_background,
        X_test              = X_test,
        mean_raman_spectrum = mean_raman_spectrum,
        wavenumbers         = wavenumbers,
        device              = device,
        output_dir          = args.output_dir,
        shap_batch          = args.shap_batch,
        smooth_sigma        = args.smooth,
    )


if __name__ == "__main__":
    main()
