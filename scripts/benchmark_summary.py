#!/usr/bin/env python
"""
benchmark_summary.py
====================
Aggregate all soft-sensor benchmark results into a single comparison table
and bar chart.

Result sources
--------------
  ✓ β-VAE + PI-LSTM      outputs/beta_sweep_e2e/sweep_results.json
  ✓ SINDYc               outputs/sindyc/sindyc_summary.json
  ✓ Neural CDE           outputs/neural_cde/ncde_summary.json  (written at end of training)
  ~ PLS K=64             hardcoded (no JSON saved by pls_pca_baseline_k64.py)
  ~ PCA-PCR K=64         hardcoded
  ~ β-VAE + MLP          hardcoded
  ~ β-VAE z52 linear     hardcoded
  ~ Latent ODE 2-dim     hardcoded
  ~ Latent ODE 64-dim    hardcoded

Outputs
-------
  outputs/benchmark/comparison_table.csv
  outputs/benchmark/fault_rmse_bar.png
  outputs/benchmark/full_rmse_heatmap.png   (R1 / Fault, colour-coded)

Usage
-----
    PYTHONPATH=. .venv/bin/python scripts/benchmark_summary.py

    # Skip Neural CDE if training not finished
    PYTHONPATH=. .venv/bin/python scripts/benchmark_summary.py --no-ncde
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OUT_DIR = _ROOT / "outputs" / "benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded results (from prior runs — no JSON saved by those scripts)
# All values are RMSE in g/L at sparse offline lab-measurement points.
# Evaluated train-on-R1, test-on-Fault.
# ─────────────────────────────────────────────────────────────────────────────

HARDCODED: List[Dict] = [
    {
        "model":       "PLS  (K=64)",
        "type":        "Static baseline",
        "r1_rmse":     0.246,
        "fault_rmse":  0.256,
        "n_params":    None,
        "notes":       "K=64 components, train on R1 spectra",
    },
    {
        "model":       "PCA-PCR  (K=64)",
        "type":        "Static baseline",
        "r1_rmse":     0.311,
        "fault_rmse":  0.318,
        "n_params":    None,
        "notes":       "K=64 principal components + OLS readout",
    },
    {
        "model":       "β-VAE + MLP  (β=0.05)",
        "type":        "Deep static",
        "r1_rmse":     0.445,
        "fault_rmse":  0.549,
        "n_params":    None,
        "notes":       "Frozen encoder, 2-layer MLP readout on z",
    },
    {
        "model":       "β-VAE z₅₂ linear  (β=0.05)",
        "type":        "Deep static",
        "r1_rmse":     0.983,
        "fault_rmse":  1.077,
        "n_params":    2,
        "notes":       "Single latent dim, linear readout: y = 10.05·z₅₂ + 15.12",
    },
    {
        "model":       "Latent ODE  (2-dim, β=0.05)",
        "type":        "Neural ODE",
        "r1_rmse":     0.927,
        "fault_rmse":  1.197,
        "n_params":    None,
        "notes":       "Autonomous ODE on (z₅₂, z₂₇), no control inputs",
    },
    {
        "model":       "Latent ODE  (64-dim, β=0.05)",
        "type":        "Neural ODE",
        "r1_rmse":     0.814,
        "fault_rmse":  1.006,
        "n_params":    None,
        "notes":       "Autonomous ODE on full 64-dim latent",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Loaders for JSON-backed results
# ─────────────────────────────────────────────────────────────────────────────

def load_pi_lstm_results(all_betas: bool = False) -> List[Dict]:
    """Read β-VAE + PI-LSTM entries from beta_sweep_e2e sweep_results.json.

    By default only the two most notable betas are included:
      β=0.5  — best OOD Fault RMSE overall
      β=0.05 — reference model used for Neural CDE / SINDYc / Latent ODE

    Pass all_betas=True to include all 8 sweep betas.
    """
    path = _ROOT / "outputs" / "beta_sweep_e2e" / "sweep_results.json"
    if not path.exists():
        print(f"  [WARN] Not found: {path}")
        return []

    with open(path) as f:
        sweep = json.load(f)

    # Notable betas to show in default mode
    NOTABLE = {0.05, 0.5}

    rows = []
    for entry in sweep:
        beta  = entry.get("beta")
        r1    = entry.get("lstm_rmse_r1val")
        fault = entry.get("lstm_rmse_fault")
        if r1 is None or fault is None:
            continue
        if not all_betas and beta not in NOTABLE:
            continue
        label = "β-VAE + PI-LSTM  (β=0.05, ref.)" if abs(beta - 0.05) < 1e-9 \
                else f"β-VAE + PI-LSTM  (β={beta})"
        rows.append({
            "model":      label,
            "type":       "β-VAE + PI-LSTM",
            "r1_rmse":    round(r1,   4),
            "fault_rmse": round(fault, 4),
            "n_params":   None,
            "notes":      "seq_len=24, frozen β-VAE encoder",
        })
    return rows


def load_sindyc_results() -> List[Dict]:
    """Read SINDYc results.

    Reports two entries:
      • Pareto-optimal threshold (t=0.070): best interpretability/accuracy trade-off
      • Best RMSE threshold (t=0.250): lowest Fault RMSE regardless of sparsity
    """
    sweep_path = _ROOT / "outputs" / "sindyc" / "sweep_results.json"
    if not sweep_path.exists():
        # No sweep JSON — fall back to single hardcoded Pareto value
        return [{
            "model":      "SINDYc  (t=0.070, Pareto)",
            "type":       "SINDYc",
            "r1_rmse":    None,
            "fault_rmse": 10.60,
            "n_params":   None,
            "notes":      "deg=2, threshold=0.070, ctrl: Fs,Fg,pH,T  — Pareto optimum",
        }]

    with open(sweep_path) as f:
        sweep = json.load(f)

    # Pareto-optimal: t=0.070
    pareto_key = min(sweep, key=lambda k: abs(float(k) - 0.070))
    pareto_rmse = sweep[pareto_key]["mean_fault_rmse"]

    # Best RMSE: lowest value
    best_key   = min(sweep, key=lambda k: sweep[k]["mean_fault_rmse"])
    best_rmse  = sweep[best_key]["mean_fault_rmse"]

    rows = [{
        "model":      f"SINDYc  (t={pareto_key}, Pareto)",
        "type":       "SINDYc",
        "r1_rmse":    None,
        "fault_rmse": round(pareto_rmse, 3),
        "n_params":   None,
        "notes":      "deg=2, Pareto optimum: interpretable + low RMSE",
    }]
    # Only add best-RMSE if it differs from Pareto
    if best_key != pareto_key:
        rows.append({
            "model":      f"SINDYc  (t={best_key}, best RMSE)",
            "type":       "SINDYc",
            "r1_rmse":    None,
            "fault_rmse": round(best_rmse, 3),
            "n_params":   None,
            "notes":      "deg=2, best Fault RMSE in threshold sweep",
        })
    return rows


def load_ncde_results(skip: bool = False) -> List[Dict]:
    """Read Neural CDE results from all ncde_summary.json files.

    Scans:
      outputs/neural_cde/ncde_summary.json          (default h=32, rk4)
      outputs/neural_cde_*/ncde_summary.json         (ablation variants)
    """
    if skip:
        return []

    # Collect all candidate directories
    candidates = [_ROOT / "outputs" / "neural_cde"]
    candidates += sorted((_ROOT / "outputs").glob("neural_cde_*"))

    rows = []
    for cdir in candidates:
        path = cdir / "ncde_summary.json"
        if not path.exists():
            if cdir == candidates[0]:
                print(f"  [INFO] Neural CDE summary not yet available ({cdir.name}/ncde_summary.json)")
            continue

        with open(path) as f:
            s = json.load(f)

        fault = s.get("fault_results", {})
        r1    = s.get("r1_results") or {}
        arch  = s.get("architecture", {})
        cfg   = s.get("config", {})
        solver   = arch.get("solver",     cfg.get("solver",     "rk4"))
        h_dim    = arch.get("hidden_dim", cfg.get("hidden_dim", "?"))
        n_params = arch.get("n_params")
        ep       = s.get("epochs_trained", "?")
        dense    = not cfg.get("sparse_loss", True)   # True = dense loss was used

        loss_label = "dense" if dense else "sparse"
        model_label = f"Neural CDE  (h={h_dim}, {solver}, {loss_label})"

        rows.append({
            "model":      model_label,
            "type":       "Neural CDE",
            "r1_rmse":    round(r1.get("overall_rmse"), 4) if r1.get("overall_rmse") else None,
            "fault_rmse": round(fault.get("overall_rmse"), 4),
            "n_params":   n_params,
            "notes":      f"{ep} epochs, path=[z₅₂,z₂₇,Fs,Fg,pH,T,t]",
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Table builder
# ─────────────────────────────────────────────────────────────────────────────

TYPE_ORDER = [
    "Static baseline",
    "Deep static",
    "β-VAE + PI-LSTM",
    "Neural ODE",
    "Neural CDE",
    "SINDYc",
]

TYPE_COLORS = {
    "Static baseline":  "#aec7e8",
    "Deep static":      "#ffbb78",
    "β-VAE + PI-LSTM":  "#98df8a",
    "Neural ODE":       "#ff9896",
    "Neural CDE":       "#c5b0d5",
    "SINDYc":           "#c7c7c7",
}


def build_table(rows: List[Dict]) -> List[Dict]:
    """Sort rows by type order then by Fault RMSE."""
    def sort_key(r):
        t = r.get("type", "ZZZ")
        order = TYPE_ORDER.index(t) if t in TYPE_ORDER else 99
        fault = r.get("fault_rmse") or 999
        return (order, fault)
    return sorted(rows, key=sort_key)


def print_table(rows: List[Dict]) -> None:
    """Pretty-print the comparison table to stdout."""
    header = f"{'Model':<45}  {'R1 RMSE':>9}  {'Fault RMSE':>11}  {'Type'}"
    sep    = "─" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in rows:
        r1    = f"{r['r1_rmse']:.4f}" if r.get("r1_rmse") is not None else "    N/A"
        fault = f"{r['fault_rmse']:.4f}" if r.get("fault_rmse") is not None else "    N/A"
        print(f"  {r['model']:<43}  {r1:>9}  {fault:>11}  {r.get('type','')}")
    print(sep + "\n")


def save_csv(rows: List[Dict], path: Path) -> None:
    import csv
    fields = ["model", "type", "r1_rmse", "fault_rmse", "n_params", "notes"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Table → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_fault_bar(rows: List[Dict], out_dir: Path) -> None:
    """Horizontal bar chart: Fault RMSE for all models, coloured by type."""
    valid = [r for r in rows if r.get("fault_rmse") is not None]
    # Sort ascending (best = left)
    valid = sorted(valid, key=lambda r: r["fault_rmse"])

    labels  = [r["model"] for r in valid]
    values  = [r["fault_rmse"] for r in valid]
    colors  = [TYPE_COLORS.get(r.get("type", ""), "#bbbbbb") for r in valid]

    fig_h = max(4, 0.45 * len(valid) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.8)

    # Value annotations
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.005 * max(values),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", ha="left", fontsize=8,
        )

    ax.set_xlabel("Penicillin RMSE  (g/L)  — Fault/R4 batches")
    ax.set_title("Soft-Sensor Benchmark  —  OOD Generalisation (IndPenSim V3)")
    ax.set_xlim(left=0)
    ax.grid(True, axis="x", alpha=0.35)
    ax.invert_yaxis()

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=c, label=t)
        for t, c in TYPE_COLORS.items()
        if any(r.get("type") == t for r in valid)
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=7, loc="lower right", framealpha=0.8,
    )

    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    out_path = out_dir / "fault_rmse_bar.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar chart → {out_path}")


def plot_r1_vs_fault(rows: List[Dict], out_dir: Path) -> None:
    """
    Scatter: R1 RMSE (x) vs Fault RMSE (y).
    Points near the diagonal have little OOD degradation; points below
    the diagonal generalise *better* OOD (rare, possible if R1 is noisy).
    """
    valid = [r for r in rows if r.get("r1_rmse") is not None and r.get("fault_rmse") is not None]
    if not valid:
        print("  [SKIP] No models with both R1 and Fault RMSE — skipping scatter.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    for r in valid:
        x, y = r["r1_rmse"], r["fault_rmse"]
        c = TYPE_COLORS.get(r.get("type", ""), "#888888")
        ax.scatter(x, y, color=c, s=80, zorder=3, edgecolors="white", linewidths=0.6)
        ax.annotate(
            r["model"],
            (x, y), textcoords="offset points", xytext=(6, 3),
            fontsize=6.5, color="#333333",
        )

    # Diagonal = perfect correlation R1 ↔ Fault
    lim_max = max(
        max(r["r1_rmse"] for r in valid),
        max(r["fault_rmse"] for r in valid),
    ) * 1.1
    ax.plot([0, lim_max], [0, lim_max], ls="--", color="#888888", lw=0.9,
            label="R1 RMSE = Fault RMSE (no OOD degradation)")

    ax.set_xlabel("R1 RMSE  (g/L)  [in-distribution]")
    ax.set_ylabel("Fault RMSE  (g/L)  [out-of-distribution]")
    ax.set_title("In-distribution vs OOD Performance")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    # Legend for types
    legend_handles = [
        mpatches.Patch(facecolor=c, label=t)
        for t, c in TYPE_COLORS.items()
        if any(r.get("type") == t for r in valid)
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=7, loc="upper left", framealpha=0.8,
    )

    fig.tight_layout()
    out_path = out_dir / "r1_vs_fault_scatter.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scatter → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--no-ncde", action="store_true",
                   help="Skip Neural CDE even if summary exists")
    p.add_argument("--all-betas", action="store_true",
                   help="Include all 8 beta values from the PI-LSTM sweep (default: β=0.05 and β=0.5 only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("\n── Benchmark Summary ─────────────────────────────────────────────")

    rows: List[Dict] = []

    # 1. Hardcoded baselines
    rows.extend(HARDCODED)
    print(f"  Loaded {len(HARDCODED)} hardcoded baseline entries.")

    # 2. PI-LSTM from sweep JSON
    pi_lstm_rows = load_pi_lstm_results(all_betas=args.all_betas)
    rows.extend(pi_lstm_rows)
    print(f"  Loaded {len(pi_lstm_rows)} PI-LSTM entries from sweep JSON.")

    # 3. SINDYc
    sindyc_rows = load_sindyc_results()
    rows.extend(sindyc_rows)
    print(f"  Loaded {len(sindyc_rows)} SINDYc entries.")

    # 4. Neural CDE
    ncde_rows = load_ncde_results(skip=args.no_ncde)
    rows.extend(ncde_rows)
    if ncde_rows:
        print(f"  Loaded {len(ncde_rows)} Neural CDE entries.")
    elif not args.no_ncde:
        print("  Neural CDE training still in progress — results not yet available.")

    # Sort
    rows = build_table(rows)

    # Print
    print_table(rows)

    # Save CSV
    csv_path = OUT_DIR / "comparison_table.csv"
    save_csv(rows, csv_path)

    # Save JSON
    json_path = OUT_DIR / "comparison_table.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"  JSON  → {json_path}")

    # Plots
    plot_fault_bar(rows, OUT_DIR)
    plot_r1_vs_fault(rows, OUT_DIR)

    print(f"\nDone. Outputs in: {OUT_DIR}/\n")


if __name__ == "__main__":
    main()
