#!/usr/bin/env python3
"""
run_seeds.py
============
Run train + evaluate for one model across multiple seeds, then aggregate
results into a CSV with per-seed rows and a summary row (mean +/- std).

This is the recommended way to validate an algorithm's robustness.

Usage
-----
    # 5 seeds, default model
    python run_seeds.py --model beta_vae_pi_lstm

    # Custom seeds
    python run_seeds.py --model cdae_pi_lstm --seeds 0 1 2 3 4

    # 10 seeds, regime ablation, limit epochs
    python run_seeds.py --model cvae_pi_lstm --n-seeds 10 --regime-ablation --epochs 50

    # Resume: skip seeds whose checkpoints already exist
    python run_seeds.py --model beta_vae_pi_lstm --skip-existing

    # Evaluate only (no training, just aggregate existing checkpoints)
    python run_seeds.py --model beta_vae_pi_lstm --eval-only

Output
------
    outputs/seed_sweep/<model>/
        results.csv            — per-seed metrics + summary row
        seed_<N>.log           — per-seed training log
        checkpoints/           — per-seed .pt files

Examples
--------
    # Quick validation (3 seeds, 20 epochs, CPU)
    python run_seeds.py --model pi_lstm --n-seeds 3 --epochs 20 --device cpu

    # Full validation for paper (10 seeds, GPU)
    python run_seeds.py --model beta_vae_pi_lstm --n-seeds 10

    # Compare two models
    python run_seeds.py --model beta_vae_pi_lstm --n-seeds 5
    python run_seeds.py --model cvae_pi_lstm --n-seeds 5
    # Then compare outputs/seed_sweep/*/results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SEEDS  = [42, 123, 456, 789, 1024]
DEFAULT_N_SEEDS = 5
OUT_ROOT       = _ROOT / "outputs" / "seed_sweep"

# Models that can be trained via train.py
TRAINABLE_MODELS = [
    "pi_lstm", "neural_ode",
    "cdae_pi_lstm", "cvae_pi_lstm", "beta_vae_pi_lstm",
    "pca_pi_lstm", "pls_pi_lstm",
    "delta_cdae_pi_lstm", "cdae_process_pi_lstm", "process_pi_lstm",
    "beta_vae_lp_pi_lstm", "beta_vae_lstm",
    "beta_vae_latent_lp_pi_lstm",
    "beta_vae_mamba", "beta_vae_liquid",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ckpt_path(out_dir: Path, model: str, seed: int) -> Path:
    return out_dir / "checkpoints" / f"{model}_s{seed}.pt"


def _log_path(out_dir: Path, seed: int) -> Path:
    return out_dir / f"seed_{seed}.log"


def _run_train(
    model: str,
    seed: int,
    out_dir: Path,
    epochs: Optional[int],
    device: str,
    regime_ablation: bool,
    extra_args: List[str],
) -> int:
    """Launch train.py for one seed. Returns exit code."""
    ckpt = _ckpt_path(out_dir, model, seed)
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(_ROOT / "train.py"),
        "--model", model,
        "--seed", str(seed),
        "--ckpt", str(ckpt),
        "--no-plots",
    ]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if device:
        cmd += ["--device", device]
    if regime_ablation:
        cmd += ["--regime-ablation"]
    cmd += extra_args

    log = _log_path(out_dir, seed)
    print(f"  [seed={seed}] Training → {ckpt.name}  (log: {log.name})")

    with open(log, "w") as f:
        result = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT,
            cwd=str(_ROOT),
        )
    return result.returncode


def _run_eval(
    model: str,
    seed: int,
    out_dir: Path,
    device: str,
    regime_ablation: bool,
) -> Optional[Dict]:
    """
    Run evaluate.py for one seed and parse the metrics from the checkpoint.
    Returns dict with metrics or None on failure.
    """
    import torch

    ckpt = _ckpt_path(out_dir, model, seed)
    if not ckpt.exists():
        print(f"  [seed={seed}] Checkpoint not found: {ckpt}")
        return None

    # Load metrics directly from checkpoint (saved by train.py)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    val_m  = state.get("val_metrics", {})
    test_m = state.get("test_metrics", {})
    fault_m = state.get("fault_metrics", {})

    def _extract(metrics_dict, target="pen"):
        m = metrics_dict.get(target, {})
        return {
            "RMSE": m.get("RMSE", float("nan")),
            "MAE":  m.get("MAE", float("nan")),
            "R2":   m.get("R2", float("nan")),
            "n":    m.get("n", 0),
        }

    result = {
        "seed": seed,
        "val_pen_rmse":   _extract(val_m)["RMSE"],
        "val_pen_mae":    _extract(val_m)["MAE"],
        "val_pen_r2":     _extract(val_m)["R2"],
        "test_pen_rmse":  _extract(test_m)["RMSE"],
        "test_pen_mae":   _extract(test_m)["MAE"],
        "test_pen_r2":    _extract(test_m)["R2"],
    }

    # Also include fault metrics if available (regime ablation)
    if fault_m:
        result["fault_pen_rmse"] = _extract(fault_m)["RMSE"]
        result["fault_pen_mae"]  = _extract(fault_m)["MAE"]
        result["fault_pen_r2"]   = _extract(fault_m)["R2"]

    return result


def _aggregate_results(results: List[Dict], out_dir: Path, model: str) -> None:
    """Write per-seed CSV + summary row with mean/std."""
    if not results:
        print("  No results to aggregate.")
        return

    csv_path = out_dir / "results.csv"
    metric_cols = [k for k in results[0].keys() if k != "seed"]

    # Per-seed rows
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed"] + metric_cols)
        writer.writeheader()
        for r in results:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                             for k, v in r.items()})

        # Summary row: mean
        summary_mean = {"seed": "mean"}
        summary_std  = {"seed": "std"}
        for col in metric_cols:
            vals = [r[col] for r in results if not np.isnan(r[col])]
            summary_mean[col] = f"{np.mean(vals):.6f}" if vals else "nan"
            summary_std[col]  = f"{np.std(vals):.6f}" if vals else "nan"
        writer.writerow(summary_mean)
        writer.writerow(summary_std)

    print(f"\n{'═' * 70}")
    print(f"  SEED SWEEP RESULTS — {model}")
    print(f"{'═' * 70}")
    print(f"  Seeds: {[r['seed'] for r in results]}")
    print(f"  n_runs: {len(results)}")
    print(f"{'─' * 70}")

    for col in metric_cols:
        vals = [r[col] for r in results if not np.isnan(r[col])]
        if vals:
            mean, std = np.mean(vals), np.std(vals)
            print(f"  {col:<22s}  {mean:.4f} +/- {std:.4f}  "
                  f"(min={min(vals):.4f}, max={max(vals):.4f})")

    print(f"{'─' * 70}")
    print(f"  CSV saved → {csv_path}")
    print(f"{'═' * 70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed training and evaluation sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", required=True, choices=TRAINABLE_MODELS,
        help="Model to train/evaluate.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Explicit list of seeds (e.g. --seeds 0 1 2 3 4).",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=DEFAULT_N_SEEDS,
        help=f"Number of seeds if --seeds not given (default: {DEFAULT_N_SEEDS}). "
             f"Uses seeds: {DEFAULT_SEEDS[:DEFAULT_N_SEEDS]}",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override n_epochs from config.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Force device (cpu/cuda).",
    )
    parser.add_argument(
        "--regime-ablation", action="store_true",
        help="Use Leave-One-Regime-Out split.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip seeds whose checkpoints already exist.",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, only aggregate existing checkpoints.",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory (default: outputs/seed_sweep/<model>/).",
    )
    args = parser.parse_args()

    # Resolve seeds
    if args.seeds:
        seeds = args.seeds
    else:
        # Generate deterministic seed list
        all_seeds = DEFAULT_SEEDS + list(range(2000, 2000 + 100))
        seeds = all_seeds[:args.n_seeds]

    # Output directory
    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 70}")
    print(f"  MULTI-SEED SWEEP")
    print(f"  Model  : {args.model}")
    print(f"  Seeds  : {seeds}")
    print(f"  Output : {out_dir}")
    print(f"{'═' * 70}\n")

    # ── Train ────────────────────────────────────────────────────────────────
    if not args.eval_only:
        for seed in seeds:
            ckpt = _ckpt_path(out_dir, args.model, seed)
            if args.skip_existing and ckpt.exists():
                print(f"  [seed={seed}] Checkpoint exists, skipping.")
                continue

            rc = _run_train(
                model=args.model,
                seed=seed,
                out_dir=out_dir,
                epochs=args.epochs,
                device=args.device or "",
                regime_ablation=args.regime_ablation,
                extra_args=[],
            )
            if rc != 0:
                print(f"  [seed={seed}] FAILED (exit code {rc}). "
                      f"Check {_log_path(out_dir, seed)}")

    # ── Evaluate + aggregate ─────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  Collecting results ...")

    results = []
    for seed in seeds:
        r = _run_eval(
            model=args.model,
            seed=seed,
            out_dir=out_dir,
            device=args.device or "cpu",
            regime_ablation=args.regime_ablation,
        )
        if r is not None:
            results.append(r)
            print(f"  [seed={seed}] test_pen_rmse = {r['test_pen_rmse']:.4f} g/L")

    _aggregate_results(results, out_dir, args.model)


if __name__ == "__main__":
    main()
