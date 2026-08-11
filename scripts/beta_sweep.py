#!/usr/bin/env python
"""
beta_sweep.py
=============
Train one β-VAE per β value in the sweep grid and collect val reconstruction
losses, so the effect of β on encoder quality can be compared before
full PI-LSTM retraining.

β grid : [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]

Each run uses a FIXED β (no annealing ramp) achieved by setting
anneal_epochs=1 and delay_epochs=0 so LinearBetaSchedule returns
max_beta from epoch 1 onwards.  All other hyper-parameters match the
production β-VAE training (n_epochs=300, latent_dim=64, seed=42, …).

Outputs
-------
  checkpoints/beta_sweep/b<β>/best.pt          — encoder checkpoint
  checkpoints/beta_sweep/b<β>/scaler.joblib    — StandardScaler
  outputs/beta_sweep/summary.csv               — β | val_recon_loss | epoch
  outputs/beta_sweep/summary.png               — val_recon vs β log-plot

Usage
-----
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/beta_sweep.py
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/beta_sweep.py --betas 1.0 0.1 0.01
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/beta_sweep.py --epochs 200 --amp
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

import torch

# ── Project root on sys.path ─────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import the training function directly — no subprocess overhead
from scripts.train_autoencoder import train_beta_vae   # noqa: E402

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_BETAS   = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
CSV_PATH        = "./100_Batches_IndPenSim_V3.csv"
CKPT_ROOT       = Path("./checkpoints/beta_sweep")
OUTPUT_DIR      = Path("./outputs/beta_sweep")
N_EPOCHS        = 300
LATENT_DIM      = 64
SEED            = 42


def _beta_tag(beta: float) -> str:
    """Return a filesystem-safe tag for a β value, e.g. 0.05 → 'b0.050'."""
    return f"b{beta:.3f}"


def run_sweep(
    betas:       List[float],
    csv_path:    str,
    n_epochs:    int,
    latent_dim:  int,
    seed:        int,
    device:      str,
    use_amp:     bool,
) -> None:
    """
    Train one β-VAE per entry in *betas* sequentially and write a summary.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)

    summary_csv  = OUTPUT_DIR / "summary.csv"
    results: List[dict] = []

    print(f"\n{'='*65}")
    print(f"  β-VAE sweep — {len(betas)} runs, {n_epochs} epochs each")
    print(f"  β values : {betas}")
    print(f"  device   : {device}  |  AMP={use_amp}")
    print(f"{'='*65}\n")

    for β in betas:
        tag       = _beta_tag(β)
        run_dir   = CKPT_ROOT / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_out  = str(run_dir / "best.pt")
        scaler_out = str(run_dir / "scaler.joblib")

        print(f"\n{'─'*65}")
        print(f"  β = {β}   tag={tag}   ckpt → {ckpt_out}")
        print(f"{'─'*65}")

        train_beta_vae(
            csv_path        = csv_path,
            ckpt_out        = ckpt_out,
            scaler_out      = scaler_out,
            latent_dim      = latent_dim,
            max_beta        = β,
            anneal_schedule = "linear",
            anneal_epochs   = 1,        # β = max_beta from epoch 1 → fixed β
            anneal_delay    = 0,
            n_cycles        = 1,        # unused (linear schedule)
            n_epochs        = n_epochs,
            seed            = seed,
            device          = device,
            use_amp         = use_amp,
        )

        # Read the saved checkpoint to extract best val recon loss
        ckpt = torch.load(ckpt_out, map_location="cpu", weights_only=False)
        val_recon = float(ckpt.get("val_loss", float("nan")))
        epoch_saved = int(ckpt.get("epoch", -1))

        results.append({
            "beta":           β,
            "val_recon_loss": val_recon,
            "epoch_saved":    epoch_saved,
            "ckpt":           ckpt_out,
            "scaler":         scaler_out,
        })
        print(f"\n  → β={β}  val_recon={val_recon:.6f}  epoch={epoch_saved}")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["beta", "val_recon_loss", "epoch_saved", "ckpt", "scaler"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Summary CSV saved → {summary_csv}")

    # ── Summary table (stdout) ────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  {'β':>10}  {'val_recon':>14}  {'epoch':>7}")
    print(f"{'─'*55}")
    for r in results:
        print(f"  {r['beta']:>10.4f}  {r['val_recon_loss']:>14.6f}  {r['epoch_saved']:>7d}")
    print(f"{'='*55}")

    # ── Summary plot ──────────────────────────────────────────────────────────
    _plot_summary(results)


def _plot_summary(results: List[dict]) -> None:
    """Log-scale plot of val_recon_loss vs β."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        betas  = [r["beta"]           for r in results]
        losses = [r["val_recon_loss"] for r in results]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(betas, losses, "o-", color="#2176ae", linewidth=1.8, markersize=7)
        for β, loss in zip(betas, losses):
            ax.annotate(f"{loss:.4f}", xy=(β, loss),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=7.5)
        ax.set_xscale("log")
        ax.set_xlabel("β  (log scale)", fontsize=11)
        ax.set_ylabel("Best val reconstruction loss", fontsize=11)
        ax.set_title("β-VAE: validation reconstruction loss vs β", fontsize=12)
        ax.invert_xaxis()   # low β on right, high β on left
        ax.grid(True, which="both", alpha=0.3, linewidth=0.6)
        fig.tight_layout()

        for ext in ("png", "pdf"):
            out = OUTPUT_DIR / f"summary.{ext}"
            fig.savefig(out, dpi=150)
            print(f"  Plot saved → {out}")
        plt.close(fig)

    except Exception as exc:
        print(f"  [plot] Skipped (error: {exc})")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="β-VAE β sweep — trains one model per β value"
    )
    parser.add_argument(
        "--betas", nargs="+", type=float, default=DEFAULT_BETAS,
        metavar="β",
        help="List of β values to sweep (default: 1.0 0.5 0.1 0.05 0.01 0.005 0.001)",
    )
    parser.add_argument("--csv",        default=CSV_PATH,  help="Path to IndPenSim V3 CSV")
    parser.add_argument("--epochs",     type=int,   default=N_EPOCHS)
    parser.add_argument("--latent-dim", type=int,   default=LATENT_DIM)
    parser.add_argument("--seed",       type=int,   default=SEED)
    parser.add_argument("--device",     default="cuda", help="cuda | cpu")
    parser.add_argument("--amp",        action="store_true",
                        help="Enable automatic mixed precision (CUDA only)")
    args = parser.parse_args()

    run_sweep(
        betas      = args.betas,
        csv_path   = args.csv,
        n_epochs   = args.epochs,
        latent_dim = args.latent_dim,
        seed       = args.seed,
        device     = args.device,
        use_amp    = args.amp,
    )


if __name__ == "__main__":
    main()
