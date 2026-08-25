#!/usr/bin/env python
"""
train.py
========
CLI entry point for training any benchmark temporal model.

Available models
----------------
    pi_lstm               PI-LSTM, 23 process features only (2-head)
    neural_ode            Neural ODE, 23 process features only
    cdae_pi_lstm          PI-LSTM, 64 CDAE Raman latents (pen-only)
    cvae_pi_lstm          PI-LSTM, 64 CVAE Raman latents (pen-only)
    pca_pi_lstm           PI-LSTM,  4 PCA  Raman latents (pen-only)
    pls_pi_lstm           PI-LSTM,  4 PLS  Raman latents (pen-only)
    cdae_process_pi_lstm  PI-LSTM, 64 CDAE latents + 23 process vars (pen-only)
    delta_cdae_pi_lstm    PI-LSTM, 64 CDAE latents, ΔP head (pen-only, ablation)
    process_pi_lstm       PI-LSTM, 23 process features only (pen-only, no Raman)

Usage
-----
    python train.py --model pi_lstm
    python train.py --model cdae_pi_lstm

    # Override epochs and checkpoint path at the command line:
    python train.py --model pi_lstm --epochs 2 --ckpt /tmp/test.pt

    # Force CPU even when a GPU is available:
    python train.py --model neural_ode --device cpu
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Regime-ablation split helpers ─────────────────────────────────────────────
# Batch IDs are 0-indexed in _batch_id (batch 1 == id 0, batch 30 == id 29).

_REGIME_1_IDS = set(range(0,  30))   # batches  1-30  recipe-driven
_REGIME_2_IDS = set(range(30, 60))   # batches 31-60  operator-controlled
_REGIME_3_IDS = set(range(60, 90))   # batches 61-90  APC closed-loop


def _regime_ablation_splits(dm, seed: int = 42):
    """
    Leave-One-Regime-Out split (Regime 2 held out as test):

      Train / Val  →  Regime 1 (batches 1-30, recipe-driven), 80/20 random
      Test         →  Regime 2 (batches 31-60, operator-controlled)

    Faulty batches (91-100) are excluded; they remain available via
    dm.batches_fault if needed separately.
    """
    regime1 = [
        b for b in dm.batches_clean
        if b["_batch_id"].iloc[0] in _REGIME_1_IDS
    ]
    regime2 = [
        b for b in dm.batches_clean
        if b["_batch_id"].iloc[0] in _REGIME_2_IDS
    ]

    rng     = np.random.default_rng(seed)
    perm    = rng.permutation(len(regime1))
    n_train = int(len(regime1) * 0.80)
    train_b = [regime1[i] for i in perm[:n_train]]
    val_b   = [regime1[i] for i in perm[n_train:]]
    test_b  = regime2

    print(
        f"  [regime-ablation]  Train: {len(train_b)} (Regime 1 80%)  |  "
        f"Val: {len(val_b)} (Regime 1 20%)  |  "
        f"Test: {len(test_b)} (Regime 2, operator-controlled)"
    )
    return train_b, val_b, test_b


MODEL_NAMES = [
    "pi_lstm", "neural_ode",
    "cdae_pi_lstm", "cvae_pi_lstm", "beta_vae_pi_lstm",
    "pca_pi_lstm", "pls_pi_lstm",
    "delta_cdae_pi_lstm", "cdae_process_pi_lstm", "process_pi_lstm",
    "beta_vae_lp_pi_lstm",
    "beta_vae_lstm",
    "beta_vae_latent_lp_pi_lstm",
    "beta_vae_mamba",
    "beta_vae_liquid",
]


def _resolve_config_path(model_name: str) -> Path:
    """Return path to configs/<model_name>.yaml, raising if not found."""
    root = Path(__file__).parent
    p    = root / "configs" / f"{model_name}.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"Config not found: {p}\n"
            f"Expected one of: {', '.join(MODEL_NAMES)}"
        )
    return p


def _maybe_build_raman_encoder(cfg, device: torch.device):
    """
    Return the appropriate frozen Raman encoder, or ``None`` if
    ``use_raman=False`` in the config.

    Supported ``raman_encoder_type`` values
    ----------------------------------------
        cdae_v2  -> src.data.cdae_encoder.CDAERamanEncoderV2  (64-d)
        cvae_v2  -> src.data.cvae_encoder.CVAERamanEncoderV2  (64-d, posterior mean)
        beta_vae -> src.data.beta_vae_encoder.BetaVAERamanEncoderV2 (64-d, posterior mean)
        pca_v2   -> src.data.pca_encoder.PCARamanEncoderV1    (K-d, fit by pca_baseline.py)
        pls_v2   -> src.data.pls_encoder.PLSRamanEncoderV1    (K-d, fit by pls_baseline.py)

    All four share the SAME preprocessing pipeline:
        Savitzky-Golay first derivative (window=15, polyorder=2)
        -> StandardScaler.transform()
        -> frozen encoder
    """
    use_raman = cfg.data.get("use_raman", False)
    if not use_raman:
        return None

    enc_type    = cfg.data.get("raman_encoder_type", None)
    ckpt_path   = cfg.data.get("raman_ckpt",   None)
    scaler_path = cfg.data.get("raman_scaler", None)

    if enc_type == "cdae_v2":
        ckpt_path   = ckpt_path   or "./checkpoints/cdae_best.pt"
        scaler_path = scaler_path or "./checkpoints/cdae_scaler.joblib"
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"CDAE-V2 checkpoint not found: {ckpt_path}")
        if not Path(scaler_path).exists():
            raise FileNotFoundError(f"CDAE-V2 scaler not found: {scaler_path}")
        from src.data.cdae_encoder import CDAERamanEncoderV2
        print(f"  Loading CDAERamanEncoderV2 from {ckpt_path}  (scaler: {scaler_path}) ...")
        return CDAERamanEncoderV2(ckpt_path, scaler_path, device)

    if enc_type == "cvae_v2":
        ckpt_path   = ckpt_path   or "./checkpoints/cvae_best.pt"
        scaler_path = scaler_path or "./checkpoints/cvae_scaler.joblib"
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"CVAE-V2 checkpoint not found: {ckpt_path}")
        if not Path(scaler_path).exists():
            raise FileNotFoundError(f"CVAE-V2 scaler not found: {scaler_path}")
        from src.data.cvae_encoder import CVAERamanEncoderV2
        print(f"  Loading CVAERamanEncoderV2 from {ckpt_path}  (scaler: {scaler_path}) ...")
        return CVAERamanEncoderV2(ckpt_path, scaler_path, device)

    if enc_type == "beta_vae":
        ckpt_path   = ckpt_path   or "./checkpoints/beta_vae_best.pt"
        scaler_path = scaler_path or "./checkpoints/beta_vae_scaler.joblib"
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"β-VAE checkpoint not found: {ckpt_path}")
        if not Path(scaler_path).exists():
            raise FileNotFoundError(f"β-VAE scaler not found: {scaler_path}")
        from src.data.beta_vae_encoder import BetaVAERamanEncoderV2
        print(f"  Loading BetaVAERamanEncoderV2 from {ckpt_path}  (scaler: {scaler_path}) ...")
        return BetaVAERamanEncoderV2(ckpt_path, scaler_path, device)

    if enc_type == "pca_v2":
        ckpt_path   = ckpt_path   or "./checkpoints/pca_best.joblib"
        scaler_path = scaler_path or "./checkpoints/pca_scaler.joblib"
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"PCA encoder checkpoint not found: {ckpt_path}\n"
                "Run `python scripts/pca_baseline.py` first to fit and save the PCA encoder."
            )
        if not Path(scaler_path).exists():
            raise FileNotFoundError(
                f"PCA scaler not found: {scaler_path}\n"
                "Run `python scripts/pca_baseline.py` first to fit and save the PCA scaler."
            )
        n_components = int(cfg.data.get("raman_latent_dim", 4))
        from src.data.pca_encoder import PCARamanEncoderV1
        print(
            f"  Loading PCARamanEncoderV1 from {ckpt_path}  "
            f"(scaler: {scaler_path}, n_components={n_components}) ..."
        )
        return PCARamanEncoderV1(
            pca_model_path=ckpt_path,
            scaler_path=scaler_path,
            n_components=n_components,
        )

    if enc_type == "pls_v2":
        ckpt_path   = ckpt_path   or "./checkpoints/pls_best.joblib"
        scaler_path = scaler_path or "./checkpoints/pls_scaler.joblib"
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"PLS encoder checkpoint not found: {ckpt_path}\n"
                "Run `python scripts/pls_baseline.py` first to fit and save the PLS encoder."
            )
        if not Path(scaler_path).exists():
            raise FileNotFoundError(
                f"PLS scaler not found: {scaler_path}\n"
                "Run `python scripts/pls_baseline.py` first to fit and save the PLS scaler."
            )
        n_components = int(cfg.data.get("raman_latent_dim", 4))
        from src.data.pls_encoder import PLSRamanEncoderV1
        print(
            f"  Loading PLSRamanEncoderV1 from {ckpt_path}  "
            f"(scaler: {scaler_path}, n_components={n_components}) ..."
        )
        return PLSRamanEncoderV1(
            pls_model_path=ckpt_path,
            scaler_path=scaler_path,
            n_components=n_components,
        )

    raise ValueError(
        f"Unknown raman_encoder_type '{enc_type}'. "
        f"Choose one of: 'cdae_v2', 'cvae_v2', 'pca_v2', 'pls_v2'."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a PI-LSTM or Neural ODE benchmark model."
    )
    parser.add_argument(
        "--model", required=True,
        choices=MODEL_NAMES,
        help="Which model variant to train.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to a YAML config (overrides default configs/<model>.yaml).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override n_epochs from config.",
    )
    parser.add_argument(
        "--ckpt", default=None,
        help="Override output checkpoint path from config.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Force device ('cpu' or 'cuda'). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip saving prediction / history plots.",
    )
    parser.add_argument(
        "--regime-ablation", action="store_true", dest="regime_ablation",
        help=(
            "Leave-One-Regime-Out split: train/val on batches 1-30 (recipe), "
            "test on batches 31-60 (operator). "
            "Overrides the train_frac / split_strategy from config."
        ),
    )
    parser.add_argument(
        "--no-early-stopping", action="store_true", dest="no_early_stopping",
        help=(
            "Disable early stopping and run for the full n_epochs. "
            "Overrides es_patience from the config."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help=(
            "Override the random seed (default: from config, usually 42). "
            "Also appends '_s{seed}' to the checkpoint filename for "
            "multi-seed experiments."
        ),
    )
    args = parser.parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 65}")
    print(f"  Model  : {args.model}")
    print(f"  Device : {device}")
    print(f"  Seed   : {args.seed if args.seed is not None else 'config default'}")
    print(f"{'=' * 65}\n")

    # ── Config ────────────────────────────────────────────────────────────────
    from src.config import Config
    cfg_path = Path(args.config) if args.config else _resolve_config_path(args.model)
    cfg = Config.from_yaml(cfg_path)

    # Command-line overrides
    if args.epochs is not None:
        cfg.training.n_epochs = args.epochs
    if args.ckpt is not None:
        cfg.output.checkpoint = args.ckpt

    # Seed override — also patch checkpoint/plots paths to avoid collisions
    if args.seed is not None:
        cfg.data["seed"] = args.seed
        suffix = f"_s{args.seed}"
        # Auto-rename checkpoint: foo.pt → foo_s{seed}.pt
        if args.ckpt is None:
            ckpt_base = str(cfg.output.checkpoint)
            if ckpt_base.endswith(".pt"):
                cfg.output["checkpoint"] = ckpt_base[:-3] + suffix + ".pt"
            else:
                cfg.output["checkpoint"] = ckpt_base + suffix
        # Auto-rename plots dir
        plots_base = str(cfg.output.plots_dir).rstrip("/")
        cfg.output["plots_dir"] = plots_base + suffix

    seed = int(cfg.data.get("seed", 42))
    _set_seed(seed)

    # ── Raman encoder (if needed) ─────────────────────────────────────────────
    raman_encoder = _maybe_build_raman_encoder(cfg, device)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("[1] Loading data ...")
    from src.data.dataset import PenicillinDataModule
    dm = PenicillinDataModule(cfg.data, raman_encoder=raman_encoder)
    dm.load()

    if args.regime_ablation:
        train_b, val_b, test_b = _regime_ablation_splits(dm, seed=seed)
    else:
        strategy   = cfg.data.get("split_strategy", "random")
        train_frac = float(cfg.data.get("train_frac", 0.80))
        train_b, val_b, test_b = dm.get_splits(
            train_frac=train_frac,
            seed=seed,
            strategy=strategy,
        )
        print(
            f"  Train: {len(train_b)}  |  Val: {len(val_b)}  |  "
            f"Test (faulty): {len(test_b)}"
        )

    # ── Algorithm ────────────────────────────────────────────────────────────
    print(f"\n[2] Initialising algorithm '{args.model}' ...")
    from src.algorithms import get_algorithm
    AlgoCls = get_algorithm(args.model)
    alg     = AlgoCls(cfg, device)

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n[3] Training ...")
    if args.no_early_stopping:
        cfg.training.es_patience = int(cfg.training.n_epochs) + 1
        print(f"  [no-early-stopping] es_patience set to {cfg.training.es_patience}")
    history = alg.fit(train_b, val_b, verbose=True)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\n[4] Evaluating ...")
    from src.evaluation.metrics import print_metrics_table

    val_label  = "Val (Regime 1 holdout)" if args.regime_ablation else "Val (fault-free)"
    test_label = "Test (Regime 2)"        if args.regime_ablation else "Test (faulty)"

    print(f"  --- {val_label} ---")
    val_metrics = alg.evaluate(val_b)
    print_metrics_table(val_metrics, label="Val")

    print(f"  --- {test_label} ---")
    test_metrics = alg.evaluate(test_b)
    print_metrics_table(test_metrics, label="R2" if args.regime_ablation else "Test")

    # ── Extra regime evaluation (only when --regime-ablation) ─────────────────
    r3_metrics    = None
    fault_metrics = None
    if args.regime_ablation:
        regime3_b = [
            b for b in dm.batches_clean
            if b["_batch_id"].iloc[0] in _REGIME_3_IDS
        ]
        fault_b = dm.batches_fault

        print("  --- Test (Regime 3, APC closed-loop) ---")
        r3_metrics = alg.evaluate(regime3_b)
        print_metrics_table(r3_metrics, label="R3")

        print("  --- Test (Faulty batches 91-100) ---")
        fault_metrics = alg.evaluate(fault_b)
        print_metrics_table(fault_metrics, label="Fault")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    print("\n[5] Saving checkpoint ...")
    ckpt_path = str(cfg.output.checkpoint)
    os.makedirs(Path(ckpt_path).parent, exist_ok=True)
    alg.save(
        ckpt_path,
        extra={
            "val_metrics":   val_metrics,
            "test_metrics":  test_metrics,
            "r3_metrics":    r3_metrics,
            "fault_metrics": fault_metrics,
        },
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("\n[6] Saving plots ...")
        plots_dir = Path(str(cfg.output.plots_dir))
        os.makedirs(plots_dir, exist_ok=True)

        from src.visualization.plots import (
            plot_training_history,
            plot_predictions_grid,
        )

        plot_training_history(
            history,
            save_path=str(plots_dir / "training_history.png"),
            title=f"{args.model} — training history",
        )

        eval_splits = [("val", val_b, val_metrics), ("r2", test_b, test_metrics)]
        if args.regime_ablation:
            eval_splits += [
                ("r3",    regime3_b, r3_metrics),
                ("fault", fault_b,   fault_metrics),
            ]

        for split_name, batches, res in eval_splits:
            for tgt in ["bio", "pen"]:
                tgt_label = "Biomass (g/L)" if tgt == "bio" else "Penicillin (g/L)"
                m = res.get(tgt, {})
                plot_predictions_grid(
                    batches=batches,
                    y_preds=res[f"y_preds_{tgt}"],
                    y_sparses=res[f"y_sparses_{tgt}"],
                    target_label=tgt_label,
                    title=(
                        f"{args.model}  [{split_name}]  {tgt_label}  "
                        f"RMSE={m.get('RMSE', float('nan')):.3f}  "
                        f"R²={m.get('R2', float('nan')):.4f}"
                    ),
                    save_path=str(plots_dir / f"{split_name}_{tgt}.png"),
                )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  {args.model} training complete.")
    print_metrics_table(val_metrics,  label="Val ")
    print_metrics_table(test_metrics, label="R2  " if args.regime_ablation else "Test")
    if args.regime_ablation:
        print_metrics_table(r3_metrics,    label="R3  ")
        print_metrics_table(fault_metrics, label="Fault")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
