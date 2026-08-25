#!/usr/bin/env python
"""
evaluate.py
===========
CLI entry point for evaluating trained model checkpoints.

Usage
-----
    # Evaluate a single model
    python evaluate.py --model pi_lstm

    # Evaluate every model with a checkpoint on disk
    python evaluate.py --model all

    # Point to a custom checkpoint
    python evaluate.py --model neural_ode --ckpt ./checkpoints/neural_ode.pt

    # Also save per-batch prediction grids and a scatter comparison plot
    python evaluate.py --model all --plots --scatter ./outputs/scatter.png
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, Optional

import torch

warnings.filterwarnings("ignore")


# ── Regime-ablation split helpers ─────────────────────────────────────────────
# Batch IDs are 0-indexed in _batch_id (batch 1 == id 0, batch 30 == id 29).

_REGIME_1_IDS = set(range(0,  30))   # batches  1-30  recipe-driven
_REGIME_2_IDS = set(range(30, 60))   # batches 31-60  operator-controlled
_REGIME_3_IDS = set(range(60, 90))   # batches 61-90  APC closed-loop


def _regime_ablation_splits(dm, seed: int = 42, test_regime: int = 2):
    """
    Leave-One-Regime-Out split.

    Train / Val  →  Regime 1 (batches 1-30, recipe-driven), 80/20 random
    Test         →  ``test_regime``
                      2 → Regime 2 (batches 31-60, operator-controlled)
                      3 → Regime 3 (batches 61-90, APC closed-loop)

    Faulty batches (91-100) are excluded; they remain available via
    dm.batches_fault if needed separately.
    """
    import numpy as np

    _TEST_IDS = {
        2: _REGIME_2_IDS,
        3: _REGIME_3_IDS,
    }
    _TEST_LABELS = {
        2: "Regime 2, operator-controlled",
        3: "Regime 3, APC closed-loop",
    }
    if test_regime not in _TEST_IDS:
        raise ValueError(f"test_regime must be 2 or 3, got {test_regime}")

    regime1 = [
        b for b in dm.batches_clean
        if b["_batch_id"].iloc[0] in _REGIME_1_IDS
    ]
    test_b = [
        b for b in dm.batches_clean
        if b["_batch_id"].iloc[0] in _TEST_IDS[test_regime]
    ]

    rng     = np.random.default_rng(seed)
    perm    = rng.permutation(len(regime1))
    n_train = int(len(regime1) * 0.80)
    train_b = [regime1[i] for i in perm[:n_train]]
    val_b   = [regime1[i] for i in perm[n_train:]]

    print(
        f"  [regime-ablation]  Train: {len(train_b)} (Regime 1 80%)  |  "
        f"Val: {len(val_b)} (Regime 1 20%)  |  "
        f"Test: {len(test_b)} ({_TEST_LABELS[test_regime]})"
    )
    return train_b, val_b, test_b


ALL_MODELS = [
    "pi_lstm", "neural_ode",
    "cdae_pi_lstm", "cvae_pi_lstm", "beta_vae_pi_lstm",
    "pca_pi_lstm", "pls_pi_lstm",
    "delta_cdae_pi_lstm", "cdae_process_pi_lstm", "process_pi_lstm",
]

DEFAULT_CKPTS = {
    "pi_lstm":                "./checkpoints/pi_lstm.pt",
    "neural_ode":             "./checkpoints/neural_ode.pt",
    "cdae_pi_lstm":           "./checkpoints/cdae_pi_lstm.pt",
    "cvae_pi_lstm":           "./checkpoints/cvae_pi_lstm.pt",
    "beta_vae_pi_lstm":       "./checkpoints/beta_vae_pi_lstm.pt",
    "pca_pi_lstm":            "./checkpoints/pca_pi_lstm.pt",
    "pls_pi_lstm":            "./checkpoints/pls_pi_lstm.pt",
    "delta_cdae_pi_lstm":     "./checkpoints/delta_cdae_pi_lstm.pt",
    "cdae_process_pi_lstm":   "./checkpoints/cdae_process_pi_lstm.pt",
    "process_pi_lstm":        "./checkpoints/process_pi_lstm.pt",
}


# ── Frozen Raman encoder factory (same registry as train.py) ──────────────────

def _maybe_build_raman_encoder(cfg, device: torch.device):
    """
    Return the appropriate frozen Raman encoder, or ``None`` if
    ``use_raman=False`` in the config.

    Supported ``raman_encoder_type`` values
    ----------------------------------------
        cdae_v2  -> CDAERamanEncoderV2  (64-d)
        cvae_v2  -> CVAERamanEncoderV2  (64-d, posterior mean)
        beta_vae -> BetaVAERamanEncoderV2 (64-d, posterior mean)
        pca_v2   -> PCARamanEncoderV1   (K-d, fit by pca_baseline.py)
        pls_v2   -> PLSRamanEncoderV1   (K-d, fit by pls_baseline.py)
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
            raise FileNotFoundError(f"PCA scaler not found: {scaler_path}")
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
            raise FileNotFoundError(f"PLS scaler not found: {scaler_path}")
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


# ── Data loading helper ───────────────────────────────────────────────────────

_DEFAULT_CSV = "./100_Batches_IndPenSim_V3.csv"


def _patch_portable_paths(cfg) -> None:
    """
    Replace any embedded checkpoint-time absolute path that does not exist on
    the current filesystem with its project-root default.  This makes
    checkpoints portable between machines (e.g. Windows -> Linux).
    """
    csv_path = cfg.data.get("csv_path", _DEFAULT_CSV)
    if not Path(csv_path).exists() and Path(_DEFAULT_CSV).exists():
        print(f"  [portable] csv_path not found: {csv_path!r}\n"
              f"  [portable] falling back to {_DEFAULT_CSV!r}")
        cfg.data["csv_path"] = _DEFAULT_CSV


def _load_data(cfg, device: torch.device, regime_ablation: bool = False,
               test_regime: int = 2):
    """Return ``(train_b, val_b, test_b)`` using the split stored in cfg."""
    import random
    import numpy as np
    seed = int(cfg.data.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    _patch_portable_paths(cfg)
    raman_enc = _maybe_build_raman_encoder(cfg, device)

    from src.data.dataset import PenicillinDataModule
    dm = PenicillinDataModule(cfg.data, raman_encoder=raman_enc)
    dm.load()

    if regime_ablation:
        return _regime_ablation_splits(dm, seed=seed, test_regime=test_regime)

    strategy   = cfg.data.get("split_strategy", "random")
    train_frac = float(cfg.data.get("train_frac", 0.80))
    return dm.get_splits(train_frac=train_frac, seed=seed, strategy=strategy)


def _evaluate_one(
    model_name: str,
    ckpt_path:  str,
    device:     torch.device,
    save_plots: bool = False,
    regime_ablation: bool = False,
    test_regime: int = 2,
    seed_override: Optional[int] = None,
) -> Optional[Dict]:
    """Load checkpoint, reload data, run ``evaluate()`` on val + test splits."""
    if not Path(ckpt_path).exists():
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return None

    print(f"\n{'─' * 60}")
    print(f"  Model : {model_name}")
    print(f"  Ckpt  : {ckpt_path}")
    if seed_override is not None:
        print(f"  Seed  : {seed_override} (override)")

    from src.algorithms import BaseAlgorithm

    alg = BaseAlgorithm.load(ckpt_path, device)
    cfg = alg.cfg

    # Apply seed override to the loaded config
    if seed_override is not None:
        cfg.data["seed"] = seed_override

    _, val_b, test_b = _load_data(cfg, device, regime_ablation=regime_ablation,
                                   test_regime=test_regime)

    from src.evaluation.metrics import print_metrics_table

    _test_regime_labels = {
        2: "Test (Regime 2, operator-controlled):",
        3: "Test (Regime 3, APC closed-loop):",
    }
    val_label  = "Val (Regime 1 holdout):" if regime_ablation else "Val (fault-free):"
    test_label = _test_regime_labels.get(test_regime, f"Test (Regime {test_regime}):") \
                 if regime_ablation else "Test (faulty):"

    print(f"  {val_label}")
    val_m = alg.evaluate(val_b)
    print_metrics_table(val_m, label="Val")

    print(f"  {test_label}")
    test_m = alg.evaluate(test_b)
    print_metrics_table(test_m, label="Test")

    if save_plots:
        from src.visualization.plots import plot_predictions_grid
        plots_dir = Path(str(cfg.output.plots_dir))
        plots_dir.mkdir(parents=True, exist_ok=True)

        for split_name, batches, res in [
            ("val",  val_b,  val_m),
            ("test", test_b, test_m),
        ]:
            for tgt in ["bio", "pen"]:
                tgt_label = "Biomass (g/L)" if tgt == "bio" else "Penicillin (g/L)"
                m = res.get(tgt, {})
                plot_predictions_grid(
                    batches=batches,
                    y_preds=res[f"y_preds_{tgt}"],
                    y_sparses=res[f"y_sparses_{tgt}"],
                    target_label=tgt_label,
                    title=(
                        f"{model_name}  [{split_name}]  {tgt_label}  "
                        f"RMSE={m.get('RMSE', float('nan')):.3f}  "
                        f"R²={m.get('R2', float('nan')):.4f}"
                    ),
                    save_path=str(plots_dir / f"eval_{split_name}_{tgt}.png"),
                )

    return {"val": val_m, "test": test_m}


def _print_comparison_table(all_results: Dict[str, Dict]) -> None:
    """Print a side-by-side comparison table for all evaluated models."""
    print(f"\n{'=' * 80}")
    print(f"  {'Model':<18}  {'Split':>5}  "
          f"{'Bio RMSE':>9}  {'Bio R²':>8}  "
          f"{'Pen RMSE':>9}  {'Pen R²':>8}")
    print(f"{'─' * 80}")
    for model_name, res in all_results.items():
        if res is None:
            continue
        for split_key in ("val", "test"):
            m   = res.get(split_key, {})
            bio = m.get("bio", {})
            pen = m.get("pen", {})
            print(
                f"  {model_name:<18}  {split_key:>5}  "
                f"{bio.get('RMSE', float('nan')):>9.3f}  "
                f"{bio.get('R2',   float('nan')):>8.4f}  "
                f"{pen.get('RMSE', float('nan')):>9.3f}  "
                f"{pen.get('R2',   float('nan')):>8.4f}"
            )
    print(f"{'=' * 80}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained PI-LSTM / Neural ODE benchmark checkpoints."
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=ALL_MODELS + ["all"],
        help="Which model to evaluate. 'all' iterates over every checkpoint on disk.",
    )
    parser.add_argument(
        "--ckpt", default=None,
        help="Override checkpoint path (only valid when --model is a single model).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Force device ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--plots", action="store_true",
        help="Save per-batch prediction grid plots.",
    )
    parser.add_argument(
        "--scatter", default=None,
        help="Path to save a scatter comparison PNG (all models).",
    )
    parser.add_argument(
        "--regime-ablation", action="store_true", dest="regime_ablation",
        help=(
            "Leave-One-Regime-Out evaluation: val on Regime 1 holdout "
            "(batches 1-30), test set chosen by --test-regime. "
            "Overrides the default fault-free val / faulty test split."
        ),
    )
    parser.add_argument(
        "--test-regime", type=int, default=2, dest="test_regime",
        choices=[2, 3],
        help=(
            "Which regime to use as the test set when --regime-ablation is active. "
            "2 = Regime 2 (batches 31-60, operator-controlled, default). "
            "3 = Regime 3 (batches 61-90, APC closed-loop)."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help=(
            "Override random seed for data splitting (default: from checkpoint config). "
            "When set, also auto-resolves checkpoint path to <model>_s{seed}.pt."
        ),
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\nDevice: {device}")

    models_to_run = ALL_MODELS if args.model == "all" else [args.model]

    all_results: Dict[str, Optional[Dict]] = {}
    scatter_inputs: Dict[str, Dict] = {}

    for model_name in models_to_run:
        if args.ckpt and len(models_to_run) == 1:
            ckpt_path = args.ckpt
        else:
            base_ckpt = DEFAULT_CKPTS[model_name]
            # Auto-resolve seed-suffixed checkpoint
            if args.seed is not None and base_ckpt.endswith(".pt"):
                seeded = base_ckpt[:-3] + f"_s{args.seed}.pt"
                if Path(seeded).exists():
                    ckpt_path = seeded
                else:
                    ckpt_path = base_ckpt
            else:
                ckpt_path = base_ckpt

        res = _evaluate_one(
            model_name, ckpt_path, device,
            save_plots=args.plots,
            regime_ablation=args.regime_ablation,
            test_regime=args.test_regime,
            seed_override=args.seed,
        )
        all_results[model_name] = res
        if res is not None and args.scatter:
            scatter_inputs[model_name] = {
                "y_preds_bio":   (res["val"]["y_preds_bio"]   + res["test"]["y_preds_bio"]),
                "y_preds_pen":   (res["val"]["y_preds_pen"]   + res["test"]["y_preds_pen"]),
                "y_sparses_bio": (res["val"]["y_sparses_bio"] + res["test"]["y_sparses_bio"]),
                "y_sparses_pen": (res["val"]["y_sparses_pen"] + res["test"]["y_sparses_pen"]),
            }

    if len(models_to_run) > 1:
        _print_comparison_table(all_results)

    if args.scatter and scatter_inputs:
        from src.visualization.plots import plot_scatter_comparison
        Path(args.scatter).parent.mkdir(parents=True, exist_ok=True)
        plot_scatter_comparison(scatter_inputs, args.scatter)


if __name__ == "__main__":
    main()
