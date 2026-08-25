#!/usr/bin/env python
"""
multi_seed_train.py
====================
Train β-VAE + PI-LSTM with 4 seeds under two protocols:
  Protocol 1: Train on R1–R3 (batches 0–89)
  Protocol 2: Train on R1 only (batches 0–29)
Test on R4 (fault, batches 90–99).

Seeds: 1, 2, 3, 4  (seed=42 already exists from prior runs)

Usage
-----
    PYTHONPATH=. python scripts/multi_seed_train.py --protocol 1 --device cuda:0
    PYTHONPATH=. python scripts/multi_seed_train.py --protocol 2 --device cuda:0
    PYTHONPATH=. python scripts/multi_seed_train.py --protocol both --device cuda:0
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.algorithms  # noqa — populate REGISTRY
from src.algorithms.base import BaseAlgorithm, REGISTRY
from src.autoencoder.model import BetaVAE_Raman
from src.data.beta_vae_encoder import BetaVAERamanEncoderV2
from src.data.dataset import RAMAN_START_COL_IDX, TARGET_COLS
from src.algorithms.base import Config

# ── Constants ─────────────────────────────────────────────────────────────────
N_RAMAN_COLS   = 2001
LATENT_DIM     = 64
SG_WINDOW, SG_POLY, SG_DERIV = 15, 2, 1
VAE_BATCH_SIZE = 512
VAE_N_EPOCHS   = 300
VAE_LR         = 1e-3
VAE_WD         = 1e-4
VAE_LR_FACTOR  = 0.5
VAE_LR_PAT     = 15
VAE_ES_PAT     = 30
VAE_GRAD_CLIP  = 5.0
BETA           = 0.005   # Selected β from sweep (best S_sep + RMSE trade-off)

LSTM_N_EPOCHS  = 200

REGIME_IDS = {
    "R1":    set(range(0,  30)),
    "R2":    set(range(30, 60)),
    "R3":    set(range(60, 90)),
    "Fault": set(range(90, 100)),
}

SEEDS = [1, 2, 3, 4]

# PI-LSTM config template
_PILSTM_CFG_DICT = {
    "name": "beta_vae_pi_lstm",
    "data": {
        "csv_path":           "./100_Batches_IndPenSim_V3.csv",
        "use_raman":          True,
        "raman_encoder_type": "beta_vae",
        "raman_ckpt":         "",
        "raman_scaler":       "",
        "raman_latent_dim":   LATENT_DIM,
        "split_strategy":     "random",
        "train_frac":         0.80,
        "seed":               42,
        "min_len":            50,
    },
    "model": {
        "hidden_size":     64,
        "num_lstm_layers": 2,
        "fc_hidden":       32,
        "lstm_dropout":    0.2,
        "k_prod_init":     1.0,
    },
    "training": {
        "seq_len":               24,
        "dt":                    0.2,
        "batch_size":            64,
        "n_epochs":              LSTM_N_EPOCHS,
        "lr":                    1e-3,
        "lr_physics":            1e-2,
        "weight_decay":          1e-4,
        "lr_factor":             0.5,
        "lr_patience":           15,
        "es_patience":           30,
        "grad_clip":             5.0,
        "vanilla_mode":          False,
        "lambda_physics_start":  0.0,
        "lambda_physics_end":    0.1,
        "lambda_anneal_delay":   30,
        "lambda_anneal_epochs":  80,
    },
    "output": {
        "checkpoint": "",
        "plots_dir":  "",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_regime_batches(csv_path: str = "./100_Batches_IndPenSim_V3.csv") -> Dict[str, List[pd.DataFrame]]:
    """Load CSV, detect batches, split by regime."""
    print(f"\nLoading CSV: {csv_path} ...")
    df = pd.read_csv(csv_path)
    t_vals = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid
    df["_batch_id"] = batch_ids
    df["_penicillin_sparse"] = df[TARGET_COLS[1]].copy()
    df["_biomass_sparse"]    = df[TARGET_COLS[0]].copy()
    for tc in TARGET_COLS:
        df[tc] = df[tc].interpolate(method="linear").ffill().bfill()

    buckets: Dict[str, List[pd.DataFrame]] = {k: [] for k in REGIME_IDS}
    for b_id in range(int(df["_batch_id"].max()) + 1):
        b = df[df["_batch_id"] == b_id].copy().reset_index(drop=True)
        if len(b) < 50:
            continue
        for regime, ids in REGIME_IDS.items():
            if b_id in ids:
                buckets[regime].append(b)
                break

    for r, bl in buckets.items():
        print(f"  {r}: {len(bl)} batches")
    return buckets


def split_batches(
    batches: List[pd.DataFrame],
    train_frac: float = 0.80,
    seed: int = 42,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """80/20 random split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(batches))
    n_tr = int(len(batches) * train_frac)
    return ([batches[i] for i in perm[:n_tr]],
            [batches[i] for i in perm[n_tr:]])


# ─────────────────────────────────────────────────────────────────────────────
# β-VAE training
# ─────────────────────────────────────────────────────────────────────────────

def _apply_sg(X: np.ndarray) -> np.ndarray:
    return savgol_filter(
        X.astype(np.float64), window_length=SG_WINDOW,
        polyorder=SG_POLY, deriv=SG_DERIV, axis=1,
    ).astype(np.float32)


def _extract_raman(batches: List[pd.DataFrame]) -> np.ndarray:
    parts = []
    for b in batches:
        all_cols = b.columns.tolist()
        n_use = min(N_RAMAN_COLS, len(all_cols) - RAMAN_START_COL_IDX)
        raman_cols = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + n_use]
        X = b[raman_cols].values.astype(np.float32)
        energy = np.abs(X).sum(axis=1) > 0
        if energy.any():
            parts.append(X[energy])
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, N_RAMAN_COLS), np.float32)


def train_beta_vae(
    train_batches: List[pd.DataFrame],
    val_batches:   List[pd.DataFrame],
    ckpt_out:      str,
    scaler_out:    str,
    device:        torch.device,
    seed:          int,
) -> Dict[str, float]:
    """Train β-VAE encoder with fixed β=0.005."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n  [VAE] β={BETA:.4f}  seed={seed}  device={device}")

    X_tr_raw = _extract_raman(train_batches)
    X_va_raw = _extract_raman(val_batches)
    print(f"    Spectra — train: {len(X_tr_raw):,}  val: {len(X_va_raw):,}")

    def _clean(X): return np.clip(np.nan_to_num(X, 0.0), 0.0, None)
    X_tr = _apply_sg(_clean(X_tr_raw))
    X_va = _apply_sg(_clean(X_va_raw))
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_va = scaler.transform(X_va).astype(np.float32)

    dl_tr = DataLoader(TensorDataset(torch.from_numpy(X_tr)),
                       batch_size=VAE_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    dl_va = DataLoader(TensorDataset(torch.from_numpy(X_va)),
                       batch_size=VAE_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = BetaVAE_Raman(input_length=N_RAMAN_COLS, latent_dim=LATENT_DIM).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=VAE_LR, weight_decay=VAE_WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode="min", factor=VAE_LR_FACTOR, patience=VAE_LR_PAT)

    best_recon = float("inf")
    no_improve = 0
    Path(ckpt_out).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for epoch in range(1, VAE_N_EPOCHS + 1):
        model.train()
        tr_recon = 0.0
        for (x,) in dl_tr:
            x = x.to(device)
            optim.zero_grad()
            recon, mu, lv = model(x)
            rec = nn.functional.mse_loss(recon, x)
            kl = -0.5 * torch.mean(torch.sum(1.0 + lv - mu.pow(2) - lv.exp(), dim=1))
            loss = rec + BETA * kl
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), VAE_GRAD_CLIP)
            optim.step()
            tr_recon += rec.item() * len(x)
        tr_recon /= len(dl_tr.dataset)

        model.eval()
        va_recon = 0.0
        with torch.no_grad():
            for (x,) in dl_va:
                x = x.to(device)
                recon, mu, lv = model(x)
                rec = nn.functional.mse_loss(recon, x)
                va_recon += rec.item() * len(x)
        va_recon /= len(dl_va.dataset)
        sched.step(va_recon)

        if epoch % 50 == 0 or epoch == 1:
            print(f"    ep {epoch:3d}/{VAE_N_EPOCHS}  tr_rec={tr_recon:.5f}  "
                  f"va_rec={va_recon:.5f}  lr={optim.param_groups[0]['lr']:.1e}")

        if va_recon < best_recon - 1e-7:
            best_recon = va_recon
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_recon,
                "input_length": N_RAMAN_COLS,
                "latent_dim": LATENT_DIM,
                "beta": BETA,
            }, ckpt_out)
        else:
            no_improve += 1
            if no_improve >= VAE_ES_PAT:
                print(f"    Early stop at epoch {epoch}.")
                break

    elapsed = time.time() - t0
    joblib.dump(scaler, scaler_out)
    print(f"    Best val recon={best_recon:.6f}  time={elapsed/60:.1f}min  → {ckpt_out}")
    return {"val_recon_mse": float(best_recon), "time_min": elapsed/60}


# ─────────────────────────────────────────────────────────────────────────────
# Encode batches with frozen VAE
# ─────────────────────────────────────────────────────────────────────────────

def encode_batches(
    batches:  List[pd.DataFrame],
    encoder:  BetaVAERamanEncoderV2,
) -> List[pd.DataFrame]:
    encoded = []
    for b in batches:
        b2 = b.copy()
        latents = encoder.encode_dataframe(b2)
        for d in range(latents.shape[1]):
            b2[f"_raman_{d}"] = latents[:, d]
        encoded.append(b2)
    return encoded


# ─────────────────────────────────────────────────────────────────────────────
# PI-LSTM training
# ─────────────────────────────────────────────────────────────────────────────

def train_pilstm(
    train_enc:  List[pd.DataFrame],
    val_enc:    List[pd.DataFrame],
    ckpt_out:   str,
    device:     torch.device,
    seed:       int,
) -> object:
    """Train PI-LSTM on pre-encoded batches."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg_dict = copy.deepcopy(_PILSTM_CFG_DICT)
    cfg_dict["data"]["seed"] = seed
    cfg_dict["output"]["checkpoint"] = ckpt_out
    cfg_dict["output"]["plots_dir"] = str(Path(ckpt_out).parent / "plots")
    cfg = Config.from_dict(cfg_dict)
    Path(ckpt_out).parent.mkdir(parents=True, exist_ok=True)

    AlgoCls = REGISTRY["beta_vae_pi_lstm"]
    alg = AlgoCls(cfg, device)

    print(f"    Training PI-LSTM  seed={seed} → {Path(ckpt_out).name}")
    t0 = time.time()
    alg.fit(train_enc, val_enc, verbose=False)
    elapsed = time.time() - t0
    alg.save(ckpt_out)
    print(f"    PI-LSTM done in {elapsed/60:.1f}min → {ckpt_out}")
    return alg


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_batches(alg, batches_enc: List[pd.DataFrame]) -> Dict[str, float]:
    metrics = alg.evaluate(batches_enc)
    return {
        "RMSE": float(metrics["pen"]["RMSE"]),
        "MAE":  float(metrics["pen"]["MAE"]),
        "R2":   float(metrics["pen"]["R2"]),
        "n":    int(metrics["pen"]["n"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_protocol(
    protocol:    int,          # 1 or 2
    buckets:     Dict[str, List[pd.DataFrame]],
    device:      torch.device,
    out_dir:     Path,
):
    """Run full training + eval for one protocol across all seeds."""
    print(f"\n{'='*70}")
    print(f"  PROTOCOL {protocol}  —  {'R1–R3' if protocol == 1 else 'R1 only'}")
    print(f"{'='*70}")

    # Select training batches
    if protocol == 1:
        all_train_batches = buckets["R1"] + buckets["R2"] + buckets["R3"]
    else:
        all_train_batches = buckets["R1"]

    fault = buckets["Fault"]

    results = []

    for seed in SEEDS:
        print(f"\n{'─'*60}")
        print(f"  Protocol {protocol} | Seed {seed}")
        print(f"{'─'*60}")

        # Split into train/val (using the same seed for data split)
        train_batches, val_batches = split_batches(all_train_batches, seed=seed)
        print(f"  Train: {len(train_batches)} batches  Val: {len(val_batches)} batches  Test (R4): {len(fault)}")

        # Paths
        tag = f"p{protocol}_s{seed}"
        vae_ckpt = str(out_dir / f"vae_{tag}.pt")
        vae_scaler = str(out_dir / f"vae_scaler_{tag}.joblib")
        lstm_ckpt = str(out_dir / f"lstm_{tag}.pt")

        # Skip if LSTM already trained (resume support)
        if Path(lstm_ckpt).exists():
            print(f"  LSTM checkpoint exists, loading for eval: {lstm_ckpt}")
            alg = BaseAlgorithm.load(lstm_ckpt, device)
            encoder = BetaVAERamanEncoderV2(
                ckpt_path=vae_ckpt, scaler_path=vae_scaler, device=device,
            )
            val_enc = encode_batches(val_batches, encoder)
            fault_enc = encode_batches(fault, encoder)
            vae_info = {"val_recon_mse": 0.0, "time_min": 0.0}
        else:
            # 1. Train VAE (skip if checkpoint exists)
            if not Path(vae_ckpt).exists():
                vae_info = train_beta_vae(
                    train_batches, val_batches, vae_ckpt, vae_scaler, device, seed,
                )
            else:
                print(f"  VAE checkpoint exists, skipping: {vae_ckpt}")
                vae_info = {"val_recon_mse": 0.0, "time_min": 0.0}

            # 2. Encode all batches
            encoder = BetaVAERamanEncoderV2(
                ckpt_path=vae_ckpt, scaler_path=vae_scaler, device=device,
            )
            print(f"  Encoding batches ...")
            train_enc = encode_batches(train_batches, encoder)
            val_enc   = encode_batches(val_batches, encoder)
            fault_enc = encode_batches(fault, encoder)

            # 3. Train PI-LSTM
            alg = train_pilstm(train_enc, val_enc, lstm_ckpt, device, seed)

        # 4. Evaluate
        print(f"  Evaluating ...")
        val_metrics  = evaluate_on_batches(alg, val_enc)
        test_metrics = evaluate_on_batches(alg, fault_enc)

        print(f"    Val  RMSE={val_metrics['RMSE']:.4f}  R²={val_metrics['R2']:.4f}")
        print(f"    Test RMSE={test_metrics['RMSE']:.4f}  R²={test_metrics['R2']:.4f}")

        results.append({
            "protocol": protocol,
            "seed": seed,
            "vae_val_recon": vae_info["val_recon_mse"],
            "vae_time_min": vae_info["time_min"],
            "val_RMSE":  val_metrics["RMSE"],
            "val_MAE":   val_metrics["MAE"],
            "val_R2":    val_metrics["R2"],
            "test_RMSE": test_metrics["RMSE"],
            "test_MAE":  test_metrics["MAE"],
            "test_R2":   test_metrics["R2"],
            "test_n":    test_metrics["n"],
        })

        # Free memory
        del alg, encoder, val_enc, fault_enc
        if 'train_enc' in dir():
            del train_enc
        torch.cuda.empty_cache()

    # Summary
    val_rmses  = [r["val_RMSE"] for r in results]
    test_rmses = [r["test_RMSE"] for r in results]
    print(f"\n{'='*70}")
    print(f"  PROTOCOL {protocol} SUMMARY (4 seeds)")
    print(f"{'='*70}")
    print(f"  Val  RMSE: {np.mean(val_rmses):.4f} ± {np.std(val_rmses):.4f}")
    print(f"  Test RMSE: {np.mean(test_rmses):.4f} ± {np.std(test_rmses):.4f}")
    print(f"  Individual test RMSEs: {[f'{r:.4f}' for r in test_rmses]}")
    print(f"{'='*70}")

    # Save
    results_path = out_dir / f"protocol{protocol}_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"  Saved → {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-seed β-VAE + PI-LSTM training")
    parser.add_argument("--protocol", default="both", choices=["1", "2", "both"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path("outputs/multi_seed")
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets = load_regime_batches()

    all_results = []
    if args.protocol in ("1", "both"):
        r1 = run_protocol(1, buckets, device, out_dir)
        all_results.extend(r1)
    if args.protocol in ("2", "both"):
        r2 = run_protocol(2, buckets, device, out_dir)
        all_results.extend(r2)

    # Final combined summary
    if args.protocol == "both":
        combined = out_dir / "all_results.json"
        combined.write_text(json.dumps(all_results, indent=2))
        print(f"\nAll results saved → {combined}")


if __name__ == "__main__":
    main()
