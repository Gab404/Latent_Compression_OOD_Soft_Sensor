"""
scripts/train_capacity_vae_lstm.py
====================================
β-VAE with Capacity Annealing (Burgess et al., 2018) + downstream PI-LSTM.

Standard β-VAE loss (Higgins 2017):
    Loss = MSE + β · KL

Capacity-annealed loss (Burgess 2018):
    Loss = MSE + γ · |KL − C(t)|

where C(t) is linearly annealed from 0 → C_max over `epochs_to_C_max` epochs,
then held constant at C_max.  γ is a high constant (e.g. 100) that forces
KL ≈ C exactly — the model is budgeted to use exactly C nats of information.

Equivalence to β = 0.05
------------------------
At convergence, β=0.05 produced:
    total_KL ≈ 2.17 nats  (summed across 64 dims, meaned over batch)
    active dims = 2

To reproduce this information budget:
    C_max  = 4.0   (slightly above 2.17 to give headroom for 2 active dims)
    γ      = 100   (strong constraint: forces KL → C within ~epochs_to_C_max)
    epochs_to_C_max = 100  (C ramps over first 100 epochs, held fixed after)

The key advantage over fixed β: the model is explicitly told HOW MUCH
information capacity to use, rather than indirectly penalising it.
The γ·|KL-C| term acts as a bottleneck controller: if KL > C the model is
pushed to compress; if KL < C it is encouraged to use more capacity.

Training regime
---------------
    Train  : R1 (batch IDs 0–29, 80/20 split for VAE and LSTM)
    Val    : R1 val (20 %)
    Test   : R2 (IDs 30–59), R3 (IDs 60–89), Fault (IDs 90–99)

Usage
-----
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/train_capacity_vae_lstm.py
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/train_capacity_vae_lstm.py \\
        --c-max 4.0 --gamma 100 --epochs-to-c-max 100 --vae-epochs 300
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Project root ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.algorithms  # noqa — populates REGISTRY
from src.algorithms.base import REGISTRY
from src.autoencoder.model import BetaVAE_Raman
from src.config import Config
from src.data.beta_vae_encoder import BetaVAERamanEncoderV2
from src.data.dataset import RAMAN_START_COL_IDX, TARGET_COLS

# ── Hyper-parameters ───────────────────────────────────────────────────────────
N_RAMAN_COLS    = 2001
LATENT_DIM      = 64
SG_WINDOW, SG_POLY, SG_DERIV = 15, 2, 1

# Capacity annealing defaults (equivalent to β=0.05)
GAMMA_DEFAULT         = 100.0   # constraint strength
C_MAX_DEFAULT         = 4.0     # target KL budget (nats)  — β=0.05 settled at ~2.17
EPOCHS_TO_C_MAX       = 100     # linear ramp duration (epochs)

# VAE training
VAE_BATCH_SIZE  = 512
VAE_N_EPOCHS    = 300
VAE_LR          = 1e-3
VAE_WD          = 1e-4
VAE_LR_FACTOR   = 0.5
VAE_LR_PAT      = 15
VAE_ES_PAT      = 30
VAE_GRAD_CLIP   = 5.0

# LSTM training
LSTM_N_EPOCHS   = 200
SEED            = 42

REGIME_IDS = {
    "R1":    list(range(0,  30)),
    "R2":    list(range(30, 60)),
    "R3":    list(range(60, 90)),
    "Fault": list(range(90, 100)),
}

# PI-LSTM config (pre-encoded: raman_ckpt/scaler left blank)
_PILSTM_CFG = {
    "name": "capacity_vae_pi_lstm",
    "data": {
        "csv_path":           "./100_Batches_IndPenSim_V3.csv",
        "use_raman":          True,
        "raman_encoder_type": "beta_vae",
        "raman_ckpt":         "",
        "raman_scaler":       "",
        "raman_latent_dim":   LATENT_DIM,
        "split_strategy":     "random",
        "train_frac":         0.80,
        "seed":               SEED,
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
    "output": {"checkpoint": "", "plots_dir": ""},
}

OUTPUT_DIR = Path("./outputs/capacity_vae_lstm")


# ─────────────────────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────────────────────

def capacity_vae_loss(
    recon:   torch.Tensor,
    target:  torch.Tensor,
    mu:      torch.Tensor,
    log_var: torch.Tensor,
    C:       float,
    gamma:   float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Capacity-annealed β-VAE loss (Burgess et al., 2018).

        Loss = MSE(recon, target) + γ · |KL(q||p) − C|

    KL is summed over latent dims and averaged over the batch
    (same convention as beta_sweep_e2e._beta_vae_loss), so C is
    expressed in the same nats unit as the sweep's total_KL column.

    Parameters
    ----------
    C     : current capacity target (nats) — annealed externally
    gamma : constraint strength — high value forces KL ≈ C exactly

    Returns
    -------
    total      : scalar loss for backprop
    recon_loss : MSE term (for logging)
    kl_loss    : KL term before capacity penalty (for logging)
    """
    recon_loss = nn.functional.mse_loss(recon, target)
    kl_loss    = -0.5 * torch.mean(
        torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    )
    total = recon_loss + gamma * torch.abs(kl_loss - C)
    return total, recon_loss, kl_loss


def capacity_schedule(epoch: int, epochs_to_c_max: int, c_max: float) -> float:
    """
    Linear ramp: C(epoch) = min(c_max, c_max * epoch / epochs_to_c_max).

    epoch is 1-indexed.  C reaches c_max at epoch = epochs_to_c_max.
    """
    return float(min(c_max, c_max * epoch / max(epochs_to_c_max, 1)))


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_regime_batches(
    csv_path: str,
    min_len:  int = 50,
) -> Dict[str, List[pd.DataFrame]]:
    """Load CSV and split into R1 / R2 / R3 / Fault buckets."""
    print(f"\nLoading CSV: {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape[0]:,} × {df.shape[1]}")

    t_vals    = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid
    df["_batch_id"] = batch_ids

    # Sparse penicillin / biomass columns (kept as-is for LSTM)
    for tc in TARGET_COLS:
        df[f"_sparse_{tc}"] = df[tc].copy()
        df[tc] = df[tc].interpolate(method="linear").ffill().bfill()

    pen_col = TARGET_COLS[1]
    bio_col = TARGET_COLS[0]
    df["_penicillin_sparse"] = df[f"_sparse_{pen_col}"]
    df["_biomass_sparse"]    = df[f"_sparse_{bio_col}"]

    buckets: Dict[str, List[pd.DataFrame]] = {k: [] for k in REGIME_IDS}
    for regime, ids in REGIME_IDS.items():
        for bid in ids:
            b = df[df["_batch_id"] == bid].copy().reset_index(drop=True)
            if len(b) >= min_len:
                b["_batch_id"] = bid
                buckets[regime].append(b)
        print(f"  {regime}: {len(buckets[regime])} batches")

    return buckets


def split_r1(
    r1_batches: List[pd.DataFrame],
    train_frac: float = 0.80,
    seed: int = SEED,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(len(r1_batches))
    n_tr  = max(1, int(train_frac * len(r1_batches)))
    return ([r1_batches[i] for i in idx[:n_tr]],
            [r1_batches[i] for i in idx[n_tr:]])


# ─────────────────────────────────────────────────────────────────────────────
# VAE helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_raman(batches: List[pd.DataFrame]) -> np.ndarray:
    parts = []
    for b in batches:
        cols  = b.columns[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + N_RAMAN_COLS]
        X     = b[cols].values.astype(np.float32)
        energy = np.abs(X).sum(axis=1) > 0
        parts.append(X[energy])
    return np.vstack(parts) if parts else np.zeros((0, N_RAMAN_COLS), dtype=np.float32)


def _apply_sg(X: np.ndarray) -> np.ndarray:
    return savgol_filter(X, window_length=SG_WINDOW,
                         polyorder=SG_POLY, deriv=SG_DERIV, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# VAE training
# ─────────────────────────────────────────────────────────────────────────────

def train_capacity_vae(
    train_batches:  List[pd.DataFrame],
    val_batches:    List[pd.DataFrame],
    ckpt_out:       str,
    scaler_out:     str,
    gamma:          float = GAMMA_DEFAULT,
    c_max:          float = C_MAX_DEFAULT,
    epochs_to_c_max: int  = EPOCHS_TO_C_MAX,
    n_epochs:       int   = VAE_N_EPOCHS,
    device:         torch.device = torch.device("cpu"),
    seed:           int   = SEED,
) -> Dict:
    """
    Train BetaVAE_Raman with capacity annealing on R1 spectra.

    The capacity C is linearly ramped from 0 → c_max over epochs_to_c_max,
    then held constant.  The training monitors reconstruction MSE for early
    stopping (same as the β sweep) so the checkpoint is the best recon model.

    Returns dict with val_recon_mse, val_kl, final_C, epoch.
    """
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    print(f"\n{'─'*62}")
    print(f"  Capacity VAE  γ={gamma}  C_max={c_max}  ramp={epochs_to_c_max} ep")
    print(f"{'─'*62}")

    # ── Prepare spectra ────────────────────────────────────────────────────────
    def _clean(X): return np.clip(np.nan_to_num(X, 0.0), 0.0, None)

    X_tr = _apply_sg(_clean(_extract_raman(train_batches)))
    X_va = _apply_sg(_clean(_extract_raman(val_batches)))
    print(f"  Spectra — train: {len(X_tr):,}  val: {len(X_va):,}")

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr).astype(np.float32)
    X_va   = scaler.transform(X_va).astype(np.float32)

    dl_tr = DataLoader(TensorDataset(torch.from_numpy(X_tr)),
                       batch_size=VAE_BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    dl_va = DataLoader(TensorDataset(torch.from_numpy(X_va)),
                       batch_size=VAE_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = BetaVAE_Raman(input_length=N_RAMAN_COLS, latent_dim=LATENT_DIM).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=VAE_LR, weight_decay=VAE_WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=VAE_LR_FACTOR, patience=VAE_LR_PAT, verbose=False)

    best_recon  = float("inf")
    no_improve  = 0
    history     = []   # (epoch, C, tr_recon, tr_kl, va_recon, va_kl)
    Path(ckpt_out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, n_epochs + 1):
        C = capacity_schedule(epoch, epochs_to_c_max, c_max)

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        tr_recon = tr_kl = 0.0
        for (x,) in dl_tr:
            x = x.to(device)
            optim.zero_grad()
            recon, mu, lv = model(x)
            loss, rec, kl = capacity_vae_loss(recon, x, mu, lv, C=C, gamma=gamma)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), VAE_GRAD_CLIP)
            optim.step()
            tr_recon += rec.item() * len(x)
            tr_kl    += kl.item()  * len(x)
        N = len(dl_tr.dataset)
        tr_recon /= N; tr_kl /= N

        # ── Val ────────────────────────────────────────────────────────────────
        model.eval()
        va_recon = va_kl = 0.0
        with torch.no_grad():
            for (x,) in dl_va:
                x = x.to(device)
                recon, mu, lv = model(x)
                _, rec, kl    = capacity_vae_loss(recon, x, mu, lv, C=C, gamma=gamma)
                va_recon += rec.item() * len(x)
                va_kl    += kl.item()  * len(x)
        Nv = len(dl_va.dataset)
        va_recon /= Nv; va_kl /= Nv

        sched.step(va_recon)
        history.append((epoch, C, tr_recon, tr_kl, va_recon, va_kl))

        if epoch % 20 == 0 or epoch == 1:
            print(f"  ep {epoch:4d}/{n_epochs}  C={C:.3f}/{c_max}  "
                  f"tr_rec={tr_recon:.5f} tr_kl={tr_kl:.4f}  "
                  f"va_rec={va_recon:.5f} va_kl={va_kl:.4f}  "
                  f"lr={optim.param_groups[0]['lr']:.1e}")

        # ── Checkpoint on reconstruction improvement ───────────────────────────
        if va_recon < best_recon - 1e-7:
            best_recon = va_recon
            no_improve = 0
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":         best_recon,
                "val_kl":           va_kl,
                "C":                C,
                "c_max":            c_max,
                "gamma":            gamma,
                "input_length":     N_RAMAN_COLS,
                "latent_dim":       LATENT_DIM,
            }, ckpt_out)
        else:
            no_improve += 1
            if no_improve >= VAE_ES_PAT:
                print(f"  Early stop at epoch {epoch}.")
                break

    joblib.dump(scaler, scaler_out)
    saved = torch.load(ckpt_out, map_location="cpu", weights_only=False)
    print(f"  Best val recon={saved['val_loss']:.6f}  kl={saved['val_kl']:.4f}  "
          f"C@best={saved['C']:.3f}  epoch={saved['epoch']}")
    print(f"  Saved → {ckpt_out}")

    return {
        "val_recon_mse": float(saved["val_loss"]),
        "val_kl":        float(saved["val_kl"]),
        "C_at_best":     float(saved["C"]),
        "epoch":         int(saved["epoch"]),
        "history":       history,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Encoder + batch encoding
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
# PI-LSTM training
# ─────────────────────────────────────────────────────────────────────────────

def train_pilstm(
    train_enc: List[pd.DataFrame],
    val_enc:   List[pd.DataFrame],
    ckpt_out:  str,
    device:    torch.device,
    n_epochs:  Optional[int] = None,
) -> object:
    cfg_dict = copy.deepcopy(_PILSTM_CFG)
    cfg_dict["output"]["checkpoint"] = ckpt_out
    cfg_dict["output"]["plots_dir"]  = str(Path(ckpt_out).parent / "plots")
    if n_epochs is not None:
        cfg_dict["training"]["n_epochs"] = n_epochs
    cfg = Config.from_dict(cfg_dict)
    Path(ckpt_out).parent.mkdir(parents=True, exist_ok=True)

    AlgoCls = REGISTRY["beta_vae_pi_lstm"]
    alg     = AlgoCls(cfg, device)
    print(f"\n  Training PI-LSTM  →  {Path(ckpt_out).name}")
    alg.fit(train_enc, val_enc, verbose=False)
    alg.save(ckpt_out)
    print(f"  Saved → {ckpt_out}")
    return alg


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(alg, batches: List[pd.DataFrame], label: str) -> Dict:
    metrics = alg.evaluate(batches)
    rmse = float(metrics["pen"]["RMSE"])
    mae  = float(metrics["pen"]["MAE"])
    r2   = float(metrics["pen"]["R2"])
    print(f"  [{label:<8}]  Pen RMSE={rmse:.4f} g/L  MAE={mae:.4f}  R²={r2:.4f}")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


# ─────────────────────────────────────────────────────────────────────────────
# Capacity annealing plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_capacity_training(
    history: list,
    c_max:   float,
    save_path: str,
) -> None:
    """
    Three-panel training curve:
      Top    : C(t) vs KL(t) — shows the controller driving KL → C
      Middle : reconstruction MSE (train / val)
      Bottom : γ·|KL−C| penalty term
    """
    epochs  = [h[0] for h in history]
    C_vals  = [h[1] for h in history]
    tr_rec  = [h[2] for h in history]
    tr_kl   = [h[3] for h in history]
    va_rec  = [h[4] for h in history]
    va_kl   = [h[5] for h in history]
    penalty = [abs(kl - c) for kl, c in zip(va_kl, C_vals)]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Panel 1: C vs KL
    ax = axes[0]
    ax.plot(epochs, C_vals, "k--", linewidth=2, label=f"C(t) target (→ {c_max})")
    ax.plot(epochs, va_kl,  color="#e74c3c", linewidth=1.8, label="KL val")
    ax.plot(epochs, tr_kl,  color="#e74c3c", linewidth=1.0, alpha=0.4, label="KL train")
    ax.set_ylabel("KL / C  (nats)", fontsize=10)
    ax.set_title("Capacity Controller: KL tracks C(t)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.set_ylim(bottom=0)

    # Panel 2: Reconstruction MSE
    ax2 = axes[1]
    ax2.plot(epochs, va_rec, color="#2980b9", linewidth=1.8, label="Recon MSE val")
    ax2.plot(epochs, tr_rec, color="#2980b9", linewidth=1.0, alpha=0.4, label="Recon MSE train")
    ax2.set_ylabel("MSE (normalised)", fontsize=10)
    ax2.set_title("Reconstruction Loss", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9); ax2.set_ylim(bottom=0)

    # Panel 3: Penalty |KL - C|
    ax3 = axes[2]
    ax3.plot(epochs, penalty, color="#27ae60", linewidth=1.8)
    ax3.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax3.set_ylabel("|KL − C|  (nats)", fontsize=10)
    ax3.set_xlabel("Epoch", fontsize=10)
    ax3.set_title("Capacity Constraint Violation |KL − C|", fontsize=11, fontweight="bold")
    ax3.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curve saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    gamma:           float = GAMMA_DEFAULT,
    c_max:           float = C_MAX_DEFAULT,
    epochs_to_c_max: int   = EPOCHS_TO_C_MAX,
    vae_epochs:      int   = VAE_N_EPOCHS,
    lstm_epochs:     int   = LSTM_N_EPOCHS,
    csv_path:        str   = "./100_Batches_IndPenSim_V3.csv",
    out_dir:         Path  = OUTPUT_DIR,
    device:          torch.device = torch.device("cuda"),
) -> None:
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"\n{'='*62}")
    print(f"  Capacity VAE + PI-LSTM")
    print(f"  γ={gamma}  C_max={c_max}  ramp={epochs_to_c_max} ep  device={device}")
    print(f"{'='*62}")

    # ── 1. Load data ───────────────────────────────────────────────────────────
    regime_batches = load_regime_batches(csv_path)
    r1_train, r1_val = split_r1(regime_batches["R1"], seed=SEED)
    print(f"\n  R1 split — train: {len(r1_train)}  val: {len(r1_val)}")

    # ── 2. Train capacity-annealed VAE ─────────────────────────────────────────
    vae_ckpt   = str(out_dir / "vae.pt")
    vae_scaler = str(out_dir / "vae_scaler.joblib")

    vae_metrics = train_capacity_vae(
        r1_train, r1_val,
        ckpt_out=vae_ckpt, scaler_out=vae_scaler,
        gamma=gamma, c_max=c_max, epochs_to_c_max=epochs_to_c_max,
        n_epochs=vae_epochs, device=device, seed=SEED,
    )

    plot_capacity_training(
        vae_metrics["history"], c_max=c_max,
        save_path=str(out_dir / "vae_training_curve.png"),
    )

    # ── 3. Freeze encoder, pre-encode all regimes ──────────────────────────────
    torch.cuda.empty_cache()
    encoder = BetaVAERamanEncoderV2(vae_ckpt, vae_scaler, device)

    print("\n  Pre-encoding all regimes ...")
    r1_train_enc = encode_batches(r1_train, encoder)
    r1_val_enc   = encode_batches(r1_val,   encoder)
    r2_enc       = encode_batches(regime_batches["R2"],    encoder)
    r3_enc       = encode_batches(regime_batches["R3"],    encoder)
    fault_enc    = encode_batches(regime_batches["Fault"], encoder)

    # ── 4. Train PI-LSTM on R1 ─────────────────────────────────────────────────
    lstm_ckpt = str(out_dir / "lstm.pt")
    alg = train_pilstm(r1_train_enc, r1_val_enc, lstm_ckpt, device,
                       n_epochs=lstm_epochs)

    # ── 5. Evaluate on all regimes ─────────────────────────────────────────────
    torch.cuda.empty_cache()
    print(f"\n{'─'*40}")
    print("  Downstream RMSE — Penicillin (g/L)")
    print(f"{'─'*40}")
    m_r1val = evaluate(alg, r1_val_enc,  "R1 val")
    m_r2    = evaluate(alg, r2_enc,      "R2")
    m_r3    = evaluate(alg, r3_enc,      "R3")
    m_fault = evaluate(alg, fault_enc,   "Fault")

    elapsed = (time.time() - t0) / 60

    # ── 6. Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  CAPACITY VAE + PI-LSTM — FINAL RESULTS")
    print(f"{'='*62}")
    print(f"  VAE  — best recon MSE : {vae_metrics['val_recon_mse']:.5f}")
    print(f"       — best val KL    : {vae_metrics['val_kl']:.4f}  (target C_max={c_max})")
    print(f"       — C at best epoch: {vae_metrics['C_at_best']:.3f}")
    print(f"       — best epoch     : {vae_metrics['epoch']}")
    print(f"  LSTM — R1 val  RMSE   : {m_r1val['RMSE']:.4f} g/L")
    print(f"       — R2      RMSE   : {m_r2['RMSE']:.4f} g/L")
    print(f"       — R3      RMSE   : {m_r3['RMSE']:.4f} g/L")
    print(f"       — Fault   RMSE   : {m_fault['RMSE']:.4f} g/L")
    print(f"  OOD degradation (Fault vs R1 val): "
          f"{(m_fault['RMSE'] - m_r1val['RMSE']) / (m_r1val['RMSE'] + 1e-8):.1%}")
    print(f"  Total time: {elapsed:.1f} min")
    print(f"  Outputs: {out_dir}/")
    print(f"{'='*62}")

    # Save results CSV
    import csv
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["gamma",           gamma])
        w.writerow(["c_max",           c_max])
        w.writerow(["epochs_to_c_max", epochs_to_c_max])
        w.writerow(["vae_val_recon",   vae_metrics["val_recon_mse"]])
        w.writerow(["vae_val_kl",      vae_metrics["val_kl"]])
        w.writerow(["vae_C_at_best",   vae_metrics["C_at_best"]])
        w.writerow(["vae_epoch",       vae_metrics["epoch"]])
        w.writerow(["lstm_rmse_r1val", m_r1val["RMSE"]])
        w.writerow(["lstm_rmse_r2",    m_r2["RMSE"]])
        w.writerow(["lstm_rmse_r3",    m_r3["RMSE"]])
        w.writerow(["lstm_rmse_fault", m_fault["RMSE"]])
        w.writerow(["ood_degradation", (m_fault["RMSE"] - m_r1val["RMSE"]) / (m_r1val["RMSE"] + 1e-8)])
    print(f"  Results CSV → {out_dir / 'results.csv'}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capacity-annealed β-VAE + PI-LSTM (Burgess et al. 2018)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gamma",           type=float, default=GAMMA_DEFAULT,
                        help="Capacity constraint strength γ")
    parser.add_argument("--c-max",           type=float, default=C_MAX_DEFAULT,
                        help="Maximum capacity C_max (nats)")
    parser.add_argument("--epochs-to-c-max", type=int,   default=EPOCHS_TO_C_MAX,
                        help="Epochs to linearly ramp C from 0 to C_max")
    parser.add_argument("--vae-epochs",      type=int,   default=VAE_N_EPOCHS,
                        help="Max VAE training epochs")
    parser.add_argument("--lstm-epochs",     type=int,   default=LSTM_N_EPOCHS,
                        help="Max LSTM training epochs")
    parser.add_argument("--csv",             default="./100_Batches_IndPenSim_V3.csv")
    parser.add_argument("--out-dir",         default=str(OUTPUT_DIR))
    parser.add_argument("--device",          default="cuda")
    args = parser.parse_args()

    main(
        gamma           = args.gamma,
        c_max           = args.c_max,
        epochs_to_c_max = args.epochs_to_c_max,
        vae_epochs      = args.vae_epochs,
        lstm_epochs     = args.lstm_epochs,
        csv_path        = args.csv,
        out_dir         = Path(args.out_dir),
        device          = torch.device(args.device if torch.cuda.is_available() else "cpu"),
    )
