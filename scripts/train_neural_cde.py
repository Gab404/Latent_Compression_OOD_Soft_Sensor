#!/usr/bin/env python
"""
train_neural_cde.py
===================
Full training pipeline for the Neural CDE soft-sensor on IndPenSim V3.

Pipeline summary
----------------
  frozen β-VAE (β=0.05)
      ↓  encode Raman spectra  →  Z = [z₅₂, z₂₇]   (2-D active latent)
      ↓
  build_path(Z, U, t)          →  X  (B, T, 7)      path tensor
  compute_coeffs(X, t)         →  coeffs (B,T-1,28)  cubic spline (offline)
      ↓
  NeuralCDE.forward(coeffs, t) →  ŷ(t) (B, T)       Penicillin g/L
      ↓
  MSE loss at sparse lab-measurement points only
  AdamW + ReduceLROnPlateau + early stopping

Input path  X = [z₅₂, z₂₇, Fs, Fg, pH, T_ctrl, t_norm]
  • z₅₂, z₂₇  — β-VAE active latent dims (Penicillin and Glucose proxies)
  • Fs, Fg     — glucose feed rate and aeration rate (key process drivers)
  • pH, T_ctrl — pH and temperature (kinetics modulators)
  • t_norm     — physical time normalised to [0, 1]

Compared to Neural ODE
-----------------------
  Neural ODE  dh/dt = f(h, t)         — autonomous; blind to U after t=0
  Neural CDE  dh    = f(h) dX(t)      — every control change is injected
                                         continuously through dX/dt

Control variables are NOT fed as discrete inputs to a recurrent gate;
they enter continuously through the path derivative dX/dt, so even
piecewise-constant control signals (e.g. step changes in feed rate) are
exactly represented by the spline and faithfully integrated.

Usage
-----
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_neural_cde.py

    # Larger hidden dim for richer dynamics
    PYTHONPATH=. .venv/bin/python scripts/train_neural_cde.py --hidden-dim 64

    # Use dopri5 adaptive solver (slower, more accurate on stiff batches)
    PYTHONPATH=. .venv/bin/python scripts/train_neural_cde.py --solver dopri5

    # Evaluate R1 train set as well
    PYTHONPATH=. .venv/bin/python scripts/train_neural_cde.py --eval-r1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.beta_vae_encoder import BetaVAERamanEncoderV2
from src.data.dataset import TARGET_COLS
from src.models.neural_cde import NeuralCDE, build_path, compute_coeffs

# ─────────────────────────────────────────────────────────────────────────────
# Configuration constants
# ─────────────────────────────────────────────────────────────────────────────

CSV_PATH   = str(_ROOT / "100_Batches_IndPenSim_V3.csv")
VAE_CKPT   = str(_ROOT / "outputs/beta_sweep_e2e/checkpoints/b0.05/vae.pt")
VAE_SCALER = str(_ROOT / "outputs/beta_sweep_e2e/checkpoints/b0.05/vae_scaler.joblib")
OUT_DIR    = _ROOT / "outputs/neural_cde"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Regime batch ID boundaries (0-indexed)
R1_IDS    = set(range(0,  30))
FAULT_IDS = set(range(90, 100))

# β-VAE active latent dimensions (from latent trajectory analysis)
ACTIVE_DIMS  = [52, 27]   # z₅₂ = Penicillin proxy, z₂₇ = Glucose proxy
N_LATENT     = 2

# Control variable columns  →  4 channels
CONTROL_COLS = [
    "Sugar feed rate(Fs:L/h)",
    "Aeration rate(Fg:L/h)",
    "pH(pH:pH)",
    "Temperature(T:K)",
]
CONTROL_NAMES = ["Fs", "Fg", "pH", "T"]
N_CTRL        = len(CONTROL_COLS)

# input_dim = N_LATENT + N_CTRL + 1 (time channel)
INPUT_DIM = N_LATENT + N_CTRL + 1   # = 7

# Physical time step
DT = 0.2   # hours

SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_batches(
    csv_path:   str,
    regime_ids: set,
    min_len:    int = 50,
) -> List[pd.DataFrame]:
    """Load IndPenSim CSV, detect batch boundaries, return per-batch DataFrames."""
    print(f"\nLoading CSV: {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

    t_vals    = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid
    df["_batch_id"] = batch_ids

    # Preserve sparse lab measurements before interpolating targets
    df["_penicillin_sparse"] = df[TARGET_COLS[1]].copy()
    df["_biomass_sparse"]    = df[TARGET_COLS[0]].copy()

    # Dense-interpolate targets (used for dense training loss)
    for tc in TARGET_COLS:
        df[tc] = df[tc].interpolate(method="linear").ffill().bfill()

    batches = []
    for b_id in sorted(regime_ids):
        sub = df[df["_batch_id"] == b_id].reset_index(drop=True)
        if len(sub) >= min_len:
            batches.append(sub)
    print(f"  Loaded {len(batches)} batches.")
    return batches


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Encoding + control extraction
# ─────────────────────────────────────────────────────────────────────────────

def encode_latents(
    batches: List[pd.DataFrame],
    encoder: BetaVAERamanEncoderV2,
    active_dims: List[int] = ACTIVE_DIMS,
) -> List[np.ndarray]:
    """Encode each batch with the frozen β-VAE, extract active dims → Z list."""
    Z_list = []
    for b in batches:
        mu_full = encoder.encode_dataframe(b)             # (T, 64)
        Z_list.append(mu_full[:, active_dims].astype(np.float32))  # (T, 2)
    return Z_list


def extract_controls(
    batches:    List[pd.DataFrame],
    ctrl_cols:  List[str] = CONTROL_COLS,
    scaler:     Optional[MinMaxScaler] = None,
    fit:        bool = True,
) -> Tuple[List[np.ndarray], MinMaxScaler]:
    """
    Extract control variables from each batch and min-max scale to [0, 1].

    Scaling is important: U channels would otherwise dominate the path X
    relative to Z (which is already near-unit variance from the VAE).

    Returns U_list and the fitted scaler (pass scaler back for test batches).
    """
    all_u = np.vstack([b[ctrl_cols].values.astype(np.float32) for b in batches])
    all_u = np.nan_to_num(all_u)
    if scaler is None:
        scaler = MinMaxScaler()
    if fit:
        scaler.fit(all_u)
    U_list = []
    for b in batches:
        u = np.nan_to_num(b[ctrl_cols].values.astype(np.float32))
        U_list.append(scaler.transform(u).astype(np.float32))  # (T, 4)
    return U_list, scaler


def extract_penicillin(
    batches: List[pd.DataFrame],
    scaler:  Optional[MinMaxScaler] = None,
    fit:     bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray], MinMaxScaler]:
    """
    Extract dense (interpolated) and sparse (lab-measurement) Penicillin arrays.

    The model is trained on dense targets (every 0.2 h step) but evaluated
    only at the sparse offline lab-measurement points (~every 12 h).
    """
    dense_col  = TARGET_COLS[1]                 # "Penicillin concentration(P:g/L)"
    sparse_col = "_penicillin_sparse"

    all_dense = np.concatenate([b[dense_col].values for b in batches]).reshape(-1, 1)
    if scaler is None:
        scaler = MinMaxScaler()
    if fit:
        scaler.fit(all_dense.astype(np.float32))

    Y_dense_list  = []
    Y_sparse_list = []
    for b in batches:
        y_d = scaler.transform(b[dense_col].values.reshape(-1, 1)).flatten()
        Y_dense_list.append(y_d.astype(np.float32))
        Y_sparse_list.append(b[sparse_col].values.astype(np.float32))  # NaN outside lab pts
    return Y_dense_list, Y_sparse_list, scaler


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Dataset: pre-compute spline coefficients
# ─────────────────────────────────────────────────────────────────────────────

class PenicillinCDEDataset(Dataset):
    """
    PyTorch Dataset that pre-computes cubic spline coefficients.

    Pre-computing coefficients here (once, in __init__) rather than in the
    forward pass means:
      • DataLoader workers do the heavy lifting once.
      • The forward pass is pure tensor ops (CubicSpline just wraps coeffs).

    Each item contains:
        coeffs  : (T-1, 4*INPUT_DIM)  — cubic spline coefficients for X path
        t_norm  : (T,)                — normalised time grid [0, 1]
        y_dense : (T,)                — normalised Penicillin (dense)
        y_sparse: (T,)                — Penicillin g/L at lab pts (NaN elsewhere)
        X0      : (INPUT_DIM,)        — first path observation X(t₀)  [for reference]

    Parameters
    ----------
    Z_list      : list of (T_i, 2)   — β-VAE active latent dims per batch
    U_list      : list of (T_i, 4)   — scaled control vars per batch
    Y_dense_list: list of (T_i,)     — normalised dense Penicillin targets
    Y_sparse_list: list of (T_i,)    — sparse (NaN-masked) Penicillin in g/L
    pad_to      : optional int       — pad/truncate all sequences to this length
                  If None, each item has its own length (use collate_fn for
                  variable-length batching, or set pad_to for fixed-length).
    """

    def __init__(
        self,
        Z_list:       List[np.ndarray],
        U_list:       List[np.ndarray],
        Y_dense_list: List[np.ndarray],
        Y_sparse_list: List[np.ndarray],
        pad_to:       Optional[int] = None,
    ) -> None:
        assert len(Z_list) == len(U_list) == len(Y_dense_list)
        self.items: List[Dict[str, torch.Tensor]] = []

        for Z, U, Y_d, Y_s in zip(Z_list, U_list, Y_dense_list, Y_sparse_list):
            T = len(Z)
            if pad_to is not None:
                if T > pad_to:
                    # Truncate sequences longer than pad_to
                    T = pad_to
                    Z, U, Y_d, Y_s = Z[:T], U[:T], Y_d[:T], Y_s[:T]
                elif T < pad_to:
                    # Pad sequences shorter than pad_to by repeating the last
                    # observation.  In the CDE path, dZ/dt and dU/dt → 0 in
                    # the padded region (flat spline), so the CDE dynamics are
                    # driven only by the time channel — a mild, well-defined
                    # extrapolation.  The padded target rows are NaN in the
                    # sparse signal (no lab measurements there), so they do not
                    # contribute to the sparse loss.
                    pad = pad_to - T
                    Z   = np.concatenate([Z,   np.tile(Z[-1:],   (pad, 1))], axis=0)
                    U   = np.concatenate([U,   np.tile(U[-1:],   (pad, 1))], axis=0)
                    Y_d = np.concatenate([Y_d, np.tile(Y_d[-1:], (pad,))],   axis=0)
                    # Sparse labels for padded region are NaN (no measurements)
                    Y_s = np.concatenate([Y_s, np.full(pad, np.nan)],         axis=0)
                    T   = pad_to

            # ── Normalised time grid: 0..1 ────────────────────────────────────
            # Normalising to [0, 1] regardless of physical batch length ensures
            # the ODE sees the same time range during training and inference.
            t_norm = torch.linspace(0.0, 1.0, T)

            # ── Build path tensor X = [Z, U, t] ───────────────────────────────
            Z_t = torch.tensor(Z, dtype=torch.float32)   # (T, 2)
            U_t = torch.tensor(U, dtype=torch.float32)   # (T, 4)
            # build_path expects (B, T, *) — add and remove batch dim
            X = build_path(
                Z_t.unsqueeze(0),    # (1, T, 2)
                U_t.unsqueeze(0),    # (1, T, 4)
                t_norm,              # (T,)
            ).squeeze(0)             # → (T, 7)

            # ── Pre-compute spline coefficients (the expensive step) ───────────
            # natural_cubic_coeffs(X, t) takes X: (B, T, I) and t: (T,).
            # We add a dummy batch dim, compute, then remove it.
            coeffs = compute_coeffs(
                X.unsqueeze(0),      # (1, T, 7)
                t_norm,              # (T,)
            ).squeeze(0)             # → (T-1, 28)
            # coeffs shape: (T-1, 4 * INPUT_DIM) = (T-1, 28)
            # Each of the (T-1) segments has 4 polynomial coefficients per
            # input channel.  Stored as float32, ~0.1 MB per batch.

            self.items.append({
                "coeffs":   coeffs,                               # (T-1, 28)
                "t_norm":   t_norm,                               # (T,)
                "y_dense":  torch.tensor(Y_d, dtype=torch.float32),  # (T,)
                "y_sparse": torch.tensor(Y_s, dtype=torch.float32),  # (T,) w/ NaN
                "X0":       X[0],                                 # (7,) first obs
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.items[idx]


def variable_length_collate(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, List[torch.Tensor]]:
    """
    Collate function for variable-length sequences.

    Because fermentation batches have different lengths (850–1450 steps),
    we return lists of tensors rather than a stacked tensor.  The model's
    forward pass handles each trajectory independently through its own spline.

    For fixed-length training (pad_to=N in Dataset), use the default collate.
    """
    keys = batch[0].keys()
    return {k: [item[k] for item in batch] for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_one_batch(
    model:      NeuralCDE,
    coeffs:     torch.Tensor,   # (B, T-1, 28)
    t_norm:     torch.Tensor,   # (T,)           — same grid for all in mini-batch
    y_target:   torch.Tensor,   # (B, T)
    sparse_mask: Optional[torch.Tensor],  # (B, T) bool | None
    loss_fn:    nn.Module,
    optimizer:  Optional[torch.optim.Optimizer],
    training:   bool,
    grad_clip:  float = 5.0,
) -> float:
    """
    Execute one forward (+backward) pass for a mini-batch.

    This is the canonical training-loop snippet showing the full flow:
        coeffs → CubicSpline → cdeint → readout → MSE loss → backprop

    Parameters
    ----------
    coeffs      : (B, T-1, 4*I)  — spline coefficients from DataLoader
    t_norm      : (T,)            — normalised time grid
    y_target    : (B, T)          — Penicillin targets (normalised)
    sparse_mask : (B, T) bool or None
                  If provided, loss is restricted to sparse lab-measurement
                  time points.  This matches the IndPenSim evaluation protocol.
                  If None, all time steps contribute to the loss (dense).
    loss_fn     : MSELoss or HuberLoss
    optimizer   : set to None for validation (no backprop)
    training    : if True, call model.train() and backprop; else model.eval()
    grad_clip   : max-norm for gradient clipping

    Returns
    -------
    loss_value : float
    """
    if training:
        model.train()
        optimizer.zero_grad(set_to_none=True)
    else:
        model.eval()

    # ── Forward pass ──────────────────────────────────────────────────────────
    # model.forward(coeffs, t_norm) internally:
    #   1. spline = CubicSpline(coeffs, t=t_norm)     — wrap precomputed coeffs
    #   2. X0 = spline.evaluate(t_norm[0])             — first path observation
    #   3. h0 = initial_net(X0)                        — (B, H)
    #   4. h  = cdeint(spline, cde_func, h0, t_norm)   — (B, T, H)
    #   5. ŷ  = readout(h).squeeze(-1)                 — (B, T)
    with torch.set_grad_enabled(training):
        y_hat = model(coeffs, t_norm)                  # (B, T)

        # ── Loss: sparse or dense ──────────────────────────────────────────────
        if sparse_mask is not None:
            # Evaluate loss only at the ~20 offline lab measurement points.
            # This avoids penalising the model on densely-interpolated targets
            # (which are not real measurements) and matches the evaluation KPI.
            y_hat_sp  = y_hat[sparse_mask]             # (n_pts,)
            y_tgt_sp  = y_target[sparse_mask]          # (n_pts,)
            loss = loss_fn(y_hat_sp, y_tgt_sp)
        else:
            # Dense loss: all ~1150 time steps.  Faster convergence but
            # risks over-fitting to the interpolated targets.
            loss = loss_fn(y_hat, y_target)

    if training and loss.requires_grad:
        loss.backward()
        # Gradient clipping: the adjoint ODE can produce large gradients for
        # long sequences if the CDE dynamics are near-unstable.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

    return float(loss.item())


def train(
    model:          NeuralCDE,
    train_loader:   DataLoader,
    val_loader:     DataLoader,
    n_epochs:       int,
    lr:             float,
    weight_decay:   float,
    lr_factor:      float,
    lr_patience:    int,
    es_patience:    int,
    grad_clip:      float,
    device:         torch.device,
    use_sparse_loss: bool = True,
    pen_scaler:     Optional[MinMaxScaler] = None,
    verbose:        bool = True,
) -> Dict[str, list]:
    """
    Full training loop with ReduceLROnPlateau scheduler and early stopping.

    Returns training history dict with keys:
        train_loss, val_loss, lr, patience
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor,
        patience=lr_patience, min_lr=1e-6,
    )
    loss_fn = nn.MSELoss()

    best_val   = float("inf")
    best_state = None
    pat_ctr    = 0
    history    = {"train_loss": [], "val_loss": [], "lr": [], "patience": []}

    for epoch in range(1, n_epochs + 1):
        # ── Training epoch ────────────────────────────────────────────────────
        train_losses = []
        for batch in train_loader:
            # Each batch item is a list when using variable_length_collate,
            # or a stacked tensor when using a fixed pad_to length.
            # Here we handle the fixed-length (stacked) case for simplicity.
            coeffs   = batch["coeffs"].to(device)     # (B, T-1, 28)
            t_norm   = batch["t_norm"][0].to(device)  # (T,) — same for all in batch
            y_dense  = batch["y_dense"].to(device)    # (B, T)
            y_sparse = batch["y_sparse"].to(device)   # (B, T) with NaN

            # Build sparse mask: True where offline lab measurements exist
            sparse_mask = ~torch.isnan(y_sparse) if use_sparse_loss else None

            loss = run_one_batch(
                model=model,
                coeffs=coeffs,
                t_norm=t_norm,
                y_target=y_dense,           # model trains on normalised dense targets
                sparse_mask=sparse_mask,    # but loss is restricted to sparse pts
                loss_fn=loss_fn,
                optimizer=optimizer,
                training=True,
                grad_clip=grad_clip,
            )
            train_losses.append(loss)

        # ── Validation epoch ──────────────────────────────────────────────────
        val_losses = []
        for batch in val_loader:
            coeffs   = batch["coeffs"].to(device)
            t_norm   = batch["t_norm"][0].to(device)
            y_dense  = batch["y_dense"].to(device)
            y_sparse = batch["y_sparse"].to(device)
            sparse_mask = ~torch.isnan(y_sparse) if use_sparse_loss else None

            loss = run_one_batch(
                model=model,
                coeffs=coeffs, t_norm=t_norm,
                y_target=y_dense, sparse_mask=sparse_mask,
                loss_fn=loss_fn, optimizer=None,
                training=False,
            )
            val_losses.append(loss)

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        scheduler.step(val_loss)
        lr_now = float(optimizer.param_groups[0]["lr"])

        improved = val_loss < best_val
        if improved:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_ctr    = 0
        else:
            pat_ctr += 1

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr_now)
        history["patience"].append(pat_ctr)

        if verbose:
            star = "*" if improved else " "
            print(
                f"  {star} Ep {epoch:4d}/{n_epochs}"
                f"  train={train_loss:.5f}  val={val_loss:.5f}"
                f"  lr={lr_now:.2e}  pat={pat_ctr}"
            )

        if pat_ctr >= es_patience:
            print(f"\n  [EarlyStopping] stopped at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"\n  Best weights restored  (val_loss={best_val:.6f})")

    model.eval()
    return history


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Evaluation + plotting
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_batches(
    model:      NeuralCDE,
    batches_df: List[pd.DataFrame],
    Z_list:     List[np.ndarray],
    U_list:     List[np.ndarray],
    pen_scaler: MinMaxScaler,
    device:     torch.device,
    out_dir:    Path,
    label:      str = "eval",
    batch_id_offset: int = 0,
) -> Dict[str, object]:
    """
    Run full-batch inference on a list of batches and compute RMSE.

    Unlike the DataLoader path (which uses a fixed pad_to), this function
    processes each batch at its native length — no truncation.

    For each batch:
      • Build X and coefficients at native length
      • Run model.forward() → ŷ (normalised)
      • Inverse-transform to g/L
      • Compute RMSE at sparse lab-measurement points
      • Save a prediction plot
    """
    model.eval()
    pen_true_all, pen_pred_all = [], []
    per_batch = []

    for i, (batch_df, Z, U) in enumerate(zip(batches_df, Z_list, U_list)):
        T = len(Z)
        t_norm = torch.linspace(0.0, 1.0, T).to(device)

        Z_t = torch.tensor(Z, dtype=torch.float32, device=device)   # (T, 2)
        U_t = torch.tensor(U, dtype=torch.float32, device=device)   # (T, 4)

        # Build path and coeffs with batch dim=1
        X      = build_path(Z_t.unsqueeze(0), U_t.unsqueeze(0), t_norm)  # (1, T, 7)
        coeffs = compute_coeffs(X, t_norm)                                # (1, T-1, 28)

        # Forward pass → (1, T) normalised Penicillin
        y_hat_norm = model(coeffs, t_norm)                    # (1, T)
        y_hat_norm = y_hat_norm.squeeze(0).cpu().numpy()      # (T,)

        # Inverse-transform to g/L
        pen_pred = pen_scaler.inverse_transform(
            y_hat_norm.reshape(-1, 1)
        ).flatten()                                           # (T,)

        # Ground-truth sparse measurements
        pen_sparse = batch_df["_penicillin_sparse"].values    # (T,), NaN between pts
        pen_dense  = batch_df[TARGET_COLS[1]].values          # (T,) dense-interpolated GT

        # Sparse RMSE
        mask = ~np.isnan(pen_sparse) & ~np.isnan(pen_pred)
        if mask.sum() > 0:
            rmse = float(np.sqrt(np.mean((pen_sparse[mask] - pen_pred[mask]) ** 2)))
            pen_true_all.append(pen_sparse[mask])
            pen_pred_all.append(pen_pred[mask])
        else:
            rmse = float("nan")

        global_id = batch_id_offset + i
        per_batch.append({"batch_id": global_id, "rmse_gL": rmse, "n_sparse": int(mask.sum())})

        # ── Plot ───────────────────────────────────────────────────────────────
        time_h = batch_df["Time (h)"].values
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(time_h, pen_dense,  color="silver", lw=1.2, ls="--",
                label="Ground truth (dense interp.)")
        ax.plot(time_h, pen_pred,   color="#1f77b4", lw=1.8,
                label=f"Neural CDE  (RMSE={rmse:.3f} g/L)")
        if mask.sum() > 0:
            ax.scatter(time_h[mask], pen_sparse[mask], color="black", s=35,
                       zorder=4, label="Offline lab meas.")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Penicillin (g/L)")
        ax.set_title(f"{label} — Batch {global_id + 1}  (ID={global_id})")
        ax.legend(fontsize=7)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = out_dir / f"ncde_{label}_batch_{global_id:02d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Batch {global_id + 1}: RMSE={rmse:.4f} g/L  n_sparse={mask.sum()}")

    # Aggregate
    if pen_true_all:
        all_true = np.concatenate(pen_true_all)
        all_pred = np.concatenate(pen_pred_all)
        overall_rmse = float(np.sqrt(np.mean((all_true - all_pred) ** 2)))
        overall_mae  = float(np.mean(np.abs(all_true - all_pred)))
        r2           = float(1 - np.sum((all_true - all_pred)**2) /
                             np.sum((all_true - all_true.mean())**2))
    else:
        overall_rmse = overall_mae = r2 = float("nan")

    return {
        "overall_rmse": overall_rmse,
        "overall_mae":  overall_mae,
        "overall_r2":   r2,
        "batches":      per_batch,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Loss curve plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curve(history: Dict[str, list], out_dir: Path) -> None:
    """Save a 2-panel loss-curve and LR-schedule plot."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

    axes[0].semilogy(history["train_loss"], label="train", color="#1f77b4")
    axes[0].semilogy(history["val_loss"],   label="val",   color="#ff7f0e")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE loss (log scale)")
    axes[0].set_title("Neural CDE — Training curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(history["lr"], color="#2ca02c")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning rate")
    axes[1].set_title("LR schedule (ReduceLROnPlateau)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "ncde_training_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curve saved → {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Neural CDE soft-sensor training")
    p.add_argument("--hidden-dim",   type=int,   default=32,
                   help="CDE hidden state size (default: 32)")
    p.add_argument("--n-layers",     type=int,   default=2,
                   help="CDEFunc MLP depth (default: 2)")
    p.add_argument("--solver",       type=str,   default="rk4",
                   help="ODE solver: rk4 | dopri5 | euler (default: rk4)")
    p.add_argument("--adjoint",      action="store_true", default=True,
                   help="Use adjoint method for backprop (default: True)")
    p.add_argument("--no-adjoint",   action="store_false", dest="adjoint",
                   help="Disable adjoint (faster for short seqs, more memory)")
    p.add_argument("--n-epochs",     type=int,   default=200)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-factor",    type=float, default=0.5)
    p.add_argument("--lr-patience",  type=int,   default=15)
    p.add_argument("--es-patience",  type=int,   default=30)
    p.add_argument("--grad-clip",    type=float, default=5.0)
    p.add_argument("--batch-size",   type=int,   default=4,
                   help="Mini-batch size (keep small for long sequences)")
    p.add_argument("--pad-to",       type=int,   default=1000,
                   help="Truncate sequences to this length for fixed-size batching")
    p.add_argument("--val-frac",     type=float, default=0.2,
                   help="Fraction of R1 batches used for validation")
    p.add_argument("--sparse-loss",  action="store_true",  default=True,
                   help="Compute loss only at sparse lab-measurement points (default)")
    p.add_argument("--dense-loss",   action="store_false", dest="sparse_loss",
                   help="Use dense MSE loss over all time steps")
    p.add_argument("--eval-r1",      action="store_true",  default=False,
                   help="Also evaluate on R1 (train) batches")
    p.add_argument("--out-dir",      type=str,   default=None,
                   help="Output directory (default: outputs/neural_cde)")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)
    print(f"Device: {device}")

    # ── Output directory ──────────────────────────────────────────────────────
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n── Loading data ...")
    r1_batches    = load_batches(CSV_PATH, R1_IDS)
    fault_batches = load_batches(CSV_PATH, FAULT_IDS)

    # ── 2. Encode with frozen β-VAE ───────────────────────────────────────────
    print("\n── Encoding with frozen β=0.05 VAE ...")
    encoder = BetaVAERamanEncoderV2(
        ckpt_path=VAE_CKPT, scaler_path=VAE_SCALER, device=device,
    )
    Z_r1    = encode_latents(r1_batches,    encoder)
    Z_fault = encode_latents(fault_batches, encoder)

    # ── 3. Extract + scale control variables ─────────────────────────────────
    print("\n── Extracting control variables ...")
    U_r1, ctrl_scaler = extract_controls(r1_batches, fit=True)
    U_fault, _        = extract_controls(fault_batches, scaler=ctrl_scaler, fit=False)

    # ── 4. Extract + scale Penicillin targets ─────────────────────────────────
    Y_r1_dense, Y_r1_sparse, pen_scaler = extract_penicillin(r1_batches, fit=True)
    Y_fault_dense, Y_fault_sparse, _    = extract_penicillin(
        fault_batches, scaler=pen_scaler, fit=False,
    )

    # ── 5. Build Datasets with pre-computed spline coefficients ───────────────
    print("\n── Pre-computing cubic spline coefficients ...")
    print(f"   pad_to={args.pad_to} steps  ({args.pad_to * DT:.0f} h)")
    t0 = time.time()
    full_ds = PenicillinCDEDataset(
        Z_list=Z_r1, U_list=U_r1,
        Y_dense_list=Y_r1_dense, Y_sparse_list=Y_r1_sparse,
        pad_to=args.pad_to,
    )
    print(f"   {len(full_ds)} training batches, coeffs computed in {time.time()-t0:.1f}s")

    # Train / validation split
    n_val   = max(1, int(len(full_ds) * args.val_frac))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    print(f"   Train: {n_train}  Val: {n_val}  Batch size: {args.batch_size}")

    # ── 6. Build model ────────────────────────────────────────────────────────
    print("\n── Building NeuralCDE ...")
    # Solver kwargs: for rk4 with adjoint, step_size must go inside 'options'.
    # However, since our t_eval is already a dense grid (pad_to points),
    # torchdiffeq uses the inter-point gaps as natural step size — no need
    # to specify step_size explicitly.
    solver_kws = {} if args.solver == "rk4" else {}

    model = NeuralCDE(
        input_dim=INPUT_DIM,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        adjoint=args.adjoint,
        solver=args.solver,
        solver_kws=solver_kws,
        predict_all_steps=True,
    ).to(device)
    print(model.describe())

    # ── 7. Train ──────────────────────────────────────────────────────────────
    print(f"\n── Training for up to {args.n_epochs} epochs ...")
    print(f"   loss={'sparse-MSE' if args.sparse_loss else 'dense-MSE'}")
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.n_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        es_patience=args.es_patience,
        grad_clip=args.grad_clip,
        device=device,
        use_sparse_loss=args.sparse_loss,
        pen_scaler=pen_scaler,
        verbose=True,
    )
    plot_training_curve(history, out_dir)

    # Save checkpoint
    ckpt_path = out_dir / "ncde_best.pt"
    torch.save({
        "model_state":  model.state_dict(),
        "model_config": {
            "input_dim": INPUT_DIM, "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers, "adjoint": args.adjoint,
            "solver": args.solver, "solver_kws": solver_kws,
        },
        "history":        history,
        "pen_scaler":     pen_scaler,
        "ctrl_scaler":    ctrl_scaler,
        "active_dims":    ACTIVE_DIMS,
        "control_cols":   CONTROL_COLS,
    }, ckpt_path)
    print(f"\n  Checkpoint → {ckpt_path}")

    # ── 8. Evaluate on Fault / R4 batches ─────────────────────────────────────
    print("\n── Evaluating on Fault/R4 batches ...")
    fault_results = evaluate_batches(
        model=model,
        batches_df=fault_batches,
        Z_list=Z_fault,
        U_list=U_fault,
        pen_scaler=pen_scaler,
        device=device,
        out_dir=out_dir,
        label="Fault",
        batch_id_offset=90,
    )
    print(
        f"\n  Fault RMSE : {fault_results['overall_rmse']:.4f} g/L"
        f"  MAE={fault_results['overall_mae']:.4f}"
        f"  R²={fault_results['overall_r2']:.4f}"
    )

    # ── 9. Optionally evaluate on R1 ──────────────────────────────────────────
    r1_results = None
    if args.eval_r1:
        print("\n── Evaluating on R1 (train) batches ...")
        r1_results = evaluate_batches(
            model=model,
            batches_df=r1_batches,
            Z_list=Z_r1,
            U_list=U_r1,
            pen_scaler=pen_scaler,
            device=device,
            out_dir=out_dir,
            label="R1",
            batch_id_offset=0,
        )
        print(f"\n  R1 RMSE : {r1_results['overall_rmse']:.4f} g/L")

    # ── 10. Save summary ──────────────────────────────────────────────────────
    summary = {
        "config": vars(args),
        "architecture": {
            "input_dim":  INPUT_DIM,
            "hidden_dim": args.hidden_dim,
            "n_params":   model.n_params(),
            "solver":     args.solver,
            "adjoint":    args.adjoint,
        },
        "fault_results": fault_results,
        "r1_results":    r1_results,
        "epochs_trained": len(history["train_loss"]),
    }
    summary_path = out_dir / "ncde_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary → {summary_path}")
    print(f"\nDone. Outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
