#!/usr/bin/env python
"""
Compute |Pearson r| between z_38 and z_4 (the two active latent dims)
of the β-VAE (β=0.05) on training (fault-free) batches.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from scipy.stats import pearsonr

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.autoencoder.model import BetaVAE_Raman
from src.data.dataset import RAMAN_START_COL_IDX, FAULT_COL

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH   = "./100_Batches_IndPenSim_V3.csv"
CKPT_PATH  = "./checkpoints/beta_vae_best.pt"
SCALER_PATH = "./checkpoints/beta_vae_scaler.joblib"
N_RAMAN    = 2001
SG_WINDOW  = 15
SG_POLY    = 2
SG_DERIV   = 1
DIM_A      = 37   # z_38 (0-indexed)
DIM_B      = 3    # z_4  (0-indexed)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading CSV...")
df = pd.read_csv(CSV_PATH)

# Detect batches and keep only fault-free
t_vals = df["Time (h)"].values
batch_ids = np.zeros(len(df), dtype=np.int32)
bid = 0
for i in range(1, len(df)):
    if t_vals[i] < t_vals[i - 1]:
        bid += 1
    batch_ids[i] = bid

fault_col = df[FAULT_COL].values
faulty_batches = set()
for b in range(bid + 1):
    if fault_col[batch_ids == b].max() > 0:
        faulty_batches.add(b)

fault_free_mask = np.array([batch_ids[i] not in faulty_batches for i in range(len(df))])
print(f"Fault-free rows: {fault_free_mask.sum():,} / {len(df):,}")

# ── Extract and preprocess Raman ──────────────────────────────────────────────
raman_cols = df.columns[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + N_RAMAN]
X_raw = df.loc[fault_free_mask, raman_cols].values.astype(np.float32)

# Valid rows (non-zero energy)
energy = np.abs(X_raw).sum(axis=1)
valid_mask = energy > 0
X_valid = X_raw[valid_mask]
X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)
X_valid = np.clip(X_valid, 0.0, None)

# SG 1st derivative
X_sg = savgol_filter(
    X_valid.astype(np.float64),
    window_length=SG_WINDOW, polyorder=SG_POLY, deriv=SG_DERIV, axis=1
).astype(np.float32)

# StandardScaler
scaler = joblib.load(SCALER_PATH)
X_scaled = scaler.transform(X_sg).astype(np.float32)

# ── Encode ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
model = BetaVAE_Raman(
    input_length=int(state.get("input_length", N_RAMAN)),
    latent_dim=int(state.get("latent_dim", 64)),
)
model.load_state_dict(state["model_state_dict"])
model.eval().to(device)

Z_parts = []
with torch.no_grad():
    for start in range(0, len(X_scaled), 1024):
        chunk = torch.tensor(X_scaled[start:start+1024], device=device)
        mu, _ = model.encode(chunk)
        Z_parts.append(mu.cpu().numpy())
Z = np.concatenate(Z_parts, axis=0)

# ── Compute |Pearson r| between z_38 and z_4 ─────────────────────────────────
r, p_value = pearsonr(Z[:, DIM_A], Z[:, DIM_B])
print(f"|r| between z_{DIM_A+1} and z_{DIM_B+1} = {abs(r):.6f}")
