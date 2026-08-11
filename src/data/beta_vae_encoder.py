"""
src/data/beta_vae_encoder.py
=============================
BetaVAERamanEncoderV2 -- Frozen β-VAE Raman feature extractor.

Drop-in replacement for CVAERamanEncoderV2, but loads BetaVAE_Raman instead
of CVAE_Raman.  The preprocessing pipeline is identical:
    raw Raman (2001D) -> SG(d=1,w=15,p=2) -> StandardScaler -> BetaVAE encoder (mu)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
from scipy.signal import savgol_filter

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.autoencoder.model import BetaVAE_Raman
from src.data.dataset import RAMAN_START_COL_IDX, RAMAN_LATENT_DIM


class BetaVAERamanEncoderV2:
    """Frozen β-VAE Raman feature extractor (identical API to CVAERamanEncoderV2)."""

    def __init__(
        self,
        ckpt_path:    str,
        scaler_path:  str,
        device:       torch.device,
        n_raman_cols: int = 2001,
        latent_dim:   int = RAMAN_LATENT_DIM,
        encode_batch: int = 1024,
        sg_window:    int = 15,
        sg_poly:      int = 2,
        sg_deriv:     int = 1,
    ) -> None:
        self.device       = device
        self.n_raman_cols = n_raman_cols
        self.latent_dim   = latent_dim
        self.encode_batch = encode_batch
        self.sg_window    = sg_window
        self.sg_poly      = sg_poly
        self.sg_deriv     = sg_deriv

        state             = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        input_length      = int(state.get("input_length", n_raman_cols))
        loaded_latent_dim = int(state.get("latent_dim",   latent_dim))

        self._model = BetaVAE_Raman(
            input_length=input_length,
            latent_dim=loaded_latent_dim,
        ).to(device)
        self._model.load_state_dict(state["model_state_dict"])
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._model.eval()
        self.latent_dim = loaded_latent_dim

        self._scaler = joblib.load(scaler_path)

        print(
            f"  [BetaVAERamanEncoderV2] Loaded from {Path(ckpt_path).name} | "
            f"input={input_length}  latent={self.latent_dim}  "
            f"SG(w={sg_window},p={sg_poly},d={sg_deriv})"
        )

    def _apply_sg(self, X: np.ndarray) -> np.ndarray:
        return savgol_filter(
            X.astype(np.float64),
            window_length=self.sg_window,
            polyorder=self.sg_poly,
            deriv=self.sg_deriv,
            axis=1,
        ).astype(np.float32)

    @torch.no_grad()
    def encode(self, spectra: np.ndarray) -> np.ndarray:
        if len(spectra) == 0:
            return np.zeros((0, self.latent_dim), dtype=np.float32)
        out_parts = []
        for start in range(0, len(spectra), self.encode_batch):
            chunk = torch.tensor(
                spectra[start : start + self.encode_batch],
                dtype=torch.float32,
                device=self.device,
            )
            mu, _ = self._model.encode(chunk)
            out_parts.append(mu.cpu().numpy())
        return np.concatenate(out_parts, axis=0)

    @torch.no_grad()
    def encode_dataframe(self, df) -> np.ndarray:
        N       = len(df)
        latents = np.zeros((N, self.latent_dim), dtype=np.float32)

        all_cols   = df.columns.tolist()
        n_use      = min(self.n_raman_cols, len(all_cols) - RAMAN_START_COL_IDX)
        raman_cols = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + n_use]
        X_raw      = df[raman_cols].values.astype(np.float32)

        energy_mask = np.abs(X_raw).sum(axis=1) > 0
        n_valid     = int(energy_mask.sum())
        if n_valid == 0:
            return latents

        print(f"  Raman (β-VAE V2): encoding {n_valid:,} / {N:,} rows ...")

        X_valid  = np.clip(
            np.nan_to_num(X_raw[energy_mask].copy(), nan=0.0, posinf=0.0, neginf=0.0),
            0.0, None,
        )
        X_sg     = self._apply_sg(X_valid)
        X_scaled = self._scaler.transform(X_sg).astype(np.float32)
        latents[energy_mask] = self.encode(X_scaled)
        return latents

    @property
    def model(self) -> BetaVAE_Raman:
        return self._model
