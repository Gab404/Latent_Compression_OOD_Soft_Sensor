"""
src/algorithms/beta_vae_lp_pi_lstm.py
======================================
BetaVAELPPILSTMAlgorithm — PI-LSTM soft-sensor driven by β-VAE Raman latents
with the FULL Logistic + Luedeking-Piret ODE (dual output: biomass + penicillin).

Architecture difference from BetaVAEPILSTMAlgorithm
-----------------------------------------------------
BetaVAEPILSTMAlgorithm  → CDAEPILSTMModel  → simplified mass-balance (pen only)
BetaVAELPPILSTMAlgorithm → BioreactorLogiPINN → full Logistic + LP (bio + pen)

Data flow
---------
  CSV  →  PenicillinDataModule (use_raman=True, encoder=BetaVAERamanEncoderV2)
       →  batch DataFrames  with  _raman_0 … _raman_63  (64D β-VAE latents / mu)
       →  BetaVAELPPILSTMAlgorithm._build_dataset()
               MinMaxScale the 64D latents  → X windows  (seq_len, 64)
               MinMaxScale Biomass + Pen    → y windows  (seq_len, 2)
       →  BioreactorLogiPINN (LSTM → [bio_norm, pen_norm])
       →  compute_pinn_loss  (data MSE + Logistic + LP residuals)

This subclasses PILSTMAlgorithm and overrides only two methods:
  _feature_cols()  — return β-VAE latent column names instead of process features
  _fit_scalers()   — compute dt_n from raw Time (h) column, not from feat scaler
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.algorithms.base import REGISTRY, ScalerBundle
from src.algorithms.pi_lstm import PILSTMAlgorithm
from src.config import Config
from src.data.dataset import TARGET_COLS

import torch


class BetaVAELPPILSTMAlgorithm(PILSTMAlgorithm):
    """
    Physics-Informed LSTM (full Logistic + Luedeking-Piret ODE)
    driven exclusively by β-VAE Raman latents.

    Config sections expected
    ------------------------
    data.use_raman          : true
    data.raman_encoder_type : "beta_vae"
    data.raman_ckpt         : checkpoints/beta_vae_best.pt
    data.raman_scaler       : checkpoints/beta_vae_scaler.joblib
    data.raman_latent_dim   : 64
    model.hidden_size       : 64
    model.num_lstm_layers   : 1
    model.fc_hidden         : 32
    model.lstm_dropout      : 0.0
    model.r_n_init          : 45.0
    model.y_max_n_init      : 1.05
    model.alpha_n_init      : 1.0
    model.beta_n_init       : 1.0
    training.*              : same as pi_lstm.yaml
    """

    name = "beta_vae_lp_pi_lstm"

    # ── Override: use β-VAE latent columns instead of process features ────────

    def _feature_cols(self) -> List[str]:
        n_raman = int(self.cfg.data.get("raman_latent_dim", 64))
        return [f"_raman_{i}" for i in range(n_raman)]

    # ── Override: compute dt_n from Time (h) column, not from feat scaler ────

    def _fit_scalers(self, train_batches) -> ScalerBundle:
        """
        Fit MinMaxScalers for β-VAE latent features and dual targets.

        dt_n is computed from the raw Time (h) column (not from the latent
        feature scaler, since latent dims have no time interpretation).
        """
        feat_cols = self._feature_cols()

        X_all   = np.vstack([b[feat_cols].values for b in train_batches])
        bio_all = np.concatenate(
            [b[TARGET_COLS[0]].values for b in train_batches]
        ).reshape(-1, 1)
        pen_all = np.concatenate(
            [b[TARGET_COLS[1]].values for b in train_batches]
        ).reshape(-1, 1)

        feat_scaler = MinMaxScaler().fit(X_all)
        bio_scaler  = MinMaxScaler().fit(bio_all)
        pen_scaler  = MinMaxScaler().fit(pen_all)

        # Normalised time step: dt_n = dt_physical / time_range
        t_all    = np.concatenate([b["Time (h)"].values for b in train_batches])
        t_range  = float(t_all.max() - t_all.min())
        self._dt_n = float(self.cfg.training.dt) / max(t_range, 1e-8)

        print(
            f"  Scalers fitted  |  "
            f"latent_dim={X_all.shape[1]}  "
            f"n_train_rows={X_all.shape[0]:,}  "
            f"dt_n={self._dt_n:.6f}"
        )
        return ScalerBundle(feat=feat_scaler, bio=bio_scaler, pen=pen_scaler)


# ── Register ──────────────────────────────────────────────────────────────────
REGISTRY[BetaVAELPPILSTMAlgorithm.name] = BetaVAELPPILSTMAlgorithm
