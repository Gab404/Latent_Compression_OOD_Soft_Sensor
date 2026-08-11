"""
src/models/beta_vae_latent_lp.py
=================================
BetaVAELatentLPModel — Physics-Informed LSTM for Penicillin prediction
using the Latent Luedeking-Piret ODE.

Motivation
----------
The classic "trivial solution" trap in the CDAE-PI-LSTM is avoided here by
eliminating the unconstrained ``r_net`` head entirely.  Instead of letting
the network invent a free rate proxy, we anchor the physics constraint to
a *real* physical state that is already encoded in the β-VAE latent space.

The Latent Correlation Heatmap demonstrated that β-VAE latent dimension z38
(0-indexed: dim 37) has Pearson |r| = 0.99 with offline Biomass measurements.
This makes it a near-perfect surrogate for X in the Luedeking-Piret ODE:

    dP/dt = α · dX/dt + β · X

Discretised for a sliding window of normalised features:

    ΔP_phy = (α · ΔX_latent + β · X_latent) · dt_n

where X_latent = x_input[:, :, latent_bio_idx] (index 37 by default).

Architecture
------------
  Input  : (B, T, 64)  — MinMaxScaled β-VAE latent vectors
  LSTM   : 64 → hidden_size
  Shared : LayerNorm → Linear → LeakyReLU
  Head   : Linear(fc_hidden, 1) → Sigmoid  →  pen_norm ∈ (0, 1)
  Output : (B, T, 1)  — pen_norm only  (no r_net head)

Physics parameters (both positive via Softplus, learnable)
----------------------------------------------------------
  α  (alpha) : growth-associated LP coefficient   (dP proportional to dX)
  β  (beta)  : maintenance LP coefficient          (dP proportional to X)

These are optimised at a higher learning rate (1e-2 vs 1e-3 for network).
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaVAELatentLPModel(nn.Module):
    """
    β-VAE Latent Luedeking-Piret Physics-Informed LSTM.

    Parameters
    ----------
    input_size      : β-VAE latent dimension (default 64)
    hidden_size     : LSTM hidden state size
    num_lstm_layers : stacked LSTM layers
    fc_hidden       : shared FC intermediate size
    lstm_dropout    : dropout between LSTM layers (only if layers > 1)
    latent_bio_idx  : index of the biomass-correlated latent dim (default 37 = z38)
    alpha_init      : initial value of Luedeking-Piret α (growth-associated)
    beta_init       : initial value of Luedeking-Piret β (maintenance)
    """

    def __init__(
        self,
        input_size:      int   = 64,
        hidden_size:     int   = 64,
        num_lstm_layers: int   = 2,
        fc_hidden:       int   = 32,
        lstm_dropout:    float = 0.0,
        latent_bio_idx:  int   = 37,      # z38 (0-indexed)
        alpha_init:      float = 1.0,
        beta_init:       float = 1.0,
    ) -> None:
        super().__init__()

        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.latent_bio_idx = latent_bio_idx

        # ── Temporal backbone ─────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0.0,
        )

        # ── Shared projection ─────────────────────────────────────────────────
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, fc_hidden),
            nn.LeakyReLU(0.1),
        )

        # ── Single output head: pen_norm only ─────────────────────────────────
        # Sigmoid keeps output in (0, 1), consistent with MinMaxScaler range.
        # The r_net head is deliberately absent — the ODE is now grounded to
        # the real biomass signal encoded in z38 rather than a free proxy.
        self.head_pen = nn.Sequential(nn.Linear(fc_hidden, 1), nn.Sigmoid())

        # ── Latent Luedeking-Piret physics parameters ─────────────────────────
        # Both must be strictly positive (Softplus-constrained):
        #   α:  growth-associated coefficient  — scales ΔX contribution
        #   β:  maintenance coefficient        — scales X level contribution
        # Stored in unconstrained space; constrained via Softplus at access time.
        self._alpha_raw = nn.Parameter(
            torch.tensor(_softplus_inv(alpha_init), dtype=torch.float32)
        )
        self._beta_raw = nn.Parameter(
            torch.tensor(_softplus_inv(beta_init), dtype=torch.float32)
        )

        self._init_weights()

    # ── Constrained physics parameters ───────────────────────────────────────

    @property
    def alpha(self) -> torch.Tensor:
        """Growth-associated LP coefficient α = Softplus(_alpha_raw) > 0."""
        return F.softplus(self._alpha_raw)

    @property
    def beta(self) -> torch.Tensor:
        """Maintenance LP coefficient β = Softplus(_beta_raw) > 0."""
        return F.softplus(self._beta_raw)

    def get_physics_params(self) -> Dict[str, float]:
        return {
            "alpha": float(self.alpha.item()),
            "beta":  float(self.beta.item()),
        }

    # ── Parameter groups ─────────────────────────────────────────────────────

    def physics_parameters(self) -> List[nn.Parameter]:
        """α and β — optimised at a higher learning rate."""
        return [self._alpha_raw, self._beta_raw]

    def network_parameters(self) -> List[nn.Parameter]:
        """All parameters except physics parameters."""
        physics_ids = {id(p) for p in self.physics_parameters()}
        return [p for p in self.parameters() if id(p) not in physics_ids]

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, input_size)  — MinMaxScaled β-VAE latent vectors

        Returns
        -------
        pen_norm : (B, T, 1)  — normalised Penicillin prediction ∈ (0, 1)

        Note: x[:, :, latent_bio_idx] is used externally in the physics loss
        to extract the biomass-correlated latent signal (z38).
        """
        h, _   = self.lstm(x)       # (B, T, hidden_size)
        z      = self.shared(h)     # (B, T, fc_hidden)
        pen    = self.head_pen(z)   # (B, T, 1)
        return pen

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        for module in [*list(self.shared), *list(self.head_pen)]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)


# ── Helper ────────────────────────────────────────────────────────────────────

def _softplus_inv(y: float, beta: float = 1.0) -> float:
    """Inverse of Softplus: x such that softplus(x) = y."""
    return math.log(math.expm1(beta * y)) / beta
