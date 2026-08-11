"""
src/models/beta_vae_liquid.py
==============================
BetaVAELiquidModel — Liquid Neural Network (CfC) temporal core for Penicillin
soft-sensing from β-VAE Raman latents.

Architecture
------------
  Input   : (B, T, 64)  — MinMaxScaled β-VAE posterior means
  CfC     : Closed-form Continuous-time RNN (ncps.torch.CfC)
              FullyConnected wiring, optional backbone MLP inside the liquid cell
              batch_first=True, returns full sequence
  FC head : LayerNorm → Linear(cfc_units, fc_hidden) → LeakyReLU(0.1)
            → Linear(fc_hidden, 1) → Sigmoid
  Output  : (B, T, 1)  — pen_norm ∈ (0, 1)

Why CfC instead of LSTM / Mamba
---------------------------------
CfC (Hasani et al., 2022) is the closed-form solution to Liquid Time-Constant
(LTC) networks, a class of continuous-time ODEs inspired by C. elegans neural
circuits.  Unlike LSTM (discrete gates) or Mamba (input-dependent SSM), CfC
solves the hidden-state ODE analytically at each step, giving it:

  1. Continuous-time inductive bias  — naturally suited to irregularly-sampled
     bioprocess data where dt is not truly uniform.
  2. Strong small-data generalisation — MIT work shows CfC outperforms LSTM
     and Transformers on time-series benchmarks with < 1000 training examples.
  3. Expressivity without depth  — the LTC ODE packs nonlinearity into the
     state update itself, so a single CfC layer with a backbone MLP rivals
     2–3 LSTM layers.

No physics loss is applied here — this is a pure data-driven baseline
positioned opposite Mamba in the temporal-architecture ablation study:

  β-VAE-LSTM     ← standard gated RNN baseline
  β-VAE-Mamba    ← input-dependent SSM (scale-heavy, high-capacity)
  β-VAE-LiquidNet← continuous-time ODE-RNN (small-data champion) ← this file

Requires: ncps >= 1.0.0
Install:  pip install ncps
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ncps.torch import CfC


# ── Main model ────────────────────────────────────────────────────────────────

class BetaVAELiquidModel(nn.Module):
    """
    β-VAE LiquidNet soft-sensor (CfC temporal core).

    Parameters
    ----------
    input_size        : β-VAE latent dimension (default 64)
    cfc_units         : CfC hidden/state size (default 64)
    mode              : CfC ODE mode — "default" (gated), "pure" (no gate),
                        "no_gate" (ablated). "default" recommended.
    backbone_units    : hidden size of the backbone MLP *inside* the CfC cell
                        (projects input before the liquid dynamics). 0 = no backbone.
    backbone_layers   : number of backbone MLP layers (default 1)
    backbone_dropout  : dropout in backbone MLP (default 0.0)
    fc_hidden         : output FC head hidden size (default 32)
    """

    def __init__(
        self,
        input_size:       int   = 64,
        cfc_units:        int   = 64,
        mode:             str   = "default",
        backbone_units:   int   = 128,
        backbone_layers:  int   = 1,
        backbone_dropout: float = 0.0,
        fc_hidden:        int   = 32,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.cfc_units  = cfc_units

        # ── CfC liquid core ───────────────────────────────────────────────────
        # Passing units as an integer lets ncps create a FullyConnected wiring
        # internally (every unit connects to every other — same connectivity
        # budget as LSTM / Mamba) while also enabling the backbone MLP option.
        # Note: passing an explicit FullyConnected *object* disables backbone.
        cfc_kwargs: dict = dict(
            input_size=input_size,
            units=cfc_units,        # int → internal FullyConnected wiring
            batch_first=True,       # (B, T, D) convention throughout project
            mode=mode,
        )
        if backbone_layers > 0 and backbone_units > 0:
            cfc_kwargs.update(
                backbone_units=backbone_units,
                backbone_layers=backbone_layers,
                backbone_dropout=backbone_dropout,
            )

        self.cfc = CfC(**cfc_kwargs)

        # ── Output norm + FC head ─────────────────────────────────────────────
        self.out_norm = nn.LayerNorm(cfc_units)
        self.fc_head  = nn.Sequential(
            nn.Linear(cfc_units, fc_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(fc_hidden, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, input_size)  — MinMaxScaled β-VAE latent vectors

        Returns
        -------
        pen_norm : (B, T, 1)  — normalised Penicillin prediction ∈ (0, 1)
        """
        h, _ = self.cfc(x)          # (B, T, cfc_units), discard final hidden
        h    = self.out_norm(h)      # stabilise before head
        return self.fc_head(h)       # (B, T, 1)

    # ── Parameter groups (no physics params — single group optimizer) ─────────

    def physics_parameters(self) -> List[nn.Parameter]:
        return []

    def network_parameters(self) -> List[nn.Parameter]:
        return list(self.parameters())

    # ── Weight init ───────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
