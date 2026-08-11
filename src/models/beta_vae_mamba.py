"""
src/models/beta_vae_mamba.py
=============================
BetaVAEMambaModel — Mamba-SSM temporal core for Penicillin soft-sensing
from β-VAE Raman latents.

Architecture
------------
  Input   : (B, T, 64)  — MinMaxScaled β-VAE posterior means
  Mamba   : num_mamba_layers × [Mamba block + residual + LayerNorm]
              Each Mamba block: d_model=64, d_state, d_conv, expand
  FC head : LayerNorm → Linear(d_model, fc_hidden) → LeakyReLU(0.1)
            → Linear(fc_hidden, 1) → Sigmoid
  Output  : (B, T, 1)  — pen_norm ∈ (0, 1)

Why Mamba instead of LSTM
--------------------------
Mamba is a selective State-Space Model (SSM) that scales linearly in
sequence length and uses a hardware-aware parallel scan.  For our
seq_len=24 windows this is not about speed, but about the recurrent
inductive bias: Mamba learns *input-dependent* gating of long-range
context, which may better capture the slow penicillin accumulation
dynamics than an LSTM whose gates are fixed after training.

No physics loss is applied here — the model is a pure data-driven
baseline and is compared against BetaVAEPILSTM* variants.

Requires: mamba-ssm >= 2.0, causal-conv1d >= 1.4
Install (pre-built wheels, Python 3.10, PyTorch 2.4, CUDA 12):
  wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
  wget https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
  pip install ninja einops "transformers==4.40.0"
  pip install causal_conv1d-*.whl
  pip install mamba_ssm-*.whl
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Mamba block with residual + pre-norm (standard Mamba layer design) ────────

class MambaBlock(nn.Module):
    """
    Single Mamba layer with pre-LayerNorm and residual connection.

    pre-norm + residual follows the original Mamba paper design:
        y = x + Mamba(LayerNorm(x))
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        from mamba_ssm import Mamba
        self.norm  = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        return x + self.mamba(self.norm(x))


# ── Main model ────────────────────────────────────────────────────────────────

class BetaVAEMambaModel(nn.Module):
    """
    β-VAE Mamba soft-sensor.

    Parameters
    ----------
    input_size       : β-VAE latent dimension (default 64)
    d_model          : Mamba model dimension — also the internal hidden dim.
                       If input_size != d_model an input projection is added.
    num_mamba_layers : number of stacked Mamba blocks (default 2)
    d_state          : SSM state expansion factor (default 16)
    d_conv           : local convolution width in Mamba (default 4)
    expand           : block expansion factor in Mamba (default 2)
    fc_hidden        : FC head hidden size (default 32)
    """

    def __init__(
        self,
        input_size:       int = 64,
        d_model:          int = 64,
        num_mamba_layers: int = 2,
        d_state:          int = 16,
        d_conv:           int = 4,
        expand:           int = 2,
        fc_hidden:        int = 32,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.d_model    = d_model

        # ── Optional input projection ─────────────────────────────────────────
        # If latent_dim != d_model, project to d_model first.
        if input_size != d_model:
            self.input_proj = nn.Linear(input_size, d_model)
        else:
            self.input_proj = nn.Identity()

        # ── Mamba stack ───────────────────────────────────────────────────────
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state,
                       d_conv=d_conv, expand=expand)
            for _ in range(num_mamba_layers)
        ])

        # ── Output norm + FC head ─────────────────────────────────────────────
        self.out_norm = nn.LayerNorm(d_model)
        self.fc_head  = nn.Sequential(
            nn.Linear(d_model, fc_hidden),
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
        h = self.input_proj(x)            # (B, T, d_model)
        for block in self.mamba_layers:
            h = block(h)                   # (B, T, d_model)
        h = self.out_norm(h)               # final norm
        return self.fc_head(h)             # (B, T, 1)

    # ── Parameter groups (no physics params — single group optimizer) ─────────

    def physics_parameters(self) -> List[nn.Parameter]:
        return []   # no physics params

    def network_parameters(self) -> List[nn.Parameter]:
        return list(self.parameters())

    # ── Weight init ───────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
