"""
src/algorithms/beta_vae_mamba.py
=================================
BetaVAEMambaAlgorithm — data-driven Mamba soft-sensor driven by β-VAE latents.

Inherits the full data pipeline from CVAEPILSTMAlgorithm (feature columns,
scaler fitting, sliding-window dataset, sparse-point evaluation) and
overrides only the model and training step:
  - Model : BetaVAEMambaModel  (Mamba stack, single pen_norm head)
  - Loss  : pure MSE on pen_norm  (no physics term)
  - Optim : single AdamW group  (no physics param group)
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.algorithms.base import REGISTRY
from src.algorithms.cvae_pi_lstm import CVAEPILSTMAlgorithm
from src.models.beta_vae_mamba import BetaVAEMambaModel


class BetaVAEMambaAlgorithm(CVAEPILSTMAlgorithm):
    """
    Mamba-based soft-sensor driven by β-VAE Raman latents (no physics loss).

    Config sections expected
    ------------------------
    data.use_raman          : true
    data.raman_encoder_type : "beta_vae"
    data.raman_ckpt         : checkpoints/beta_vae_best.pt
    data.raman_scaler       : checkpoints/beta_vae_scaler.joblib
    data.raman_latent_dim   : 64
    model.d_model           : 64
    model.num_mamba_layers  : 2
    model.d_state           : 16
    model.d_conv            : 4
    model.expand            : 2
    model.fc_hidden         : 32
    training.*              : standard (vanilla_mode=true, lambda=0)
    """

    name = "beta_vae_mamba"

    # ── Override: build Mamba model instead of CDAEPILSTMModel ───────────────

    def _build_model(self) -> BetaVAEMambaModel:
        m      = self.cfg.model
        n_feat = len(self._feature_cols())
        return BetaVAEMambaModel(
            input_size=n_feat,
            d_model=int(m.get("d_model", n_feat)),
            num_mamba_layers=int(m.get("num_mamba_layers", 2)),
            d_state=int(m.get("d_state", 16)),
            d_conv=int(m.get("d_conv",  4)),
            expand=int(m.get("expand",  2)),
            fc_hidden=int(m.get("fc_hidden", 32)),
        )

    # ── Override: pure MSE, single output head, no physics ───────────────────

    def _train_epoch(
        self,
        loader: DataLoader,
        **kwargs,           # absorbs lambda_physics etc. from base loop
    ) -> Dict[str, Any]:
        grad_clip = float(self.cfg.training.grad_clip)
        losses = []

        for x_b, y_b in loader:
            x_b = x_b.to(self.device)
            y_b = y_b.to(self.device)

            self._optimizer.zero_grad(set_to_none=True)

            pred     = self.model(x_b)          # (B, T, 1)
            pen_pred = pred[:, :, 0]             # (B, T)
            loss     = F.mse_loss(pen_pred, y_b)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self._optimizer.step()

            losses.append(loss.item())

        mean_loss = float(np.mean(losses))
        return {
            "train_loss": mean_loss,
            "train_data": mean_loss,
            "train_phys": 0.0,
        }

    # ── Override: single AdamW group (no physics params) ─────────────────────

    def _make_optimizer(self) -> torch.optim.Optimizer:
        tr = self.cfg.training
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=float(tr.lr),
            weight_decay=float(tr.weight_decay),
        )

    # ── Override: ES always active (no delay phase) ───────────────────────────

    def _is_es_active(self, epoch: int) -> bool:
        return True

    def _epoch_extra_kwargs(self, epoch: int) -> Dict[str, Any]:
        return {}


# ── Register ──────────────────────────────────────────────────────────────────
REGISTRY[BetaVAEMambaAlgorithm.name] = BetaVAEMambaAlgorithm
