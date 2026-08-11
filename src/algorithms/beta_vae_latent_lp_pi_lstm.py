"""
src/algorithms/beta_vae_latent_lp_pi_lstm.py
=============================================
BetaVAELatentLPPILSTMAlgorithm — PI-LSTM soft-sensor driven by β-VAE Raman
latents with the Latent Luedeking-Piret physics loss.

Key differences from BetaVAEPILSTMAlgorithm
--------------------------------------------
1. Model: BetaVAELatentLPModel  (single pen head, no r_net, params α + β)
2. Loss:  compute_latent_lp_loss (real LP ODE anchored to z38, not a free proxy)
3. _train_epoch passes x_b to the loss so it can extract X_latent = x_b[:,:,37]
4. Training log reports alpha and beta evolution instead of k_prod

Inherited unchanged from CVAEPILSTMAlgorithm
--------------------------------------------
- _feature_cols()    → [_raman_0 … _raman_63]
- _fit_scalers()     → MinMaxScaler on latents + targets, dt_n from Time (h)
- _build_dataset()   → sliding-window TensorDataset (X latents, y pen_norm)
- _val_epoch()       → MSE on pen_norm  (works for (B,T,1) output)
- evaluate()         → sparse-point inference on pen  (uses [:,-1,0] slice)
- _make_optimizer()  → two AdamW groups (network lr, physics lr)
- _is_es_active()    → freeze during delay phase
- _epoch_extra_kwargs() → lambda annealing schedule
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.algorithms.base import REGISTRY
from src.algorithms.cvae_pi_lstm import CVAEPILSTMAlgorithm
from src.models.beta_vae_latent_lp import BetaVAELatentLPModel
from src.physics.losses import compute_latent_lp_loss


class BetaVAELatentLPPILSTMAlgorithm(CVAEPILSTMAlgorithm):
    """
    Physics-Informed LSTM with Latent Luedeking-Piret ODE,
    driven exclusively by β-VAE Raman latents.

    Config sections expected
    ------------------------
    data.use_raman          : true
    data.raman_encoder_type : "beta_vae"
    data.raman_ckpt         : checkpoints/beta_vae_best.pt
    data.raman_scaler       : checkpoints/beta_vae_scaler.joblib
    data.raman_latent_dim   : 64
    model.hidden_size       : 64
    model.num_lstm_layers   : 2
    model.fc_hidden         : 32
    model.lstm_dropout      : 0.2
    model.latent_bio_idx    : 37      # z38 (0-indexed), biomass dim
    model.alpha_init        : 1.0     # Luedeking-Piret α initial value
    model.beta_init         : 1.0     # Luedeking-Piret β initial value
    training.*              : same as beta_vae_pi_lstm.yaml
    """

    name = "beta_vae_latent_lp_pi_lstm"

    # ── Override: use BetaVAELatentLPModel instead of CDAEPILSTMModel ─────────

    def _build_model(self) -> BetaVAELatentLPModel:
        m      = self.cfg.model
        n_feat = len(self._feature_cols())
        return BetaVAELatentLPModel(
            input_size=n_feat,
            hidden_size=int(m.hidden_size),
            num_lstm_layers=int(m.num_lstm_layers),
            fc_hidden=int(m.fc_hidden),
            lstm_dropout=float(m.lstm_dropout),
            latent_bio_idx=int(m.get("latent_bio_idx", 37)),
            alpha_init=float(m.get("alpha_init", 1.0)),
            beta_init=float(m.get("beta_init",  1.0)),
        )

    # ── Override: pass x_b to the Latent LP loss ──────────────────────────────

    def _train_epoch(
        self,
        loader: DataLoader,
        lambda_physics: float = 0.0,
    ) -> Dict[str, Any]:
        """One training epoch using the Latent Luedeking-Piret physics loss."""
        grad_clip = float(self.cfg.training.grad_clip)
        ep_total, ep_data, ep_phys = [], [], []
        ep_alpha, ep_beta, ep_phys_raw = [], [], []

        for x_b, y_b in loader:
            x_b = x_b.to(self.device)
            y_b = y_b.to(self.device)

            self._optimizer.zero_grad(set_to_none=True)

            pred = self.model(x_b)   # (B, T, 1): pen_norm only

            loss, ld = compute_latent_lp_loss(
                model=self.model,
                pred=pred,
                x_input=x_b,         # loss extracts x_b[:,:,latent_bio_idx]
                y_pen_norm=y_b,
                dt_n=self._dt_n,
                lambda_physics=lambda_physics,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self._optimizer.step()

            ep_total.append(ld["loss"])
            ep_data.append(ld["loss_data"])
            ep_phys.append(ld["loss_phys"])
            ep_phys_raw.append(ld["loss_phys_raw"])
            ep_alpha.append(ld["alpha"])
            ep_beta.append(ld["beta"])

        return {
            "train_loss":     float(np.mean(ep_total)),
            "train_data":     float(np.mean(ep_data)),
            "train_phys":     float(np.mean(ep_phys)),      # λ × L_phys (weighted)
            "phys_raw":       float(np.mean(ep_phys_raw)),  # L_phys unweighted
            "alpha":          float(np.mean(ep_alpha)),
            "beta":           float(np.mean(ep_beta)),
            "lambda_phys":    float(lambda_physics),        # excluded from display
        }


# ── Register ──────────────────────────────────────────────────────────────────
REGISTRY[BetaVAELatentLPPILSTMAlgorithm.name] = BetaVAELatentLPPILSTMAlgorithm
