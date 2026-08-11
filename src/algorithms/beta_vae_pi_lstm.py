"""
src/algorithms/beta_vae_pi_lstm.py
===================================
BetaVAEPILSTMAlgorithm -- PI-LSTM soft-sensor driven by β-VAE Raman latents.

This is a thin subclass of CVAEPILSTMAlgorithm.  All training, evaluation,
and physics-annealing logic is identical; the only difference is the name
token (which selects the β-VAE encoder via the config key
``data.raman_encoder_type: beta_vae``).

Config (configs/beta_vae_pi_lstm.yaml) is identical to cvae_pi_lstm.yaml
except for:
  data.raman_encoder_type : "beta_vae"
  data.raman_ckpt         : checkpoints/beta_vae_best.pt
  data.raman_scaler       : checkpoints/beta_vae_scaler.joblib
"""

from src.algorithms.base import REGISTRY
from src.algorithms.cvae_pi_lstm import CVAEPILSTMAlgorithm


class BetaVAEPILSTMAlgorithm(CVAEPILSTMAlgorithm):
    """Physics-Informed LSTM driven exclusively by β-VAE Raman latents."""

    name = "beta_vae_pi_lstm"


# ── Register ──────────────────────────────────────────────────────────────────
REGISTRY[BetaVAEPILSTMAlgorithm.name] = BetaVAEPILSTMAlgorithm
