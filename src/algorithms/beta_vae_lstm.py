"""
src/algorithms/beta_vae_lstm.py
================================
BetaVAELSTMAlgorithm — pure data-driven LSTM driven by β-VAE Raman latents
with physics loss fully disabled (vanilla_mode=true ablation baseline).

This is a thin subclass of BetaVAEPILSTMAlgorithm.  All training, evaluation,
and architecture logic is identical; the only differences are:
  - name token "beta_vae_lstm"
  - config enforces vanilla_mode=true (lambda_physics=0 throughout)

Used as the ablation control for the physics contribution study:
  PI model   → beta_vae_pi_lstm.pt   (lambda_physics > 0)
  Base model → beta_vae_lstm.pt      (lambda_physics = 0, data-only)
"""

from src.algorithms.base import REGISTRY
from src.algorithms.beta_vae_pi_lstm import BetaVAEPILSTMAlgorithm


class BetaVAELSTMAlgorithm(BetaVAEPILSTMAlgorithm):
    """Data-driven LSTM driven by β-VAE Raman latents (no physics loss)."""

    name = "beta_vae_lstm"


# ── Register ──────────────────────────────────────────────────────────────────
REGISTRY[BetaVAELSTMAlgorithm.name] = BetaVAELSTMAlgorithm
