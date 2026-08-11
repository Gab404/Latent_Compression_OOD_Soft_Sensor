"""
src/physics/losses.py
=====================
Physics-informed loss functions for the Logi-PINN model.

The physics regulariser enforces two ODEs:
  - Logistic growth (biomass):   ΔX ≈ r · X · (1 − X/X_max) · Δt
  - Luedeking-Piret (penicillin): ΔP ≈ α · ΔX + β · X · Δt

compute_cdae_pinn_loss  — loss for the CDAE-PI-LSTM (Penicillin-only,
simplified mass-balance physics: ΔP ≈ k_prod · r_net · Δt)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pi_lstm import BioreactorLogiPINN


def compute_pinn_loss(
    model:          BioreactorLogiPINN,
    x_batch:        torch.Tensor,   # (B, T, input_size)
    y_batch:        torch.Tensor,   # (B, T, 2)  [bio_norm, pen_norm]
    dt_n:           float,          # normalised time step
    lambda_physics: float = 0.0,
    lambda_r_neg:   float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Full PI-LSTM loss: data MSE + physics residual + growth-rate penalty.

    Parameters
    ----------
    model          : BioreactorLogiPINN
    x_batch        : normalised input sequences  (B, T, D)
    y_batch        : normalised target sequences (B, T, 2)
    dt_n           : normalised dt  (= dt_physical / time_range)
    lambda_physics : weight for physics residual term
    lambda_r_neg   : weight for r ≥ 0 penalty

    Returns
    -------
    loss      : scalar total loss
    loss_dict : breakdown for logging
    """
    pred = model(x_batch)   # (B, T, 2)

    # ── Data MSE (NaN-masked) ─────────────────────────────────────────────────
    valid_bio = ~torch.isnan(y_batch[..., 0])
    valid_pen = ~torch.isnan(y_batch[..., 1])

    l_data_bio = F.mse_loss(pred[..., 0][valid_bio], y_batch[..., 0][valid_bio]) \
                 if valid_bio.any() else pred.new_tensor(0.0)
    l_data_pen = F.mse_loss(pred[..., 1][valid_pen], y_batch[..., 1][valid_pen]) \
                 if valid_pen.any() else pred.new_tensor(0.0)
    l_data = l_data_bio + l_data_pen

    # ── Physics residuals ─────────────────────────────────────────────────────
    if lambda_physics > 0.0:
        r_n    = model.r
        y_max  = model.y_max
        alpha  = model.alpha
        beta   = model.beta

        bio_seq = pred[:, :, 0]   # (B, T)
        pen_seq = pred[:, :, 1]   # (B, T)

        # Consecutive differences
        delta_bio_nn = bio_seq[:, 1:] - bio_seq[:, :-1]   # (B, T-1)
        delta_pen_nn = pen_seq[:, 1:] - pen_seq[:, :-1]   # (B, T-1)

        # ODE predictions
        delta_bio_ode = (r_n * bio_seq[:, :-1]
                         * (1.0 - bio_seq[:, :-1] / y_max)
                         * dt_n)
        delta_pen_ode = alpha * delta_bio_nn + beta * bio_seq[:, :-1] * dt_n

        l_phys_bio = F.mse_loss(delta_bio_nn, delta_bio_ode)
        l_phys_pen = F.mse_loss(delta_pen_nn, delta_pen_ode)
        l_physics  = l_phys_bio + l_phys_pen
    else:
        l_physics  = pred.new_tensor(0.0)
        l_phys_bio = pred.new_tensor(0.0)
        l_phys_pen = pred.new_tensor(0.0)

    # ── Soft penalty: growth rate must be non-negative ────────────────────────
    r_neg_penalty = F.relu(-model.r) * lambda_r_neg

    # ── Total ─────────────────────────────────────────────────────────────────
    loss = l_data + lambda_physics * l_physics + r_neg_penalty

    loss_dict = {
        "loss":       float(loss.item()),
        "loss_data":  float(l_data.item()),
        "loss_phys":  float((lambda_physics * l_physics).item()),
        "loss_bio":   float(l_phys_bio.item()),
        "loss_pen":   float(l_phys_pen.item()),
    }
    return loss, loss_dict


# ── ─────────────────────────────────────────────────────────────────────────
# CDAE-PI-LSTM loss  (Penicillin-only, simplified mass-balance physics)
# ── ─────────────────────────────────────────────────────────────────────────

def compute_cdae_pinn_loss(
    model,                  # CDAEPILSTMModel — provides model.k_prod
    pred:           torch.Tensor,   # (B, T, 2)  [:,:,0]=pen_norm  [:,:,1]=r_net
    y_pen_norm:     torch.Tensor,   # (B, T)     normalised Penicillin targets
    dt_n:           float,          # normalised time step
    lambda_physics: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Physics-Informed loss for the CDAE-PI-LSTM.

    The model outputs TWO values per time step:
      pen_norm  — predicted normalised Penicillin concentration  (Sigmoid, ∈(0,1))
      r_net     — predicted net volumetric production rate       (unbounded)

    Loss components
    ---------------
    Data loss   : MSE(pen_norm_pred, y_pen_norm)          (NaN-masked)
    Physics loss: MSE(ΔP_pred, k_prod · r_net · dt_n)

      The physics loss enforces the simplified penicillin mass balance:
          dP/dt ≈ k_prod · r_net
      where k_prod is a learnable positive scale factor (Softplus-constrained).
      This makes the LSTM self-consistent: the predicted concentration
      trajectory must agree with the predicted rate of change.

    Parameters
    ----------
    model          : CDAEPILSTMModel (duck-typed for k_prod attribute)
    pred           : (B, T, 2) model output
    y_pen_norm     : (B, T) normalised penicillin targets
    dt_n           : normalised time step (dt_physical / time_range)
    lambda_physics : weight for physics residual term

    Returns
    -------
    loss      : scalar total loss
    loss_dict : breakdown dict for logging
    """
    pen_pred = pred[:, :, 0]   # (B, T)  predicted pen_norm
    r_net    = pred[:, :, 1]   # (B, T)  predicted production rate

    # ── Data loss (NaN-masked) ────────────────────────────────────────────────
    valid_pen = ~torch.isnan(y_pen_norm)
    l_data = (
        F.mse_loss(pen_pred[valid_pen], y_pen_norm[valid_pen])
        if valid_pen.any()
        else pred.new_tensor(0.0)
    )

    # ── Physics residual — simplified mass balance ────────────────────────────
    # ΔP_pred (finite difference) must equal k_prod · r_net · dt_n
    # This enforces temporal self-consistency between the two model outputs.
    if lambda_physics > 0.0:
        k_prod   = model.k_prod                                      # scalar ≥ 0
        dP_nn    = pen_pred[:, 1:] - pen_pred[:, :-1]               # (B, T-1)
        dP_phy   = k_prod * r_net[:, :-1] * dt_n                    # (B, T-1)
        l_phys   = F.mse_loss(dP_nn, dP_phy)
    else:
        l_phys   = pred.new_tensor(0.0)

    loss = l_data + lambda_physics * l_phys

    loss_dict = {
        "loss":       float(loss.item()),
        "loss_data":  float(l_data.item()),
        "loss_phys":  float((lambda_physics * l_phys).item()),
        "k_prod":     float(model.k_prod.item()),
    }
    return loss, loss_dict


# ── ─────────────────────────────────────────────────────────────────────────
# Delta-CDAE-PI-LSTM loss  (ΔP prediction, simplified mass-balance physics)
# ── ─────────────────────────────────────────────────────────────────────────

def compute_latent_lp_loss(
    model,                  # BetaVAELatentLPModel — provides model.alpha, model.beta, model.latent_bio_idx
    pred:           torch.Tensor,   # (B, T, 1)  pen_norm predictions
    x_input:        torch.Tensor,   # (B, T, D)  MinMaxScaled β-VAE latent input to the LSTM
    y_pen_norm:     torch.Tensor,   # (B, T)     normalised Penicillin targets
    dt_n:           float,          # normalised time step (dt_physical / time_range)
    lambda_physics: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Latent Luedeking-Piret Physics-Informed loss for β-VAE-PI-LSTM.

    Eliminates the "trivial solution" trap by replacing the unconstrained
    ``r_net`` head with a constraint anchored to the real biomass signal
    already encoded in the β-VAE latent space (z38, index 37).

    ODE (discretised)
    -----------------
        ΔP_phy = (α · ΔX_latent + β · X_latent) · dt_n

    where
        X_latent = x_input[:, :, latent_bio_idx]   (biomass-correlated dim, |r|=0.99)
        ΔX_latent = X_latent[:, 1:] - X_latent[:, :-1]
        ΔP_pred   = pen_norm[:, 1:] - pen_norm[:, :-1]

    Loss components
    ---------------
    Data loss   : MSE(pen_norm_pred, y_pen_norm)          (NaN-masked)
    Physics loss: MSE(ΔP_pred, ΔP_phy)

    Unlike compute_cdae_pinn_loss, there is no free r_net: the ODE rhs is
    entirely determined by the input latent trajectory, making it impossible
    for the network to trivially zero out the residual by outputting a
    matching r_net.

    Parameters
    ----------
    model          : BetaVAELatentLPModel
    pred           : (B, T, 1)  pen_norm predictions
    x_input        : (B, T, D)  MinMaxScaled latent input (same tensor as fed to LSTM)
    y_pen_norm     : (B, T)     normalised penicillin targets
    dt_n           : normalised time step
    lambda_physics : weight for physics residual term

    Returns
    -------
    loss      : scalar total loss
    loss_dict : breakdown dict for logging
    """
    pen_pred = pred[:, :, 0]   # (B, T)

    # ── Data loss (NaN-masked) ────────────────────────────────────────────────
    valid_pen = ~torch.isnan(y_pen_norm)
    l_data = (
        F.mse_loss(pen_pred[valid_pen], y_pen_norm[valid_pen])
        if valid_pen.any()
        else pred.new_tensor(0.0)
    )

    # ── Latent Luedeking-Piret physics residual ───────────────────────────────
    if lambda_physics > 0.0:
        alpha = model.alpha   # scalar > 0
        beta  = model.beta    # scalar > 0

        # Extract biomass-correlated latent dimension (z38 = index 37)
        X_lat = x_input[:, :, model.latent_bio_idx]   # (B, T)

        # Finite differences along time axis
        dP_pred = pen_pred[:, 1:] - pen_pred[:, :-1]  # (B, T-1)  predicted ΔP
        dX_lat  = X_lat[:, 1:]    - X_lat[:, :-1]     # (B, T-1)  latent ΔX

        # Luedeking-Piret ODE prediction in normalised space
        dP_phy = (alpha * dX_lat + beta * X_lat[:, :-1]) * dt_n   # (B, T-1)

        l_phys = F.mse_loss(dP_pred, dP_phy)
    else:
        l_phys = pred.new_tensor(0.0)

    loss = l_data + lambda_physics * l_phys

    loss_dict = {
        "loss":       float(loss.item()),
        "loss_data":  float(l_data.item()),
        "loss_phys":  float((lambda_physics * l_phys).item()),
        "loss_phys_raw": float(l_phys.item()),   # unweighted, for diagnostics
        "alpha":      float(model.alpha.item()),
        "beta":       float(model.beta.item()),
    }
    return loss, loss_dict


def compute_delta_cdae_pinn_loss(
    model,                  # DeltaCDAEPILSTMModel — provides model.k_prod
    pred:           torch.Tensor,   # (B, T, 2)  [:,:,0]=delta_pen_norm  [:,:,1]=r_net
    y_delta_pen_norm: torch.Tensor, # (B, T)     normalised ΔP targets
    dt_n:           float,          # normalised time step
    lambda_physics: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Physics-Informed loss for the Delta-CDAE-PI-LSTM.

    The model outputs TWO values per time step:
      delta_pen_norm — predicted normalised Penicillin increment  (linear, ∈(−∞,+∞))
      r_net          — predicted net volumetric production rate    (unbounded)

    Loss components
    ---------------
    Data loss   : MSE(delta_pen_norm_pred, y_delta_pen_norm)      (NaN-masked)
    Physics loss: MSE(delta_pen_norm_pred, k_prod · r_net · dt_n)

      The physics loss directly enforces the simplified penicillin mass balance:
          ΔP/dt ≈ k_prod · r_net
      Because the model output IS the increment ΔP, no finite-differencing of a
      predicted concentration trajectory is required.  The constraint is tighter
      than in the concentration-prediction variant (compute_cdae_pinn_loss).

    Parameters
    ----------
    model             : DeltaCDAEPILSTMModel (duck-typed for k_prod attribute)
    pred              : (B, T, 2) model output
    y_delta_pen_norm  : (B, T) normalised Penicillin increment targets
    dt_n              : normalised time step (dt_physical / time_range)
    lambda_physics    : weight for physics residual term

    Returns
    -------
    loss      : scalar total loss
    loss_dict : breakdown dict for logging
    """
    delta_pred = pred[:, :, 0]   # (B, T)  predicted ΔP_norm
    r_net      = pred[:, :, 1]   # (B, T)  predicted production rate

    # ── Data loss (NaN-masked) ────────────────────────────────────────────────
    valid_delta = ~torch.isnan(y_delta_pen_norm)
    l_data = (
        F.mse_loss(delta_pred[valid_delta], y_delta_pen_norm[valid_delta])
        if valid_delta.any()
        else pred.new_tensor(0.0)
    )

    # ── Physics residual — direct mass balance on ΔP ─────────────────────────
    # The model directly outputs ΔP_norm; the physics loss enforces:
    #   delta_pred ≈ k_prod · r_net · dt_n  (all T time steps, no shifting)
    if lambda_physics > 0.0:
        k_prod  = model.k_prod                       # scalar ≥ 0
        dP_phy  = k_prod * r_net * dt_n              # (B, T)
        l_phys  = F.mse_loss(delta_pred, dP_phy)
    else:
        l_phys  = pred.new_tensor(0.0)

    loss = l_data + lambda_physics * l_phys

    loss_dict = {
        "loss":       float(loss.item()),
        "loss_data":  float(l_data.item()),
        "loss_phys":  float((lambda_physics * l_phys).item()),
        "k_prod":     float(model.k_prod.item()),
    }
    return loss, loss_dict
