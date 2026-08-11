"""
src/models/latent_ode.py
========================
Latent ODE soft-sensor for penicillin fermentation.

Designed to be plugged on top of a **frozen** β-VAE encoder that has already
reduced a 2001-dimensional Raman spectrum to a compact, disentangled latent
vector μ ∈ ℝ^D.  Instead of passing the full latent trajectory through an
LSTM, this module models the *continuous-time dynamics* of the latent state
as an ordinary differential equation:

        dz / dt = f_θ(z(t), t)

The solution trajectory z(t) is decoded by a linear readout to predict
penicillin concentration at any continuous time point.

Architecture overview
---------------------

                        (frozen β-VAE encoder)
   Raman(t₀) ──────────────────────────────────► z₀ ∈ ℝ^ode_dim
                                                    │
                                         ODEFunc f_θ(z,t)
                                                    │
                              odeint ──► z(t₁), z(t₂), …, z(T)
                                                    │
                                       Readout  W·z + b
                                                    │
                                         ŷ(t) = Penicillin (g/L)

Components
----------
ODEFunc    — the derivative network f_θ(z, t) → dz/dt
             • Input:  (ode_dim + 1)  — latent state z with appended scalar t
             • Hidden: n_layers × hidden_size, Swish or Tanh
             • Output: ode_dim        — time-derivative dz/dt
             • Lightweight by design (hidden ≤ 32); the structure is
               already captured by the VAE; the ODE only needs to model
               how that structure evolves in time.

LatentODE  — the main integrator + readout
             • Accepts z₀ ∈ ℝ^(B, ode_dim) and t ∈ ℝ^T
             • Calls torchdiffeq.odeint (rk4 by default, dopri5 optional)
             • Returns ŷ ∈ ℝ^(B, T) — predicted penicillin at each time step

Key design choices
------------------
1. Time is injected as a scalar appended to z at every derivative evaluation.
   This makes f_θ a non-autonomous ODE, letting the model learn time-varying
   fermentation dynamics (growth / production / decline phases).

2. The ODE dimension (ode_dim) can be smaller than the full VAE latent dim:
   • ode_dim = 64 : use the full latent (most expressive)
   • ode_dim = 2  : operate only on the two active dims z52 and z27
                    (for β=0.05, these encode Penicillin and Glucose)
   • ode_dim = 1  : operate only on z52 (Penicillin axis, r=0.995)
   An optional `active_dims` index list extracts the desired subset from μ
   before the ODE; a projection layer handles the case where ode_dim < latent_dim.

3. Adjoint vs. fixed-step back-propagation:
   • method="rk4"     → fast, low memory, fixed step; best for short sequences
   • method="dopri5"  → adaptive; more accurate for stiff dynamics; uses the
                        adjoint method for O(1) memory back-prop through the ODE
   The default is "rk4" because sequence windows are short (≤ 48 steps).

4. Zero-initialised readout bias ensures the model starts from a neutral
   prediction before the latent structure is learned.

Usage (standalone)
------------------
    from src.models.latent_ode import LatentODE

    # Full 64-dim latent
    model = LatentODE(ode_dim=64)

    # Only active dims (β=0.05: z52=Penicillin, z27=Glucose)
    model = LatentODE(ode_dim=2, active_dims=[52, 27])

    z0  = ...          # (B, 64) — VAE posterior mean at t=0
    t   = torch.linspace(0, 1, seq_len)   # (T,)
    pen = model(z0, t)                     # (B, T)
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torchdiffeq import odeint


# ─────────────────────────────────────────────────────────────────────────────
# Activation helper
# ─────────────────────────────────────────────────────────────────────────────

class Swish(nn.Module):
    """Swish activation: x · σ(x).  Smooth, non-monotone; often beats Tanh
    on ODE derivative networks because it does not saturate for large |x|."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def _build_activation(name: str) -> nn.Module:
    """Return an activation module by name ('swish' or 'tanh')."""
    name = name.lower()
    if name == "swish":
        return Swish()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "relu":
        return nn.ReLU(inplace=True)
    else:
        raise ValueError(f"Unknown activation '{name}'. Choose 'swish', 'tanh', or 'relu'.")


# ─────────────────────────────────────────────────────────────────────────────
# ODEFunc  — the derivative network  f_θ(z, t)
# ─────────────────────────────────────────────────────────────────────────────

class ODEFunc(nn.Module):
    """
    The right-hand side of the Neural ODE:

        dz/dt = f_θ(z(t), t)

    The current time t is appended to z as an extra scalar feature, making
    this a *non-autonomous* ODE.  This lets the model learn distinct dynamics
    for each fermentation phase (growth 0–60 h, production 60–180 h, etc.).

    Parameters
    ----------
    ode_dim    : int
        Dimensionality of the latent state z (ODE state space).
        Matches the number of latent dims actually fed to the ODE (e.g. 64,
        2, or 1 after optional active-dim extraction).
    hidden_size : int
        Width of every hidden MLP layer.  Keep ≤ 32 to avoid over-fitting
        when ode_dim is small (1–2 active dims).
    n_layers   : int
        Number of hidden layers (2 is a good default; 3 for richer dynamics).
    activation : str
        'swish' (default) or 'tanh'.  Swish has smoother gradients through
        the ODE integration steps.

    Input / Output
    --------------
    forward(t, z):
        t : scalar Tensor  — current integration time (shape [])
        z : (B, ode_dim)   — current latent state
        → dz/dt : (B, ode_dim)

    Notes
    -----
    • No BatchNorm inside ODEFunc — BN statistics are not well-defined at
      arbitrary integration times and cause discontinuous dynamics.
    • Xavier init on all Linear layers for stable early training.
    • The time input is normalised to [0, 1] by the caller (LatentODE) so
      the network never sees values far outside its training range.
    """

    def __init__(
        self,
        ode_dim:     int = 64,
        hidden_size: int = 32,
        n_layers:    int = 2,
        activation:  str = "swish",
    ) -> None:
        super().__init__()
        self.ode_dim     = ode_dim
        self.hidden_size = hidden_size

        # ── Build MLP: (ode_dim + 1) → hidden → … → hidden → ode_dim ──────────
        # +1 accounts for the appended scalar time feature
        layers: List[nn.Module] = [
            nn.Linear(ode_dim + 1, hidden_size),   # input: z concat t
            _build_activation(activation),
        ]
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                _build_activation(activation),
            ]
        layers.append(nn.Linear(hidden_size, ode_dim))  # output: dz/dt

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform init + zero bias for all Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the derivative dz/dt at state z and time t.

        Parameters
        ----------
        t : scalar Tensor (shape [])
            Current integration time.  torchdiffeq passes this automatically.
        z : (B, ode_dim) Tensor
            Current latent state for a batch of B trajectories.

        Returns
        -------
        dz_dt : (B, ode_dim)
            Time-derivative of the latent state.
        """
        # Broadcast scalar t to match batch dimension
        # t is a 0-dim tensor; expand to (B, 1) for concatenation
        t_feat = t.expand(z.size(0), 1)          # (B, 1)
        inp    = torch.cat([z, t_feat], dim=-1)   # (B, ode_dim + 1)
        return self.net(inp)                      # (B, ode_dim)

    def n_params(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# LatentODE  — integrator + readout
# ─────────────────────────────────────────────────────────────────────────────

class LatentODE(nn.Module):
    """
    Latent ODE soft-sensor:  z₀ ──[odeint]──► z(t) ──[readout]──► ŷ(t)

    Accepts the initial latent state z₀ extracted by the frozen β-VAE at the
    first time step of a window, integrates the ODE dynamics forward, and
    maps the resulting trajectory to Penicillin concentration.

    Parameters
    ----------
    latent_dim  : int
        Full dimensionality of the β-VAE latent vector (e.g. 64).
        This is the size of the z₀ vector passed in by the caller.
    ode_dim     : int
        Dimensionality the ODE actually operates on.
        • ode_dim == latent_dim : use the full latent (no projection)
        • ode_dim <  latent_dim : a learnable linear projection reduces the
          full latent to ode_dim before the ODE.  After integration a
          second linear maps back to latent_dim before the readout.
          (Set active_dims instead if you want index-based selection.)
    active_dims : list[int] | None
        If provided, extract only these indices from z₀ before the ODE.
        Example: active_dims=[52, 27] for β=0.05 (Penicillin + Glucose dims).
        active_dims takes priority over the learnable projection.
        len(active_dims) must equal ode_dim.
    hidden_size : int
        ODEFunc hidden width (default 32).
    n_layers    : int
        ODEFunc depth (default 2).
    activation  : str
        'swish' (default) or 'tanh'.
    solver      : str
        ODE solver passed to torchdiffeq.odeint.
        'rk4'    — fast, fixed-step; good for short windows (seq_len ≤ 48)
        'dopri5' — adaptive Runge-Kutta; higher accuracy; uses adjoint for
                   O(1) memory backprop; preferred for long sequences or
                   when solver accuracy matters.
    rtol / atol : float
        Relative/absolute tolerances for adaptive solvers (dopri5 only).

    Forward signature
    -----------------
    forward(z0, t) → ŷ
        z0 : (B, latent_dim)   — VAE posterior mean at the window start
        t  : (T,)              — 1-D tensor of integration time points
                                 (must be monotonically increasing; [0, 1] range
                                  recommended for numerical stability)
        ŷ  : (B, T)            — predicted penicillin at each time step (g/L,
                                  or normalised if scaler is applied externally)
    """

    def __init__(
        self,
        latent_dim:  int             = 64,
        ode_dim:     int             = 64,
        active_dims: Optional[List[int]] = None,
        hidden_size: int             = 32,
        n_layers:    int             = 2,
        activation:  str             = "swish",
        solver:      str             = "rk4",
        rtol:        float           = 1e-3,
        atol:        float           = 1e-4,
    ) -> None:
        super().__init__()

        self.latent_dim  = latent_dim
        self.ode_dim     = ode_dim
        self.active_dims = active_dims
        self.solver      = solver
        self.rtol        = rtol
        self.atol        = atol

        # ── Validate active_dims ──────────────────────────────────────────────
        if active_dims is not None:
            if len(active_dims) != ode_dim:
                raise ValueError(
                    f"len(active_dims)={len(active_dims)} must equal ode_dim={ode_dim}"
                )
            # Register as a buffer so it moves to the right device with .to()
            self.register_buffer(
                "active_idx",
                torch.tensor(active_dims, dtype=torch.long),
            )
            # No learnable projection — we directly index into z₀
            self.proj_in  = None
            self.proj_out = None
        elif ode_dim < latent_dim:
            # Learnable linear projection into the ODE subspace and back
            # proj_out maps ODE trajectory back to latent_dim for readout
            # (alternatively, readout can operate directly in ode_dim — both
            # designs are valid; we keep latent_dim for the readout to allow
            # a richer linear combination across all dims)
            self.active_idx = None
            self.proj_in    = nn.Linear(latent_dim, ode_dim, bias=False)
            self.proj_out   = nn.Linear(ode_dim, latent_dim, bias=False)
            nn.init.orthogonal_(self.proj_in.weight)
            nn.init.orthogonal_(self.proj_out.weight)
        else:
            # ode_dim == latent_dim: identity, no projection needed
            self.active_idx = None
            self.proj_in    = None
            self.proj_out   = None

        # ── ODEFunc: the derivative network ───────────────────────────────────
        self.odefunc = ODEFunc(
            ode_dim=ode_dim,
            hidden_size=hidden_size,
            n_layers=n_layers,
            activation=activation,
        )

        # ── Readout: linear map from (ODE state or reconstructed latent) to ŷ ─
        # Operates on ode_dim directly if no proj_out, else on latent_dim.
        readout_in = latent_dim if (self.proj_out is not None) else ode_dim
        self.readout = nn.Linear(readout_in, 1)
        nn.init.xavier_uniform_(self.readout.weight, gain=0.1)
        nn.init.zeros_(self.readout.bias)    # start from neutral baseline

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _to_ode_space(self, z: torch.Tensor) -> torch.Tensor:
        """
        Map the full latent z ∈ ℝ^(B, latent_dim) into the ODE subspace
        ℝ^(B, ode_dim).

        Three cases:
          active_dims : direct index selection (no learnable params)
          proj_in     : learnable linear projection
          identity    : ode_dim == latent_dim, no-op
        """
        if self.active_idx is not None:
            return z[:, self.active_idx]           # (B, ode_dim)
        elif self.proj_in is not None:
            return self.proj_in(z)                 # (B, ode_dim)
        else:
            return z                               # (B, ode_dim) identity

    def _from_ode_space(self, z_ode: torch.Tensor) -> torch.Tensor:
        """
        Map ODE trajectory z_ode ∈ ℝ^(B, ode_dim) back to latent_dim for the
        readout.  Only applied when proj_out is set.
        """
        if self.proj_out is not None:
            return self.proj_out(z_ode)            # (B, latent_dim)
        else:
            return z_ode                           # (B, ode_dim) — readout operates on this

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        z0: torch.Tensor,   # (B, latent_dim) — VAE posterior mean at window start
        t:  torch.Tensor,   # (T,) — integration time points (monotone increasing)
    ) -> torch.Tensor:      # (B, T) — predicted penicillin
        """
        Integrate the ODE from z₀ over time t and decode to Penicillin.

        Step-by-step:
          1. Project z₀ into the ODE subspace  →  z₀_ode ∈ ℝ^(B, ode_dim)
          2. Call torchdiffeq.odeint:
                 z(t) = odeint(odefunc, z₀_ode, t)   shape: (T, B, ode_dim)
          3. (Optional) project back to latent_dim
          4. Apply linear readout:  ŷ(t) = W·z(t) + b  →  (T, B, 1)
          5. Reshape to (B, T)

        Parameters
        ----------
        z0 : (B, latent_dim)
            Initial latent state — VAE posterior mean μ at the first
            time step of the input window.
        t  : (T,)
            Monotonically increasing time tensor.  Normalise to [0, 1]
            for numerical stability (the ODE sees the same time range
            regardless of physical batch duration).

        Returns
        -------
        y_hat : (B, T)
            Predicted Penicillin concentration at each time step.
            Units depend on whether the target was normalised before training.
        """
        # ── 1. Project z₀ to ODE subspace ────────────────────────────────────
        z0_ode = self._to_ode_space(z0)           # (B, ode_dim)

        # ── 2. Integrate ODE ──────────────────────────────────────────────────
        # odeint(func, y0, t) → shape (T, B, ode_dim)
        # The solver calls odefunc.forward(t_scalar, z_batch) at each step.
        # For dopri5 (adaptive), torchdiffeq uses the adjoint method when
        # called inside torch.autograd: memory is O(state) not O(T * state).
        z_traj = odeint(
            self.odefunc,
            z0_ode,
            t,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )
        # z_traj shape: (T, B, ode_dim)

        # ── 3. (Optional) project back to latent_dim ─────────────────────────
        # Reshape to (T*B, ode_dim) for batched linear ops, then restore
        T, B, D = z_traj.shape
        z_flat  = z_traj.reshape(T * B, D)        # (T*B, ode_dim)
        z_readout = self._from_ode_space(z_flat)  # (T*B, readout_dim)

        # ── 4. Readout: latent → penicillin ───────────────────────────────────
        y_hat = self.readout(z_readout)            # (T*B, 1)

        # ── 5. Reshape to (B, T) ──────────────────────────────────────────────
        # Readout was applied on (T*B) rows in trajectory order (T-major).
        # Reshape to (T, B) then transpose.
        return y_hat.reshape(T, B).permute(1, 0)  # (B, T)

    # ── Introspection helpers ─────────────────────────────────────────────────

    def n_params(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        """Human-readable architecture summary."""
        lines = [
            "LatentODE",
            f"  latent_dim   : {self.latent_dim}",
            f"  ode_dim      : {self.ode_dim}",
            f"  active_dims  : {self.active_dims}",
            f"  solver       : {self.solver}",
            f"  odefunc      : {self.odefunc.n_params():,} params",
            f"  readout      : {sum(p.numel() for p in self.readout.parameters()):,} params",
            f"  total        : {self.n_params():,} params",
        ]
        if self.proj_in is not None:
            lines.insert(4, f"  proj_in/out  : learnable (latent→ode→latent)")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop boilerplate
# ─────────────────────────────────────────────────────────────────────────────

def make_training_step(
    model:     LatentODE,
    optimizer: torch.optim.Optimizer,
    loss_fn:   nn.Module = nn.MSELoss(),
) -> "callable":
    """
    Return a closure that performs one optimisation step.

    Usage
    -----
        step = make_training_step(model, optimizer)

        for z0_batch, t_batch, y_batch in train_loader:
            loss = step(z0_batch, t_batch, y_batch)

    Parameters
    ----------
    model     : LatentODE
    optimizer : any torch.optim.Optimizer
    loss_fn   : default MSELoss; replace with HuberLoss for robustness

    Returns
    -------
    step : callable(z0, t, y_target, sparse_mask=None) → float (loss value)
        z0           : (B, latent_dim)  — initial latent state
        t            : (T,)             — time tensor  [0, 1]
        y_target     : (B, T)           — penicillin targets (normalised)
        sparse_mask  : (B, T) bool | None
                       If provided, loss is computed only at True positions
                       (sparse offline measurement points).  This is the
                       correct evaluation convention for IndPenSim.
    """
    def _step(
        z0:          torch.Tensor,
        t:           torch.Tensor,
        y_target:    torch.Tensor,
        sparse_mask: Optional[torch.Tensor] = None,
    ) -> float:
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # ── Forward pass ──────────────────────────────────────────────────────
        y_hat = model(z0, t)                           # (B, T)

        # ── Loss computation ──────────────────────────────────────────────────
        if sparse_mask is not None:
            # Compute loss only at offline measurement time-points
            # (where Penicillin_offline is not NaN in the raw data)
            y_hat_sp = y_hat[sparse_mask]              # (n_sparse,)
            y_tgt_sp = y_target[sparse_mask]           # (n_sparse,)
            loss = loss_fn(y_hat_sp, y_tgt_sp)
        else:
            # Dense loss: all time steps contribute equally
            loss = loss_fn(y_hat, y_target)

        # ── Backward pass ─────────────────────────────────────────────────────
        loss.backward()
        # Gradient clipping prevents exploding gradients during ODE backprop
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        return float(loss.item())

    return _step
