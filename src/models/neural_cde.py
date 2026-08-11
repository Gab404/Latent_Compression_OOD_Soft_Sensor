"""
src/models/neural_cde.py
========================
Neural Controlled Differential Equation (Neural CDE) soft-sensor for
penicillin fermentation.

Why CDE over ODE?
-----------------
A standard Neural ODE solves  dh/dt = f_θ(h, t),  an *autonomous* system
that cannot observe external interventions after t=0.  For an industrial
fed-batch process the operator continuously changes glucose feed rate, aeration,
pH, and temperature — all of which directly drive the fermentation trajectory.

A Neural CDE instead solves:

    dh(t) = f_θ(h(t)) dX(t)

where X(t) is a *continuous path* interpolated from the observed data sequence
[Z(tₙ), U(tₙ), tₙ].  The derivative dX/dt is automatically available from
the spline, so every small change in feed rate or aeration is continuously
injected into the hidden-state dynamics.  This makes the model inherently
*non-autonomous* and control-aware without requiring any architectural changes
to handle irregular or missing observations.

Path construction
-----------------
X = concat(Z, U, t)   shape: (batch, seq_len, input_dim)

    Z   — the 2 active β-VAE latent dims  [z52_pen, z27_glu]
    U   — scaled control inputs           [Fs, Fg, pH, T]
    t   — physical time in hours          [normalised to 0..1]

Including time *explicitly as a channel* of X is the standard torchcde
convention and makes f_θ a non-autonomous system without any special treatment
in the CDEFunc itself — the network only sees h (not t or X directly).

Cubic spline interpolation
--------------------------
torchcde.natural_cubic_coeffs(X) computes the unique cubic spline that passes
through all data points in X with natural (zero-second-derivative) boundary
conditions.  The coefficients are stored once and served via the DataLoader.
In the forward pass CubicSpline(coeffs) just wraps these coefficients into a
differentiable callable — it is cheap (no recomputation).

Integration
-----------
torchcde.cdeint(X=spline, func=cde_func, z0=h0, t=t_eval) calls an ODE solver
on the *derived* system:

    d/dt h(t) = f_θ(h(t)) · (dX/dt)(t)

where · is the batched matrix-vector product described below.  Any torchdiffeq
solver can be used; we default to 'rk4' (fixed-step) for speed and simplicity.

Key matrix-vector contract (the trickiest part)
-----------------------------------------------
CDEFunc.forward(t, h) must return a 3-D tensor of shape

    (batch, hidden_dim, input_dim)

This is the *matrix* W = f_θ(h).  torchcde internally evaluates:

    dX    = X.derivative(t)              # (batch, input_dim)
    dh/dt = W @ dX                       # (batch, hidden_dim, input_dim)
                                         #       @ (batch, input_dim, 1)
                                         #       = (batch, hidden_dim, 1)
                                         #  → squeezed to (batch, hidden_dim)

Equivalently:  dh_i/dt = Σ_j  W_{ij}(h) · dX_j/dt

The network thus learns how each *rate of change* of an input channel (latent
state, control, or time) contributes to the *rate of change* of each hidden
dimension.  This is strictly more expressive than a vanilla RNN gate which only
sees the instantaneous value X(t).

Architecture
------------
CDEFunc   — h ∈ ℝ^hidden_dim  →  W ∈ ℝ^(hidden_dim × input_dim)
              MLP with Tanh output (bounds the matrix entries, stabilises long
              integrations; Swish/ReLU in intermediate layers)

NeuralCDE — X(t₀) ─[initial_net]─► h₀
                          ↓
               torchcde.cdeint(spline, CDEFunc, h₀, t)
                          ↓
                h(t₁), …, h(T)   shape (batch, T, hidden_dim)
                          ↓
                [readout]  h(t) → ŷ(t) ∈ ℝ¹   (Penicillin g/L)

Usage (standalone)
------------------
    from src.models.neural_cde import NeuralCDE, build_path, compute_coeffs

    # Build the input path X from latent Z, controls U, and time
    X      = build_path(Z, U, t_norm)          # (B, T, input_dim)
    coeffs = compute_coeffs(X, t_norm)         # (B, T-1, 4*input_dim)

    model  = NeuralCDE(input_dim=7, hidden_dim=32)
    y_hat  = model(coeffs, t_norm)             # (B, T)  — Penicillin g/L
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torchcde


# ─────────────────────────────────────────────────────────────────────────────
# Activation helper  (shared with latent_ode.py)
# ─────────────────────────────────────────────────────────────────────────────

class Swish(nn.Module):
    """Swish activation: x · σ(x).  Smooth, non-monotone."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def _activation(name: str) -> nn.Module:
    return {"swish": Swish(), "tanh": nn.Tanh(), "relu": nn.ReLU(inplace=True)}[name]


# ─────────────────────────────────────────────────────────────────────────────
# Path builder helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_path(
    Z:      torch.Tensor,   # (B, T, n_latent)   — β-VAE active dims
    U:      torch.Tensor,   # (B, T, n_ctrl)     — scaled control inputs
    t_norm: torch.Tensor,   # (T,)               — normalised time [0, 1]
) -> torch.Tensor:
    """
    Concatenate latent states, control inputs, and time into a single path X.

    X(t) = [Z(t), U(t), t]   ∈ ℝ^input_dim
    input_dim = n_latent + n_ctrl + 1

    Why include time as a channel?
    --------------------------------
    torchcde's CDEFunc only receives the hidden state h — it does NOT receive
    t or X(t) directly.  By embedding time as the last channel of X, the
    *derivative* dX/dt includes dt/dt = 1, which effectively acts as a
    constant forcing term that gives the CDE awareness of its position on the
    time axis.  This replicates the non-autonomous behaviour of the Neural ODE
    without any extra design effort in CDEFunc.

    Parameters
    ----------
    Z      : (B, T, n_latent)  — β-VAE posterior means [z52, z27]
    U      : (B, T, n_ctrl)    — scaled control variables [Fs, Fg, pH, T_ctrl]
    t_norm : (T,)              — time normalised to [0, 1] for numerics

    Returns
    -------
    X : (B, T, n_latent + n_ctrl + 1)
    """
    B, T, _ = Z.shape
    # Broadcast t_norm (T,) → (B, T, 1)
    t_ch = t_norm.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)  # (B, T, 1)
    return torch.cat([Z, U, t_ch], dim=-1)                       # (B, T, input_dim)


def compute_coeffs(
    X:      torch.Tensor,   # (B, T, input_dim)
    t_norm: torch.Tensor,   # (T,)
) -> torch.Tensor:
    """
    Compute cubic spline coefficients from the input path X.

    torchcde.natural_cubic_coeffs fits a unique cubic spline that passes
    through every (t_n, X_n) point with zero second derivative at the
    boundaries (natural boundary conditions — no artefact at the ends).

    This is the *offline* step: coefficients are computed once and stored in
    the Dataset.  In the forward pass, CubicSpline(coeffs) just reads them
    without recomputation.

    Why cubic splines?
    ------------------
    Piecewise cubic interpolation gives C² continuity — the path X(t) has a
    well-defined second derivative everywhere.  This means dX/dt (used by
    cdeint) is C¹ (continuously differentiable), which is required for RK4 to
    achieve 4th-order convergence.

    Parameters
    ----------
    X      : (B, T, input_dim)  — output of build_path()
    t_norm : (T,)               — must match the t used in build_path()

    Returns
    -------
    coeffs : (B, T-1, 4 * input_dim)
        Four coefficients per piece per input dimension.
        Stored as contiguous memory; served by DataLoader without overhead.
    """
    return torchcde.natural_cubic_coeffs(X, t=t_norm)  # (B, T-1, 4 * input_dim)


# ─────────────────────────────────────────────────────────────────────────────
# CDEFunc — the matrix-valued vector field  f_θ(h) → W ∈ ℝ^(hidden × input)
# ─────────────────────────────────────────────────────────────────────────────

class CDEFunc(nn.Module):
    """
    The vector field of the Neural CDE:

        dh(t) = f_θ(h(t)) dX(t)
        f_θ : ℝ^hidden_dim  →  ℝ^(hidden_dim × input_dim)

    This is an MLP that maps the current hidden state h to a *matrix* W.
    torchcde then computes the actual hidden-state velocity as:

        dh/dt = W · (dX/dt)   [matrix-vector product]

    Dimensionality commentary (critical — read before modifying)
    ------------------------------------------------------------
    Let:
        B  = batch size
        H  = hidden_dim  (e.g. 32)
        I  = input_dim   (e.g. 7  = 2 latent + 4 ctrl + 1 time)

    forward(t, h):
        Input:
            t : scalar Tensor  (shape [])   — current integration time
                                              NOT used directly here (the CDE
                                              gets time awareness via the time
                                              channel of X, not via t)
            h : (B, H)                      — current hidden state
        Output:
            W : (B, H, I)                   — the control matrix

    torchcde internal step after forward():
        dX  : (B, I)                        — X.derivative(t) from the spline
        dh  : (B, H, I) × (B, I, 1) = (B, H, 1)   →  squeezed to  (B, H)
              i.e.  dh_i = Σ_j W_{ij} · (dX_j/dt)

    Implementation choices
    ----------------------
    • The network outputs H×I values and we reshape to (B, H, I).
      This is the simplest correct implementation.

    • Tanh is applied to the *output layer* only.  This bounds the matrix
      entries to (−1, +1), preventing the hidden state from growing without
      bound during long integrations (~230 h batches at Δt=0.2 h).
      The bound does NOT limit expressivity: the magnitude is controlled by
      the linear output layer weights before Tanh.

    • Swish activations in the *hidden layers*: smooth gradients through the
      ODE integration steps; empirically better than ReLU for CDEs.

    • No BatchNorm: BN statistics are evaluated at arbitrary t values during
      integration, which would give inconsistent normalisation.  Use LayerNorm
      if normalisation is desired (commented out below for reference).

    • The time scalar t (passed by torchcde) is intentionally NOT used here.
      Time awareness comes from the time channel embedded in X, whose derivative
      dX_time/dt = 1/T_norm is always available to the matrix-vector product.
      This avoids the t→FeatureEmbed design which can cause instability for
      long horizons far outside the training distribution.

    Parameters
    ----------
    input_dim  : int  — I  (n_latent + n_ctrl + 1)
    hidden_dim : int  — H  (keep ≤ 32 for small latent spaces)
    n_layers   : int  — number of hidden MLP layers (2 is a good default)
    activation : str  — 'swish' or 'tanh' for hidden layers
    """

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int,
        n_layers:   int = 2,
        activation: str = "swish",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim  = input_dim
        output_size     = hidden_dim * input_dim  # H × I flattened

        # ── MLP:  H → hidden → … → hidden → H*I  ──────────────────────────────
        # We DO NOT include t as an extra input (see docstring rationale above).
        layers: List[nn.Module] = [
            nn.Linear(hidden_dim, hidden_dim),   # (B, H) → (B, H)
            _activation(activation),
        ]
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                _activation(activation),
            ]
        layers.append(nn.Linear(hidden_dim, output_size))   # (B, H) → (B, H*I)
        layers.append(nn.Tanh())                             # bound entries to (−1, +1)

        self.net = nn.Sequential(*layers)

        # Xavier-uniform init with small gain keeps early dynamics stable
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        t: torch.Tensor,   # scalar []  — integration time (not used directly)
        h: torch.Tensor,   # (B, H)     — current hidden state
    ) -> torch.Tensor:     # (B, H, I)  — the control matrix W
        """
        Evaluate the matrix-valued vector field f_θ(h).

        Shape walkthrough:
            h  : (B, H)
            → self.net(h)          : (B, H * I)  — flattened matrix
            → .view(B, H, I)       : (B, H, I)   — the control matrix W

        torchcde then computes:
            dX/dt : (B, I)          — from the cubic spline
            dh/dt = W @ (dX/dt)    using torch.bmm:
                  : (B, H, I) @ (B, I, 1) = (B, H, 1)
                  → squeezed to (B, H)
        """
        B = h.shape[0]
        W_flat = self.net(h)                          # (B, H * I)
        return W_flat.view(B, self.hidden_dim, self.input_dim)   # (B, H, I)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# NeuralCDE — the full model: initial net + CDE integration + readout
# ─────────────────────────────────────────────────────────────────────────────

class NeuralCDE(nn.Module):
    """
    Neural Controlled Differential Equation soft-sensor.

    Data flow
    ---------
    (offline, in Dataset)
        X      = build_path(Z, U, t_norm)           (B, T, I)
        coeffs = compute_coeffs(X, t_norm)           (B, T-1, 4I)
                 └── stored in DataLoader

    (online, forward pass)
        spline = CubicSpline(coeffs, t=t_norm)       — fast, no recomputation
        h₀     = initial_net(X[:, 0, :])             (B, H)
        h(t)   = cdeint(spline, cde_func, h₀, t)     (B, T, H)
        ŷ(t)   = readout(h(t))                        (B, T, 1) → (B, T)

    Parameters
    ----------
    input_dim   : int  — I = n_latent + n_ctrl + 1 (time channel)
    hidden_dim  : int  — H (keep ≤ 32 for the 2-D latent space; 16 is enough)
    n_layers    : int  — CDEFunc MLP depth
    activation  : str  — 'swish' or 'tanh' for CDEFunc hidden layers
    adjoint     : bool — if True, use the adjoint method for memory-efficient
                         backprop (O(H) memory instead of O(T·H)); slightly
                         slower per step but essential for very long sequences
    solver      : str  — ODE solver key passed to torchdiffeq via torchcde
                         'rk4'    — fixed-step 4th-order Runge-Kutta (default)
                         'dopri5' — adaptive; more accurate for stiff systems
                         'euler'  — explicit Euler; fastest but least accurate
    solver_kws  : dict — extra kwargs forwarded to the solver
                         e.g. {'step_size': 0.2} for rk4 (in normalised time)
    predict_all_steps : bool
        If True  (default): return predictions at every time step → (B, T)
        If False           : return only the final hidden state → (B, 1)
                             Use False when you only care about the endpoint.
    """

    def __init__(
        self,
        input_dim:         int,
        hidden_dim:        int   = 32,
        n_layers:          int   = 2,
        activation:        str   = "swish",
        adjoint:           bool  = True,
        solver:            str   = "rk4",
        solver_kws:        Optional[dict] = None,
        predict_all_steps: bool  = True,
    ) -> None:
        super().__init__()
        self.input_dim         = input_dim
        self.hidden_dim        = hidden_dim
        self.adjoint           = adjoint
        self.solver            = solver
        self.solver_kws        = solver_kws or {}
        self.predict_all_steps = predict_all_steps

        # ── Initial network  X(t₀) → h₀ ─────────────────────────────────────
        # Maps the first observation of the path (at t=0) to the initial
        # hidden state.  A single linear layer is sufficient — it is fast and
        # differentiable, and there is no need for complex initialisation
        # because h₀ will be corrected immediately by the CDE dynamics.
        #
        # Shape:  X(t₀) : (B, I)  →  h₀ : (B, H)
        self.initial_net = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_uniform_(self.initial_net.weight, gain=0.1)
        nn.init.zeros_(self.initial_net.bias)

        # ── CDEFunc: the vector field f_θ(h) → W ─────────────────────────────
        # See CDEFunc docstring for the full dimensionality commentary.
        self.cde_func = CDEFunc(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
        )

        # ── Readout: h(t) → ŷ(t) ─────────────────────────────────────────────
        # Linear map from hidden state to predicted Penicillin concentration.
        # A single linear layer is deliberately used:
        #   (1) Preserves interpretability (weights = linear sensor over hidden
        #       dimensions)
        #   (2) Avoids over-fitting when the CDE is already expressive
        #   (3) Zero bias initialisation → neutral baseline before training
        #
        # Shape:  h(t) : (B, T, H)  →  ŷ(t) : (B, T, 1)  →  squeezed (B, T)
        self.readout = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.readout.weight, gain=0.1)
        nn.init.zeros_(self.readout.bias)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        coeffs: torch.Tensor,   # (B, T-1, 4*I)  — spline coefficients from DataLoader
        t_norm: torch.Tensor,   # (T,)            — normalised time grid [0, 1]
    ) -> torch.Tensor:          # (B, T) or (B, 1) — predicted Penicillin
        """
        Run the full Neural CDE forward pass.

        Step-by-step shape trace (B=batch, T=seq_len, H=hidden_dim, I=input_dim):

        1.  spline = CubicSpline(coeffs, t=t_norm)
                     → differentiable path object (no heavy computation)
                     → spline.evaluate(t) : (B, I)
                     → spline.derivative(t) : (B, I)

        2.  X₀ = spline.evaluate(t_norm[0])       : (B, I)
            h₀ = initial_net(X₀)                   : (B, H)

        3.  h(t) = cdeint(X=spline, func=cde_func,
                          z0=h₀, t=t_norm)          : (B, T, H)

            Internally at each solver step:
                W     = cde_func(t, h)             : (B, H, I)  — the matrix
                dX/dt = spline.derivative(t)        : (B, I)
                dh/dt = W @ dX/dt                  : (B, H)
                        = bmm(W, dX.unsqueeze(-1)).squeeze(-1)

        4.  ŷ = readout(h(t))                      : (B, T, 1)
            ŷ = ŷ.squeeze(-1)                       : (B, T)

        Parameters
        ----------
        coeffs : (B, T-1, 4*I)
            Precomputed cubic spline coefficients (from compute_coeffs, stored
            in the DataLoader).  Instantiating CubicSpline here is O(1).
        t_norm : (T,)
            Normalised time grid, must match the grid used to compute coeffs.
            Must be monotonically increasing; should be in [0, 1].

        Returns
        -------
        y_hat : (B, T)  if predict_all_steps=True
                (B, 1)  if predict_all_steps=False  (last step only)
        """
        # ── 1. Instantiate the cubic spline from pre-computed coefficients ────
        # This is cheap: CubicSpline just wraps the coefficient tensor.
        # The spline is created fresh each forward call so gradients flow
        # through it correctly with respect to coeffs.
        spline = torchcde.CubicSpline(coeffs, t=t_norm)   # differentiable path

        # ── 2. Compute initial hidden state h₀ from first path observation ───
        # spline.evaluate(t_norm[0]) gives X at the very start of the sequence.
        # Shape: (B, I)
        X0 = spline.evaluate(t_norm[0])       # (B, I)  — first path point
        h0 = self.initial_net(X0)             # (B, H)  — initial hidden state

        # ── 3. Integrate the CDE over the full time grid ─────────────────────
        # torchcde.cdeint(X, func, z0, t) solves:
        #     dh(t) = func(t, h(t)) dX(t)    with h(t_norm[0]) = h₀
        #
        # The adjoint=True flag uses the adjoint sensitivity method for
        # backpropagation: instead of storing all intermediate states (O(T·H)
        # memory), it solves a second ODE backward in time (O(H) memory).
        # This is essential for the ~1150-step fermentation sequences.
        #
        # solver='rk4' with step_size matching Δt_norm keeps the integration
        # aligned with the data measurement grid, avoiding unnecessary
        # sub-step evaluations.
        #
        # Output shape: (B, T, H)
        if self.predict_all_steps:
            t_eval = t_norm          # evaluate at every time point
        else:
            # Only start and end — cheaper; used when only endpoint matters
            t_eval = t_norm[[0, -1]]

        h_traj = torchcde.cdeint(
            X=spline,
            func=self.cde_func,
            z0=h0,
            t=t_eval,
            adjoint=self.adjoint,
            method=self.solver,
            **self.solver_kws,
        )
        # h_traj : (B, T, H)   [or (B, 2, H) if predict_all_steps=False]

        # ── 4. Linear readout: h(t) → ŷ(t) ──────────────────────────────────
        # Apply readout to every time step simultaneously (batched linear).
        # Shape: (B, T, H) → (B, T, 1) → (B, T)
        y_hat = self.readout(h_traj)           # (B, T, 1)
        return y_hat.squeeze(-1)               # (B, T)

    # ── Introspection ─────────────────────────────────────────────────────────

    def n_params(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        """Human-readable architecture summary."""
        initial_p = sum(p.numel() for p in self.initial_net.parameters())
        cde_p     = self.cde_func.n_params()
        readout_p = sum(p.numel() for p in self.readout.parameters())
        return "\n".join([
            "NeuralCDE",
            f"  input_dim   : {self.input_dim}",
            f"  hidden_dim  : {self.hidden_dim}",
            f"  solver      : {self.solver}  adjoint={self.adjoint}",
            f"  initial_net : {initial_p:,} params  ({self.input_dim} → {self.hidden_dim})",
            f"  cde_func    : {cde_p:,} params  (H → H×I matrix-valued field)",
            f"  readout     : {readout_p:,} params  ({self.hidden_dim} → 1)",
            f"  total       : {self.n_params():,} params",
        ])
