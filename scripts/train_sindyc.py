#!/usr/bin/env python
"""
train_sindyc.py
===============
SINDYc (SINDy with Control) for latent-space dynamics discovery in the
β-VAE Raman soft-sensor pipeline for IndPenSim penicillin fermentation.

Scientific goal
---------------
The β=0.05 VAE has compressed 2001-D Raman spectra into a 2-active-dim
latent space Z = [z₅₂, z₂₇] where:
    z₅₂  ↔  Penicillin  (Pearson r = +0.993)
    z₂₇  ↔  Glucose     (Pearson r = −0.959)

We want to discover the explicit ODE
    Ż = f(Z, U)          (SINDYc: SINDy with Control)
where U are the measured control/disturbance variables (feed rates, aeration,
pH, temperature) that drive the fed-batch process.

Unlike a standard Neural ODE, this gives:
  1. An interpretable, algebraic equation (white-box).
  2. Explicit control-input awareness → good OOD generalisation.
  3. A linear readout from z₅₂ to Penicillin concentration (g/L).

Algorithm overview
------------------
  1. Load R1 (train) batches, encode with frozen β-VAE → Z trajectories.
  2. Extract + scale control variables U.
  3. Build PySINDy: SmoothedFiniteDifference differentiator +
                    PolynomialLibrary(degree=2) on [Z, U] +
                    STLSQ sparse optimizer.
  4. Fit on R1, print discovered equations.
  5. Simulate a held-out Fault batch: given Z₀ and U(t), integrate Ż = f(Z, U).
  6. Project z₅₂(t) → Penicillin(t) via the pre-fitted linear readout.
  7. Save plots to outputs/sindyc/.

Usage
-----
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_sindyc.py

    # Tune sparsity threshold
    PYTHONPATH=. .venv/bin/python scripts/train_sindyc.py --threshold 0.05

    # Use degree-3 library (more terms, less sparse)
    PYTHONPATH=. .venv/bin/python scripts/train_sindyc.py --degree 3

    # Choose which control inputs to include (subset)
    PYTHONPATH=. .venv/bin/python scripts/train_sindyc.py --controls Fs Fg pH T
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.preprocessing import MinMaxScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import pysindy as ps

from src.data.beta_vae_encoder import BetaVAERamanEncoderV2
from src.data.dataset import TARGET_COLS

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Configuration constants
# ─────────────────────────────────────────────────────────────────────────────

# Paths
CSV_PATH   = str(_ROOT / "100_Batches_IndPenSim_V3.csv")
VAE_CKPT   = str(_ROOT / "outputs/beta_sweep_e2e/checkpoints/b0.05/vae.pt")
VAE_SCALER = str(_ROOT / "outputs/beta_sweep_e2e/checkpoints/b0.05/vae_scaler.joblib")
OUT_DIR    = _ROOT / "outputs/sindyc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Regime batch ID boundaries (0-indexed)
R1_IDS    = set(range(0,  30))
FAULT_IDS = set(range(90, 100))

# Time step: IndPenSim samples every 0.2 h
DT = 0.2   # hours

# Active latent dimensions from the β=0.05 posterior analysis
#   z₅₂  →  Penicillin proxy  (r = +0.993)
#   z₂₇  →  Glucose proxy     (r = −0.959)
ACTIVE_LATENT_DIMS  = [52, 27]
LATENT_NAMES        = ["z52_pen", "z27_glu"]   # human-readable names

# Pre-fitted linear readout z₅₂ → Penicillin (g/L)
# Derived from the latent trajectory analysis (see outputs/latent_trajectory/).
READOUT_SLOPE     = 10.05   # g/L per unit z₅₂
READOUT_INTERCEPT = 15.12   # g/L

# Control variable columns in the CSV and their short labels.
# We use the four mechanistically most important actuators:
#   Fs  — glucose feed rate:  drives substrate uptake → z₂₇ dynamics
#   Fg  — aeration rate:      oxygen transfer → respiratory metabolism
#   pH  — pH setpoint:        enzyme activity proxy
#   T   — temperature (K):    kinetics Arrhenius factor
CONTROL_COLS_MAP: Dict[str, str] = {
    "Sugar feed rate(Fs:L/h)":                "Fs",
    "Aeration rate(Fg:L/h)":                  "Fg",
    "pH(pH:pH)":                              "pH",
    "Temperature(T:K)":                       "T",
}
# Optional extended set (uncomment to include agitation & DO2)
# CONTROL_COLS_MAP.update({
#     "Agitator RPM(RPM:RPM)":                  "RPM",
#     "Dissolved oxygen concentration(DO2:mg/L)":"DO2",
# })

CONTROL_COLS   = list(CONTROL_COLS_MAP.keys())
CONTROL_NAMES  = list(CONTROL_COLS_MAP.values())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_batches_by_regime(
    csv_path: str,
    regime_ids: set,
    min_len: int = 50,
) -> List[pd.DataFrame]:
    """
    Load IndPenSim CSV and return DataFrames for the requested batch IDs.

    Batch boundaries are detected by a time-reset (t[i] < t[i-1]).
    Target columns (Penicillin, Biomass) are dense-interpolated so that the
    LSTM-style evaluation code stays compatible; the *sparse* original
    measurements are preserved in _penicillin_sparse / _biomass_sparse.

    Parameters
    ----------
    csv_path   : path to 100_Batches_IndPenSim_V3.csv
    regime_ids : set of 0-based batch IDs to return
    min_len    : drop batches shorter than this (artefact guard)

    Returns
    -------
    list of per-batch DataFrames, sorted by batch ID
    """
    print(f"\nLoading CSV: {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Detect batch boundaries by monotonicity break in Time (h)
    t_vals    = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid
    df["_batch_id"] = batch_ids

    # Save sparse offline lab grid before interpolating
    df["_penicillin_sparse"] = df[TARGET_COLS[1]].copy()
    df["_biomass_sparse"]    = df[TARGET_COLS[0]].copy()

    # Dense-interpolate target columns (fills NaN gaps for training targets)
    for tc in TARGET_COLS:
        df[tc] = df[tc].interpolate(method="linear").ffill().bfill()

    batches = []
    for b_id in sorted(regime_ids):
        sub = df[df["_batch_id"] == b_id].reset_index(drop=True)
        if len(sub) >= min_len:
            batches.append(sub)
        else:
            print(f"  [WARN] batch {b_id} has {len(sub)} rows — skipped")

    print(f"  Loaded {len(batches)} batches for IDs {sorted(regime_ids)[:3]}...")
    return batches


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Raman encoding  →  Z trajectories
# ─────────────────────────────────────────────────────────────────────────────

def build_z_trajectories(
    batches:  List[pd.DataFrame],
    encoder:  BetaVAERamanEncoderV2,
    active_dims: List[int] = ACTIVE_LATENT_DIMS,
) -> List[np.ndarray]:
    """
    Encode every batch with the frozen β-VAE and extract the active latent dims.

    The encoder returns the posterior *mean* μ (no sampling noise), so the
    trajectories are deterministic and relatively smooth.  Any residual
    micro-fluctuations are handled later by SmoothedFiniteDifference.

    Parameters
    ----------
    batches     : list of per-batch DataFrames (raw CSV columns still present)
    encoder     : frozen BetaVAERamanEncoderV2
    active_dims : indices of the active latent dimensions to extract

    Returns
    -------
    Z_list : list of (T_i, len(active_dims)) float32 arrays
             Each row is [z₅₂, z₂₇] at one time step.
    """
    Z_list = []
    for i, batch_df in enumerate(batches):
        # Full 64-D posterior mean μ:  shape (T, 64)
        mu_full = encoder.encode_dataframe(batch_df)   # (T, 64)
        # Extract only the 2 active dimensions
        Z = mu_full[:, active_dims].astype(np.float32)  # (T, 2)
        Z_list.append(Z)
        print(f"  Batch {i:2d}: T={len(Z)} steps, "
              f"z52 ∈ [{Z[:,0].min():.2f}, {Z[:,0].max():.2f}], "
              f"z27 ∈ [{Z[:,1].min():.2f}, {Z[:,1].max():.2f}]")
    return Z_list


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Control variable extraction + scaling
# ─────────────────────────────────────────────────────────────────────────────

def extract_control_variables(
    batches:      List[pd.DataFrame],
    ctrl_cols:    List[str] = CONTROL_COLS,
    ctrl_names:   List[str] = CONTROL_NAMES,
    scaler:       Optional[MinMaxScaler] = None,
    fit_scaler:   bool = True,
) -> Tuple[List[np.ndarray], MinMaxScaler]:
    """
    Extract and min-max scale the control inputs U from each batch.

    Why scale?
    ----------
    SINDYc builds polynomial features from [Z, U].  If U lives on wildly
    different scales than Z (e.g. temperature ~300 K vs. z₅₂ ~0-3), the
    regression will be ill-conditioned and STLSQ threshold-selection becomes
    unreliable.  We project each column to [0, 1] independently.

    Parameters
    ----------
    batches    : per-batch DataFrames
    ctrl_cols  : raw CSV column names for control inputs
    ctrl_names : short human-readable labels (for equation printing)
    scaler     : pre-fitted MinMaxScaler (pass None to fit fresh on these batches)
    fit_scaler : if True, fit the scaler on the concatenated data first

    Returns
    -------
    U_list : list of (T_i, n_controls) float32 arrays (scaled [0, 1])
    scaler : fitted MinMaxScaler (save this for later simulation)
    """
    # Check all requested columns are present
    for col in ctrl_cols:
        if col not in batches[0].columns:
            raise KeyError(
                f"Control column '{col}' not found. "
                f"Available: {list(batches[0].columns[:20])}..."
            )

    # Concatenate all batches to fit the scaler on the full training distribution
    all_ctrl = np.vstack([b[ctrl_cols].values.astype(np.float32) for b in batches])
    all_ctrl = np.nan_to_num(all_ctrl, nan=0.0)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
    if fit_scaler:
        scaler.fit(all_ctrl)

    U_list = []
    for batch_df in batches:
        u_raw = batch_df[ctrl_cols].values.astype(np.float32)
        u_raw = np.nan_to_num(u_raw, nan=0.0)
        U     = scaler.transform(u_raw).astype(np.float32)   # (T, n_ctrl)
        U_list.append(U)

    print(f"  Extracted {len(ctrl_cols)} control vars: {ctrl_names}")
    print(f"  U shapes: {[u.shape for u in U_list[:3]]}...")
    return U_list, scaler


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — SINDYc model definition
# ─────────────────────────────────────────────────────────────────────────────

def build_sindyc_model(
    latent_names:  List[str] = LATENT_NAMES,
    control_names: List[str] = CONTROL_NAMES,
    degree:        int   = 2,
    threshold:     float = 0.08,
    alpha:         float = 0.05,
    smoothing_bw:  int   = 5,
) -> ps.SINDy:
    """
    Build the SINDYc model with:
        differentiator  : SmoothedFiniteDifference (Savitzky-Golay smoothing)
        feature library : PolynomialLibrary(degree) applied to [Z, U] concat
        optimizer       : STLSQ (sequentially thresholded least squares)

    How PySINDy computes Ż
    ----------------------
    PySINDy internally differentiates Z(t) to obtain Ż(t) via the chosen
    `differentiation_method`.  We use SmoothedFiniteDifference, which:
      1. Applies a moving-average / Savitzky-Golay filter to Z(t) — this
         suppresses the micro-fluctuations in the β-VAE posterior means.
      2. Then differentiates the smoothed signal with finite differences.
    The `smoothing_filter_length` (= smoothing_bw) controls the half-window
    in time steps; a value of 5 corresponds to a 1.0 h smoothing horizon
    at Δt = 0.2 h, which is well below the ~12 h characteristic timescale
    of penicillin fermentation.

    Feature library: PolynomialLibrary on [Z, U]
    --------------------------------------------
    PySINDy's SINDy.fit(x, u=u) concatenates [x, u] column-wise before
    evaluating the library.  So PolynomialLibrary(degree=2) generates:
        degree-0 : 1
        degree-1 : z52, z27, Fs, Fg, pH, T
        degree-2 : z52², z52·z27, z52·Fs, ..., T²
    The cross-terms z52·Fs, z27·Fg, etc. are exactly the SINDYc interaction
    terms needed to capture the effect of control on latent state dynamics.

    STLSQ optimizer
    ---------------
    STLSQ alternates between:
      1. Least-squares fit of Ż = Θ(Z, U) · Ξ
      2. Hard-thresholding: set Ξ[|Ξ| < threshold] = 0 (forces sparsity)
    `threshold` is the key hyperparameter for parsimony.  Too low → dense
    equations with many noise terms; too high → important terms zeroed out.
    `alpha` is an L2 regularisation coefficient applied to the unthresholded
    least-squares step (stabilises conditioning with many candidate terms).

    Parameters
    ----------
    latent_names   : names for the state dimensions  [z52_pen, z27_glu]
    control_names  : names for the control inputs    [Fs, Fg, pH, T]
    degree         : polynomial degree for the candidate library
    threshold      : STLSQ hard-threshold (key sparsity knob)
    alpha          : STLSQ L2 regularisation
    smoothing_bw   : SmoothedFiniteDifference half-window (in time steps)

    Returns
    -------
    model : unfitted ps.SINDy instance
    """

    # ── Differentiation method ────────────────────────────────────────────────
    # SmoothedFiniteDifference first applies a causal smoothing kernel, then
    # differentiates.  This is equivalent to smoothed derivative estimation
    # (similar to the Savitzky-Golay derivative used for the Raman SG filter).
    # `smoother_kws` are forwarded to scipy.signal.savgol_filter under the hood.
    differentiator = ps.SmoothedFiniteDifference(
        smoother_kws={"window_length": smoothing_bw * 2 + 1,  # must be odd
                      "polyorder":     3},
    )

    # ── Feature library ───────────────────────────────────────────────────────
    # When SINDy.fit() is called with u=U, PySINDy *concatenates* X and U
    # column-wise → [z52, z27, Fs, Fg, pH, T] before calling lib.fit_transform.
    # So a single PolynomialLibrary naturally generates all cross-terms.
    feature_library = ps.PolynomialLibrary(
        degree=degree,
        include_interaction=True,   # enables cross-terms like z52·Fs
        include_bias=True,          # includes the constant term "1"
        interaction_only=False,     # also keeps squared terms z52²
    )

    # ── Sparse optimizer ──────────────────────────────────────────────────────
    # STLSQ: Brunton et al. (2016) original optimizer.
    # threshold : minimum coefficient magnitude to keep (hard-threshold).
    # alpha     : ridge regularisation added to the LS step for stability.
    # max_iter  : maximum alternating threshold-LS iterations.
    optimizer = ps.STLSQ(
        threshold=threshold,
        alpha=alpha,
        max_iter=20,
        normalize_columns=True,  # normalise Θ columns before thresholding
    )                            # — important when features have different scales

    # ── Assemble SINDy model ──────────────────────────────────────────────────
    # NOTE (PySINDy 2.x API): feature_names is passed to model.fit(), not here.
    # The constructor only takes optimizer, feature_library, differentiation_method.
    model = ps.SINDy(
        differentiation_method=differentiator,
        feature_library=feature_library,
        optimizer=optimizer,
    )

    n_state   = len(latent_names)
    n_ctrl    = len(control_names)
    n_library = (
        sum(
            1
            for d in range(degree + 1)
            for _ in [None]  # rough estimate; PySINDy counts exactly at fit time
        )
    )
    print(
        f"\n[SINDYc] Model built:"
        f"\n  State dims    : {n_state}   ({latent_names})"
        f"\n  Control dims  : {n_ctrl}   ({control_names})"
        f"\n  Library degree: {degree}"
        f"\n  STLSQ threshold: {threshold}  alpha: {alpha}"
        f"\n  Smoother window: {smoothing_bw * 2 + 1} steps "
        f"({(smoothing_bw * 2 + 1) * DT:.1f} h)"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Model fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_sindyc(
    model:         ps.SINDy,
    Z_list:        List[np.ndarray],
    U_list:        List[np.ndarray],
    latent_names:  List[str] = LATENT_NAMES,
    control_names: List[str] = CONTROL_NAMES,
    dt:            float = DT,
) -> ps.SINDy:
    """
    Fit SINDYc on multiple fermentation trajectories.

    Multiple-trajectory mode  (PySINDy 2.x)
    ----------------------------------------
    In PySINDy 2.x, passing *lists* of arrays for x and u automatically
    enables multiple-trajectory mode — no extra flag needed.  PySINDy:
      1. Differentiates each Z_i(t) independently (respects batch boundaries,
         so no cross-contamination between batches in the derivative estimate).
      2. Evaluates the feature library Θ([Z_i, U_i]) for each batch.
      3. *Stacks* all (Ż, Θ) pairs into one large regression problem.
      4. Runs STLSQ once on the combined stacked system.

    This is the correct approach for batch fermentation: each batch is an
    independent trajectory from a fresh initial condition.

    feature_names  (PySINDy 2.x)
    ------------------------------
    In PySINDy 2.x, feature_names is a parameter of fit(), not __init__().
    It must list ALL variable names in the order of the concatenated [x, u]
    array:  [state_dims..., control_dims...]
    PySINDy uses this to label the library columns during equation printing.

    Parameters
    ----------
    model         : unfitted ps.SINDy from build_sindyc_model()
    Z_list        : list of (T_i, 2) state arrays (z₅₂, z₂₇)
    U_list        : list of (T_i, n_ctrl) control arrays (scaled [0, 1])
    latent_names  : names for the state dimensions
    control_names : names for the control dimensions
    dt            : time step in hours (0.2 for IndPenSim)

    Returns
    -------
    model  : fitted ps.SINDy (in-place mutation + returned for convenience)
    """
    assert len(Z_list) == len(U_list), "Z_list and U_list must have the same length"
    total_steps = sum(len(z) for z in Z_list)
    n_ctrl      = U_list[0].shape[1]
    # The feature_names list must have exactly (n_state + n_ctrl) entries,
    # matching the columns of the concatenated [Z, U] array PySINDy builds.
    all_feature_names = latent_names + control_names[:n_ctrl]

    print(
        f"\n[SINDYc] Fitting on {len(Z_list)} trajectories "
        f"({total_steps:,} time steps total, dt={dt} h) ..."
        f"\n  feature_names = {all_feature_names}"
    )

    # PySINDy 2.x fit() signature:
    #   fit(x, t, x_dot=None, u=None, feature_names=None)
    # Passing lists for x and u → automatic multiple-trajectory mode.
    # t = scalar dt is interpreted as the uniform time step.
    model.fit(
        x=Z_list,
        t=dt,                       # uniform dt; PySINDy builds t_i = [0..T_i*dt]
        u=U_list,                   # list of control arrays → SINDYc mode
        feature_names=all_feature_names,  # labels for [Z, U] columns
    )

    print("\n[SINDYc] Fitting complete.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Print discovered equations
# ─────────────────────────────────────────────────────────────────────────────

def print_equations(
    model:      ps.SINDy,
    state_names: List[str] = LATENT_NAMES,
    lhs_dt:     float = DT,
) -> None:
    """
    Print the discovered SINDYc differential equations in human-readable form.

    PySINDy's model.print() outputs each equation as:
        ẋ_i = coef_1 * term_1 + coef_2 * term_2 + ...
    where only terms with |coef| > threshold are printed (zero terms omitted).

    We additionally print the coefficient matrix Ξ and a term-count summary.
    """
    print("\n" + "═" * 64)
    print("  Discovered SINDYc equations  (Ż = f(Z, U))")
    print("═" * 64)

    # model.print() renders the equations using the feature_names supplied
    # at construction time.  The LHS uses the model's internal naming.
    model.print(lhs=state_names)

    # Coefficient matrix: shape (n_library_terms, n_state_dims)
    # Row i is the coefficient for library term i in each state's equation.
    coeff_matrix = model.coefficients()   # (n_state, n_library_terms)
    feature_names = model.get_feature_names()

    print(f"\n  Library size : {len(feature_names)} candidate terms")
    print(f"  Active terms : ", end="")
    n_active_per_dim = (np.abs(coeff_matrix) > 0).sum(axis=1)
    for i, name in enumerate(state_names):
        print(f"  {name}: {n_active_per_dim[i]} terms")

    print("\n  Non-zero coefficients:")
    print(f"  {'Term':<25s}  " + "  ".join(f"{n:>14s}" for n in state_names))
    print("  " + "-" * (25 + 18 * len(state_names)))
    for j, term in enumerate(feature_names):
        row = coeff_matrix[:, j]
        if np.any(np.abs(row) > 0):
            vals = "  ".join(f"{v:>14.6f}" for v in row)
            print(f"  {term:<25s}  {vals}")
    print("═" * 64)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Simulation (integration of discovered ODE)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_trajectory(
    model:   ps.SINDy,
    Z0:      np.ndarray,
    U:       np.ndarray,
    dt:      float = DT,
    method:  str   = "RK45",
) -> np.ndarray:
    """
    Integrate Ż = f(Z, U(t)) forward in time given an initial condition Z0
    and a measured control sequence U[t].

    How PySINDy simulation works
    ----------------------------
    model.simulate(x0, t, u=u_func) calls scipy.integrate.solve_ivp with:
      • right-hand side  = model.predict(Z, u=u(t)) at each step
      • initial value    = x0
      • time array       = t  (the integration grid)
    The control u_func(t) must be callable OR an array (in which case PySINDy
    interpolates linearly between the provided samples).

    For our use-case: U is a (T, n_ctrl) array of measured control values at
    each Δt step.  We pass u=U directly — PySINDy will interpolate U(t) at
    the sub-steps used internally by the RK45 solver.

    NOTE: model.simulate() assumes the model was fitted with a scalar `t=dt`.
    We reconstruct the full time array explicitly to match the batch length.

    Parameters
    ----------
    model  : fitted ps.SINDy
    Z0     : (2,) initial latent state [z52_0, z27_0]
    U      : (T, n_ctrl) control sequence for the batch (scaled [0, 1])
    dt     : time step (hours)
    method : ODE solver  (RK45 is robust; LSODA handles stiffness)

    Returns
    -------
    Z_pred : (T, 2) simulated trajectory array [z52(t), z27(t)]
             If the integrator diverges, returns a NaN-filled array.
    """
    T = len(U)
    t_span   = np.arange(T) * dt                    # [0.0, 0.2, 0.4, ..., T*dt]
    t_eval   = t_span                                # evaluate at each Δt step

    # PySINDy model.simulate() wraps scipy solve_ivp.
    # Passing u=U passes the measured control array; PySINDy interpolates it
    # linearly to supply u(t) at sub-step query times from the RK45 solver.
    try:
        # PySINDy 2.x simulate() signature:
        #   simulate(x0, t, u=None, integrator='solve_ivp', integrator_kws={})
        # The 'method' key inside integrator_kws selects the ODE solver
        # (RK45, LSODA, etc.) passed through to scipy.integrate.solve_ivp.
        # NOTE: PySINDy will drop the last time point when u is an array
        # (it needs the control at t[-1] for interpolation).  This is expected
        # and benign for our use-case (we pad the missing step below).
        Z_pred = model.simulate(
            x0=Z0.astype(np.float64),
            t=t_eval,
            u=U,                    # (T, n_ctrl) — interpolated between steps
            integrator="solve_ivp",
            integrator_kws={
                "method":   method,      # 'RK45' by default
                "max_step": dt,          # don't skip over measurement points
                "rtol":     1e-5,
                "atol":     1e-7,
            },
        )
        # shape: (T, n_state)
        if Z_pred.shape[0] < T:
            # Solver stopped early (instability): pad with NaN
            pad = np.full((T - Z_pred.shape[0], Z_pred.shape[1]), np.nan)
            Z_pred = np.vstack([Z_pred, pad])
    except Exception as e:
        print(f"  [WARN] ODE integration failed: {e}")
        Z_pred = np.full((T, len(Z0)), np.nan)

    return Z_pred.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Linear readout  z₅₂(t) → Penicillin (g/L)
# ─────────────────────────────────────────────────────────────────────────────

def latent_to_penicillin(
    z52:       np.ndarray,
    slope:     float = READOUT_SLOPE,
    intercept: float = READOUT_INTERCEPT,
) -> np.ndarray:
    """
    Project the z₅₂ trajectory to Penicillin concentration (g/L).

    The linear readout  P(t) = slope · z₅₂(t) + intercept  was established
    from the latent trajectory analysis (Pearson r = +0.993, see
    outputs/latent_trajectory/).  Because the β-VAE encodes Penicillin in a
    linearly decodable dimension, this projection is both interpretable and
    surprisingly accurate.

    Parameters
    ----------
    z52       : (T,) array of z₅₂ values
    slope     : fitted slope  (default: 10.05 g/L per unit z₅₂)
    intercept : fitted bias   (default: 15.12 g/L)

    Returns
    -------
    pen_pred : (T,) penicillin concentration in g/L
    """
    return slope * z52 + intercept


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Evaluation + plotting
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_and_plot(
    batch_df:     pd.DataFrame,
    Z_true:       np.ndarray,
    Z_pred:       np.ndarray,
    batch_id:     int,
    out_dir:      Path,
) -> Dict[str, float]:
    """
    Compute RMSE metrics and save a 2-panel comparison figure.

    Panel 1 (top):   z₅₂(t) — Penicillin proxy  (ground truth vs SINDYc)
    Panel 2 (bottom): z₂₇(t) — Glucose proxy     (ground truth vs SINDYc)
    Rightmost column: decoded Penicillin (g/L) with sparse lab measurements.

    Parameters
    ----------
    batch_df  : raw batch DataFrame (for time axis + sparse pen measurements)
    Z_true    : (T, 2) ground-truth latent trajectory [z52, z27]
    Z_pred    : (T, 2) SINDYc-simulated trajectory (may contain NaN tail)
    batch_id  : 0-based batch index (for file naming)
    out_dir   : directory to write the PNG

    Returns
    -------
    metrics dict: rmse_z52, rmse_z27, rmse_pen_sparse, n_sparse
    """
    time = batch_df["Time (h)"].values
    T    = len(time)

    # ── Latent RMSE ───────────────────────────────────────────────────────────
    valid = ~np.isnan(Z_pred[:, 0]) & ~np.isnan(Z_pred[:, 1])
    rmse_z52 = float(np.sqrt(np.mean((Z_true[valid, 0] - Z_pred[valid, 0]) ** 2)))
    rmse_z27 = float(np.sqrt(np.mean((Z_true[valid, 1] - Z_pred[valid, 1]) ** 2)))

    # ── Penicillin RMSE at sparse points ─────────────────────────────────────
    pen_pred  = latent_to_penicillin(Z_pred[:, 0])    # (T,)
    pen_sparse = batch_df["_penicillin_sparse"].values  # NaN except lab points
    sp_mask    = ~np.isnan(pen_sparse) & valid
    if sp_mask.sum() > 0:
        rmse_pen = float(np.sqrt(np.mean(
            (pen_sparse[sp_mask] - pen_pred[sp_mask]) ** 2
        )))
    else:
        rmse_pen = float("nan")

    metrics = {
        "batch_id":       batch_id,
        "rmse_z52":       rmse_z52,
        "rmse_z27":       rmse_z27,
        "rmse_pen_sparse": rmse_pen,
        "n_sparse":       int(sp_mask.sum()),
        "n_valid_steps":  int(valid.sum()),
    }

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles  = [f"z₅₂ (Penicillin proxy)  RMSE={rmse_z52:.4f}",
               f"z₂₇ (Glucose proxy)     RMSE={rmse_z27:.4f}",
               f"Penicillin readout (g/L)  RMSE={rmse_pen:.4f} g/L"]
    colors  = ["#1f77b4", "#ff7f0e"]   # blue=pred, orange=true

    for ax, dim, title in zip(axes[:2], [0, 1], titles[:2]):
        ax.plot(time, Z_true[:, dim], color="silver", lw=1.5, linestyle="--",
                label="Ground truth (β-VAE μ)")
        ax.plot(time[valid], Z_pred[valid, dim], color=colors[0], lw=1.8,
                label="SINDYc simulation")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel(LATENT_NAMES[dim])
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Rightmost panel: decoded Penicillin
    ax = axes[2]
    # Dense ground-truth via linear readout of true z52
    pen_true_dense = latent_to_penicillin(Z_true[:, 0])
    ax.plot(time, pen_true_dense, color="silver", lw=1.5, linestyle="--",
            label="True z₅₂ → P (g/L)")
    ax.plot(time[valid], pen_pred[valid], color=colors[0], lw=1.8,
            label=f"SINDYc → P  RMSE={rmse_pen:.3f}")
    # Sparse lab measurements
    if sp_mask.sum() > 0:
        ax.scatter(time[sp_mask], pen_sparse[sp_mask], color="black", s=40,
                   zorder=4, label="Offline lab meas.")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Penicillin (g/L)")
    ax.set_title(titles[2], fontsize=9)
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"SINDYc simulation — Batch {batch_id + 1}  "
        f"(ID={batch_id}, {'Fault/R4' if batch_id >= 90 else 'R1'})",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = out_dir / f"sindyc_batch_{batch_id:02d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path.name}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — Summary plot: coefficient heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_coefficient_heatmap(
    model:        ps.SINDy,
    state_names:  List[str] = LATENT_NAMES,
    out_dir:      Path = OUT_DIR,
) -> None:
    """
    Heatmap of the SINDYc coefficient matrix Ξ.

    Rows   = state dimensions  (ż₅₂, ż₂₇)
    Columns = library terms     (1, z52, z27, Fs, ..., z52·Fs, ...)
    Color  = coefficient magnitude (diverging: blue=negative, red=positive)
    White  = exactly zero (sparse / thresholded-out terms)

    This gives the researcher an immediate visual of which physical interactions
    drive each latent dimension's dynamics.
    """
    coeff = model.coefficients()          # (n_state, n_terms)
    terms = model.get_feature_names()

    fig, ax = plt.subplots(
        figsize=(max(10, len(terms) * 0.55), 2.5 + len(state_names) * 0.8)
    )
    abs_max = max(np.abs(coeff).max(), 1e-6)
    im = ax.imshow(
        coeff,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-abs_max,
        vmax=+abs_max,
    )
    ax.set_xticks(range(len(terms)))
    ax.set_xticklabels(terms, rotation=75, ha="right", fontsize=7)
    ax.set_yticks(range(len(state_names)))
    ax.set_yticklabels([f"d{n}/dt" for n in state_names], fontsize=9)
    ax.set_title("SINDYc coefficient matrix Ξ  (white = zero / thresholded)")

    # Annotate non-zero cells
    for i in range(coeff.shape[0]):
        for j in range(coeff.shape[1]):
            val = coeff[i, j]
            if abs(val) > 0:
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=6, color="black")

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.tight_layout()
    out_path = out_dir / "sindyc_coefficients.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Coefficient heatmap saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SINDYc latent-space dynamics discovery")
    p.add_argument("--threshold",  type=float, default=0.08,
                   help="STLSQ sparsity threshold (default: 0.08)")
    p.add_argument("--degree",     type=int,   default=2,
                   help="Polynomial library degree (default: 2)")
    p.add_argument("--alpha",      type=float, default=0.05,
                   help="STLSQ L2 regularisation alpha (default: 0.05)")
    p.add_argument("--smoothing",  type=int,   default=5,
                   help="SmoothedFiniteDifference half-window steps (default: 5)")
    p.add_argument("--controls",   nargs="+",
                   default=list(CONTROL_COLS_MAP.values()),
                   help="Short names of control variables to include "
                        f"(choices: {list(CONTROL_COLS_MAP.values())})")
    p.add_argument("--eval-fault", action="store_true", default=True,
                   help="Evaluate on Fault/R4 batches after training (default: on)")
    p.add_argument("--eval-r1",    action="store_true", default=False,
                   help="Also evaluate on R1 train batches (default: off)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Filter requested control variables ───────────────────────────────────
    short_to_full = {v: k for k, v in CONTROL_COLS_MAP.items()}
    requested_ctrl_cols  = [short_to_full[s] for s in args.controls
                             if s in short_to_full]
    requested_ctrl_names = [s for s in args.controls if s in short_to_full]
    if not requested_ctrl_cols:
        raise ValueError(
            f"No valid control variables selected from {args.controls}. "
            f"Choices: {list(CONTROL_COLS_MAP.values())}"
        )

    print(f"\nControl variables ({len(requested_ctrl_cols)}): {requested_ctrl_names}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n── 1. Loading R1 (train) batches ...")
    r1_batches = load_batches_by_regime(CSV_PATH, R1_IDS)

    print("\n── 2. Loading Fault batches ...")
    fault_batches = load_batches_by_regime(CSV_PATH, FAULT_IDS)

    # ── 2. Load frozen β-VAE encoder ─────────────────────────────────────────
    print("\n── 3. Loading frozen β=0.05 VAE encoder ...")
    encoder = BetaVAERamanEncoderV2(
        ckpt_path=VAE_CKPT,
        scaler_path=VAE_SCALER,
        device=device,
    )

    # ── 3. Build Z trajectories ───────────────────────────────────────────────
    print("\n── 4. Encoding R1 batches → Z trajectories ...")
    Z_r1 = build_z_trajectories(r1_batches, encoder)

    print("\n── 5. Encoding Fault batches → Z trajectories ...")
    Z_fault = build_z_trajectories(fault_batches, encoder)

    # ── 4. Extract control variables ──────────────────────────────────────────
    print("\n── 6. Extracting control variables U from R1 ...")
    U_r1, ctrl_scaler = extract_control_variables(
        r1_batches,
        ctrl_cols=requested_ctrl_cols,
        ctrl_names=requested_ctrl_names,
        fit_scaler=True,
    )

    print("\n── 7. Applying fitted scaler to Fault batches ...")
    U_fault, _ = extract_control_variables(
        fault_batches,
        ctrl_cols=requested_ctrl_cols,
        ctrl_names=requested_ctrl_names,
        scaler=ctrl_scaler,
        fit_scaler=False,     # use the R1-fitted scaler — no data leakage
    )

    # ── 5. Build SINDYc model ─────────────────────────────────────────────────
    print("\n── 8. Building SINDYc model ...")
    sindy_model = build_sindyc_model(
        latent_names=LATENT_NAMES,
        control_names=requested_ctrl_names,
        degree=args.degree,
        threshold=args.threshold,
        alpha=args.alpha,
        smoothing_bw=args.smoothing,
    )

    # ── 6. Fit ────────────────────────────────────────────────────────────────
    print("\n── 9. Fitting SINDYc on R1 batches ...")
    sindy_model = fit_sindyc(
        sindy_model,
        Z_r1,
        U_r1,
        latent_names=LATENT_NAMES,
        control_names=requested_ctrl_names,
        dt=DT,
    )

    # ── 7. Print equations ────────────────────────────────────────────────────
    print_equations(sindy_model, state_names=LATENT_NAMES)

    # ── 8. Save coefficient heatmap ───────────────────────────────────────────
    print("\n── 10. Saving coefficient heatmap ...")
    plot_coefficient_heatmap(sindy_model, state_names=LATENT_NAMES, out_dir=OUT_DIR)

    # ── 9. Evaluate on Fault batches ──────────────────────────────────────────
    all_metrics = []

    if args.eval_fault:
        print("\n── 11. Evaluating on Fault/R4 batches ...")
        for i, (batch_df, Z_true, U_b) in enumerate(
            zip(fault_batches, Z_fault, U_fault)
        ):
            batch_global_id = 90 + i
            print(f"\n  Simulating Fault batch {batch_global_id + 1} (ID={batch_global_id}) ...")

            # Initial latent state from the β-VAE encoding of this batch
            Z0 = Z_true[0]   # (2,)  — [z52_0, z27_0]
            print(f"    Z0 = [z52={Z0[0]:.3f}, z27={Z0[1]:.3f}]")

            # Integrate Ż = f(Z, U) forward
            Z_pred = simulate_trajectory(sindy_model, Z0, U_b, dt=DT)

            # Evaluate + plot
            m = evaluate_and_plot(
                batch_df=batch_df,
                Z_true=Z_true,
                Z_pred=Z_pred,
                batch_id=batch_global_id,
                out_dir=OUT_DIR,
            )
            all_metrics.append(m)
            print(
                f"    RMSE z₅₂={m['rmse_z52']:.4f}  "
                f"z₂₇={m['rmse_z27']:.4f}  "
                f"Pen={m['rmse_pen_sparse']:.4f} g/L  "
                f"(n_sparse={m['n_sparse']})"
            )

    if args.eval_r1:
        print("\n── 12. Evaluating on R1 (train) batches ...")
        for i, (batch_df, Z_true, U_b) in enumerate(
            zip(r1_batches, Z_r1, U_r1)
        ):
            print(f"\n  Simulating R1 batch {i + 1} ...")
            Z0     = Z_true[0]
            Z_pred = simulate_trajectory(sindy_model, Z0, U_b, dt=DT)
            m = evaluate_and_plot(
                batch_df=batch_df,
                Z_true=Z_true,
                Z_pred=Z_pred,
                batch_id=i,
                out_dir=OUT_DIR,
            )
            all_metrics.append(m)

    # ── 10. Aggregate summary ─────────────────────────────────────────────────
    if all_metrics:
        pen_rmses = [m["rmse_pen_sparse"] for m in all_metrics
                     if not np.isnan(m["rmse_pen_sparse"])]
        mean_pen_rmse = float(np.mean(pen_rmses)) if pen_rmses else float("nan")

        print(f"\n── Summary ────────────────────────────────────────────────")
        print(f"  Batches evaluated   : {len(all_metrics)}")
        print(f"  Mean Pen RMSE       : {mean_pen_rmse:.4f} g/L")
        print(
            f"  Min/Max Pen RMSE    : "
            f"{min(pen_rmses):.4f} / {max(pen_rmses):.4f} g/L"
        )

        # Save JSON summary
        summary = {
            "config": {
                "degree":        args.degree,
                "threshold":     args.threshold,
                "alpha":         args.alpha,
                "smoothing_bw":  args.smoothing,
                "control_names": requested_ctrl_names,
                "readout_slope": READOUT_SLOPE,
                "readout_intercept": READOUT_INTERCEPT,
            },
            "equations": {
                state: model_eq
                for state, model_eq in zip(
                    LATENT_NAMES,
                    sindy_model.equations(precision=6),
                )
            },
            "mean_pen_rmse_gL": mean_pen_rmse,
            "batches": all_metrics,
        }
        summary_path = OUT_DIR / "sindyc_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n  Summary JSON → {summary_path}")

    print(f"\nDone. Outputs in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
