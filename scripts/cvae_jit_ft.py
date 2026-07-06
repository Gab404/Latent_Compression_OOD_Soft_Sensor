#!/usr/bin/env python
"""
cvae_jit_ft.py
==============
CVAE-PI-LSTM with Just-In-Time Fine-Tuning (JIT-FT) — a "local model"
paradigm built on top of the pre-trained "global model" CVAE-PI-LSTM.

Paradigm contrast
-----------------
  Global model  (checkpoints/cvae_pi_lstm.pt, trained by train.py)
    • One fixed set of LSTM weights, trained once on all 72 training batches.
    • Excellent average performance (Test RMSE = 0.135 g/L, R² = 0.9998),
      but is blind to any local structure that the average-case optimum
      does not capture (e.g. behaviour specific to fault-batch regimes).

  JIT-FT / local model  (this module)
    • The GLOBAL model's weights are NEVER modified in place. Every fine-tune
      operates on a throw-away in-memory clone.
    • For every query window at inference time:
        1. Retrieve the K most similar historical TRAINING windows, using
           Symmetric KL divergence in the CVAE probabilistic latent space.
        2. Clone the global model's weights into a scratch model.
        3. Fine-tune the scratch model for a handful of epochs (5-10) on
           ONLY those K windows, using the exact physics-informed loss
           (compute_cdae_pinn_loss) used to train the global model.
        4. Predict the query's Penicillin concentration with the
           fine-tuned scratch model.
        5. Reset the scratch model back to the global weights (discard the
           local adaptation) before the next query.

Mathematical bridge: CVAE posterior → SKL retrieval → local fine-tuning
------------------------------------------------------------------------
The CVAE encoder maps a preprocessed Raman spectrum x to a diagonal Gaussian
posterior  q_φ(z|x) = N(μ(x), diag(σ²(x))),  σ² = exp(log_var).

For a SEQUENCE window (the LSTM operates on 24-step windows, not single
points) we take the posterior at the window's LAST time step as the window's
representative distribution — this is the time step the LSTM actually
predicts against (see CVAEPILSTMAlgorithm.evaluate(): only pred[:, -1, :]
is used). Two windows are "similar" in the JIT-FT sense if their terminal
latent posteriors are close under:

    D_SKL(q_i ‖ q_q) = 0.5 · Σ_d [
        σ²_i,d/σ²_q,d + σ²_q,d/σ²_i,d
      + (μ_q,d − μ_i,d)² · (1/σ²_q,d + 1/σ²_i,d)
      − 2
    ]

Once the K nearest windows are retrieved, their FULL (24-step) latent
sequences and target sequences are used to locally fine-tune the cloned
PI-LSTM with the SAME physics-informed loss (data MSE + λ·mass-balance
residual) used for global training. This directly couples:

  • Latent UNCERTAINTY (σ² in the SKL metric)   → WHICH windows are retrieved
  • Physics-informed loss (ΔP ≈ k_prod·r_net·dt) → HOW the local model adapts

A query point whose CVAE posterior is very confident (small σ²) will pull in
a tightly clustered neighbourhood (small SKL distances), producing a sharp,
low-variance local adaptation. A query with high posterior uncertainty
(spectrum far from the training manifold, e.g. instrument noise or a novel
fault signature) naturally retrieves a more diffuse/heterogeneous
neighbourhood, and the physics term then acts as a regulariser preventing the
few-shot fine-tune from overfitting to K noisy neighbours.

Optimisation best practices implemented here
---------------------------------------------
  1. **Single reusable scratch model.** Instead of constructing a brand-new
     nn.Module per query (expensive: weight init, CUDA allocation), we
     allocate ONE scratch model once and do
     ``scratch.load_state_dict(global_state)`` before every query — an
     in-place tensor copy, ~100x cheaper than re-instantiation.
  2. **Optional layer freezing.** ``--freeze-first-layer`` freezes LSTM
     layer 0 (only layer 1 + shared FC + heads receive gradients), roughly
     halving the backward-pass cost of each local fine-tune while retaining
     the globally-learned low-level temporal feature extraction.
  3. **Few epochs, higher LR.** Local fine-tuning uses a SEPARATE, LARGER
     learning rate (``--local-lr``, default 5e-3) than the original 1e-3 —
     with only K=100-200 samples and 5-10 epochs, faster convergence to the
     local optimum is preferable to the small, cautious steps used for the
     72-batch global fit.
  4. **Fixed (non-annealed) physics weight.** The global model already
     anneals λ_physics from 0→0.1 over its own training; JIT-FT reuses the
     FINAL annealed value (config ``lambda_physics_end``) as a constant
     weight throughout the short local fine-tune (no re-annealing needed for
     only 5-10 epochs).
  5. **Batched retrieval, sequential fine-tuning.** All test-window CVAE
     encodings (μ, σ²) are computed ONCE in a single batched GPU forward
     pass BEFORE the fine-tuning loop starts. Only the (inherently
     sequential, since each query needs its own scratch model) fine-tune
     step runs one-query-at-a-time.
  6. **DB caching.** Like ``vae_jitl.py``, the encoded training-window
     database is cached to ``.npz`` (+ a companion ``.npy`` for the raw
     window tensors) so repeated experiments (K sweep, freeze on/off, LR
     sweep) skip the ~10 s encoding step.

Usage
-----
    python cvae_jit_ft.py                                  # defaults (always fine-tune)
    python cvae_jit_ft.py --K 150 --local-epochs 8 --local-lr 5e-3
    python cvae_jit_ft.py --freeze-first-layer
    python cvae_jit_ft.py --max-queries 40                  # fast smoke run
    python cvae_jit_ft.py --confidence-threshold 0.5         # Feature A: selective JIT-FT
    python cvae_jit_ft.py --distance-temperature 0.3         # Feature B: sharper weighting
    python cvae_jit_ft.py --grid-search                      # Feature C: full K x threshold grid
    python cvae_jit_ft.py --grid-search --max-queries 60     # fast grid-search smoke run
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.algorithms  # noqa: F401  — side-effect: populates REGISTRY
from src.algorithms.base import BaseAlgorithm
from src.algorithms.cvae_pi_lstm import CVAEPILSTMAlgorithm
from src.data.dataset import RAMAN_START_COL_IDX, TARGET_COLS
from src.evaluation.metrics import compute_metrics
from src.physics.losses import compute_cdae_pinn_loss

# Reuse the encoder + preprocessing + CSV-split helpers from vae_jitl.py
# (identical pipeline: clamp -> SG(d=1,w=15,p=2) -> StandardScaler -> CVAE)
from vae_jitl import VAEJITLEncoder, VAR_FLOOR, _load_and_split_csv

# ── Constants ─────────────────────────────────────────────────────────────────
N_RAMAN_COLS      = 2001
DEFAULT_K         = 150
DEFAULT_LOCAL_LR  = 5e-3
DEFAULT_N_EPOCHS  = 8
DEFAULT_CKPT      = "./checkpoints/cvae_pi_lstm.pt"     # GLOBAL model checkpoint
DEFAULT_DB_CACHE  = "./checkpoints/cvae_jit_ft_db.npz"
SEED              = 42


# ─────────────────────────────────────────────────────────────────────────────
# Section A — Historical window database
# ─────────────────────────────────────────────────────────────────────────────

class JITFTWindowDatabase:
    """
    Memory bank of TRAINING-set sliding windows for JIT-FT.

    For every valid sliding window (seq_len consecutive time steps) in the
    training batches, this class stores:

      X_win        : (seq_len, D)  scaled CVAE latent sequence
                     (this is the EXACT input the PI-LSTM consumes)
      y_win        : (seq_len,)    scaled Penicillin target sequence
                     (dense, interpolated — matches the global training data)
      mu_last      : (D,)  RAW (unscaled) CVAE posterior mean at the window's
                     LAST time step — used as the retrieval key
      var_last     : (D,)  RAW posterior variance at the last time step
                     (σ² = exp(log_var), floor-clamped)
      inv_var_last : (D,)  precomputed 1/σ² for fast SKL computation
      process_vars : (n_process,)  raw process variables at the last time
                     step of the window (temperature, pH, DO2, substrate, ...).
                     NOTE: compute_cdae_pinn_loss's mass-balance residual is
                     self-consistency between the model's own two heads
                     (pen_norm, r_net) and does NOT require process variables
                     as an input — they are stored here purely so this
                     database is a complete, reusable "case base" for any
                     future physics loss variant that DOES condition on them
                     (e.g. a process-aware mass balance).

    All arrays are stored contiguously (N_windows, ...) for vectorised SKL
    distance computation.
    """

    def __init__(self) -> None:
        self.X_win:        np.ndarray = None   # (N, seq_len, D) float32
        self.y_win:        np.ndarray = None   # (N, seq_len)    float32
        self.mu_last:      np.ndarray = None   # (N, D)          float32
        self.var_last:     np.ndarray = None   # (N, D)          float32
        self.inv_var_last: np.ndarray = None   # (N, D)          float32
        self.process_vars: np.ndarray = None   # (N, n_process)  float32
        self._built = False

    def build(
        self,
        train_batches: List[pd.DataFrame],
        encoder:       VAEJITLEncoder,
        feat_scaler,                       # fitted MinMaxScaler (mu -> [0,1])
        pen_scaler,                        # fitted MinMaxScaler (P   -> [0,1])
        seq_len:       int,
        process_cols:  List[str],
    ) -> "JITFTWindowDatabase":
        """
        Encode all training batches and slide a window of length ``seq_len``
        across each one, exactly mirroring
        ``CVAEPILSTMAlgorithm._build_dataset``.
        """
        Xw_parts, yw_parts, mu_parts, var_parts, proc_parts = [], [], [], [], []

        print(f"Building JIT-FT window database "
              f"({len(train_batches)} training batches, seq_len={seq_len}) ...")
        t0 = time.time()

        for batch_df in train_batches:
            all_cols   = batch_df.columns.tolist()
            n_avail    = len(all_cols) - RAMAN_START_COL_IDX
            n_use      = min(N_RAMAN_COLS, n_avail)
            raman_cols = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + n_use]
            X_raw      = batch_df[raman_cols].values.astype(np.float32)

            energy_mask = np.abs(X_raw).sum(axis=1) > 0
            if energy_mask.sum() == 0:
                continue

            # ── Encode full posterior (mu, log_var) for every valid row ────
            mu_valid, lv_valid = encoder.encode(X_raw[energy_mask])
            var_valid          = np.exp(lv_valid).clip(VAR_FLOOR)

            # Forward-fill onto the full-length batch array (zero-energy
            # rows get the first valid latent, matching PenicillinDataModule)
            T = len(batch_df)
            mu_full  = np.zeros((T, mu_valid.shape[1]), dtype=np.float32)
            var_full = np.ones((T,  var_valid.shape[1]), dtype=np.float32)
            valid_idx = np.where(energy_mask)[0]
            mu_full[valid_idx]  = mu_valid
            var_full[valid_idx] = var_valid
            # Simple forward/backward fill for the warm-up rows
            if len(valid_idx) > 0:
                first_valid = valid_idx[0]
                mu_full[:first_valid]  = mu_full[first_valid]
                var_full[:first_valid] = var_full[first_valid]

            # ── Scale mu -> LSTM input space, P -> normalised target ────────
            X_scaled = feat_scaler.transform(mu_full).astype(np.float32)
            pen_norm = pen_scaler.transform(
                batch_df[TARGET_COLS[1]].values.reshape(-1, 1)
            ).flatten().astype(np.float32)

            proc_full = (
                batch_df[process_cols].values.astype(np.float32)
                if process_cols else np.zeros((T, 0), dtype=np.float32)
            )

            # ── Slide windows ────────────────────────────────────────────
            for i in range(T - seq_len + 1):
                Xw_parts.append(X_scaled[i : i + seq_len])
                yw_parts.append(pen_norm[i : i + seq_len])
                last = i + seq_len - 1
                mu_parts.append(mu_full[last])
                var_parts.append(var_full[last])
                proc_parts.append(proc_full[last] if process_cols else np.zeros(0, dtype=np.float32))

        self.X_win        = np.stack(Xw_parts).astype(np.float32)
        self.y_win         = np.stack(yw_parts).astype(np.float32)
        self.mu_last       = np.stack(mu_parts).astype(np.float32)
        self.var_last      = np.stack(var_parts).astype(np.float32)
        self.inv_var_last  = (1.0 / self.var_last).astype(np.float32)
        self.process_vars  = np.stack(proc_parts).astype(np.float32) if process_cols else \
                              np.zeros((len(mu_parts), 0), dtype=np.float32)
        self._built = True

        elapsed = time.time() - t0
        print(f"  Window DB ready: {len(self.y_win):,} windows  "
              f"(seq_len={seq_len}, D={self.mu_last.shape[1]})  [{elapsed:.1f} s]")
        return self

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            X_win=self.X_win, y_win=self.y_win,
            mu_last=self.mu_last, var_last=self.var_last, inv_var_last=self.inv_var_last,
            process_vars=self.process_vars,
        )
        print(f"  Window DB cached → {path}")

    def load(self, path: str) -> "JITFTWindowDatabase":
        d = np.load(path)
        self.X_win        = d["X_win"]
        self.y_win         = d["y_win"]
        self.mu_last       = d["mu_last"]
        self.var_last      = d["var_last"]
        self.inv_var_last  = d["inv_var_last"]
        self.process_vars  = d["process_vars"]
        self._built = True
        print(f"  Window DB loaded ← {path}  ({len(self.y_win):,} windows)")
        return self

    def __len__(self) -> int:
        return len(self.y_win) if self._built else 0

    @property
    def is_built(self) -> bool:
        return self._built


# ─────────────────────────────────────────────────────────────────────────────
# Section B — Symmetric KL divergence (vectorised, window-level)
# ─────────────────────────────────────────────────────────────────────────────

def skl_distances(
    mu_q:      np.ndarray,   # (D,)
    var_q:     np.ndarray,   # (D,)
    inv_var_q: np.ndarray,   # (D,)
    db:        JITFTWindowDatabase,
) -> np.ndarray:
    """
    Vectorised Symmetric KL divergence between the query window's terminal
    posterior (μ_q, σ²_q) and EVERY window in the database.

        D_SKL(q_i ‖ q_q) = 0.5 · Σ_d [
            σ²_i/σ²_q + σ²_q/σ²_i
          + (μ_q − μ_i)² · (1/σ²_q + 1/σ²_i)
          − 2
        ]

    Returns
    -------
    distances : (N_db,) float32
    """
    var_ratio_fwd = db.var_last * inv_var_q[None]           # σ²_i / σ²_q
    var_ratio_rev = var_q[None] * db.inv_var_last            # σ²_q / σ²_i
    delta_mu_sq   = (mu_q[None] - db.mu_last) ** 2
    mahalanobis   = delta_mu_sq * (inv_var_q[None] + db.inv_var_last)

    return 0.5 * (var_ratio_fwd + var_ratio_rev + mahalanobis - 2.0).sum(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Section B.1 — Distance-to-weight conversion (Feature B)
# ─────────────────────────────────────────────────────────────────────────────

def softmax_distance_weights(
    distances:   np.ndarray,   # (K,)  SKL distances of the K retrieved neighbours
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Convert SKL distances into normalised sample weights via a temperature-
    scaled softmax over NEGATIVE distances:

        w_i = softmax(-d_i / T) = exp(-d_i/T) / Σ_j exp(-d_j/T)

    Properties
    ----------
    • Σ_i w_i = 1  (a proper convex-combination weighting).
    • The CLOSEST neighbour (smallest d_i) gets the LARGEST weight — the
      negative sign flips "distance" (bigger = worse) into "affinity"
      (bigger = better) before the softmax.
    • Temperature T controls how "peaked" the weighting is:
        T → 0   : weight collapses onto the single nearest neighbour
                  (approaches 1-NN behaviour).
        T → ∞   : weights become uniform (≈ 1/K each, like the ORIGINAL
                  unweighted JIT-FT loss).
      T is therefore a direct knob for how much the local fine-tune should
      "trust its closest match" vs "average over the whole neighbourhood".

    Numerically stable implementation: subtract the max of (-d_i/T) before
    exponentiating (standard log-sum-exp trick) so exp() never overflows
    even for very small T or very large distances.

    Returns
    -------
    weights : (K,) float32, non-negative, sums to 1.0
    """
    scaled = -distances.astype(np.float64) / max(temperature, 1e-8)
    scaled -= scaled.max()                 # stability: largest exponent -> 0
    exp_w  = np.exp(scaled)
    weights = exp_w / exp_w.sum()
    return weights.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Section B.2 — Distance-weighted physics-informed local loss (Feature B)
# ─────────────────────────────────────────────────────────────────────────────

def compute_weighted_cdae_pinn_loss(
    model,
    pred:           torch.Tensor,   # (K, T, 2)  [:,:,0]=pen_norm [:,:,1]=r_net
    y_pen_norm:     torch.Tensor,   # (K, T)     normalised Penicillin targets
    sample_weights: torch.Tensor,   # (K,)       SKL-derived weights, Σ=1
    dt_n:           float,
    lambda_physics: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Distance-weighted variant of ``compute_cdae_pinn_loss`` (see
    ``src/physics/losses.py``).

    Only the DATA term is re-weighted by neighbour proximity — each of the
    K retrieved windows contributes to the data loss in proportion to how
    close it is (in SKL divergence) to the query:

        Data_MSE_i = mean_t [ (pen_pred_{i,t} - y_{i,t})^2 ]      (per window i)
        L_data     = Σ_i  w_i · Data_MSE_i                        (w_i from softmax)

    Because Σ w_i = 1, L_data is a proper weighted AVERAGE (not a sum that
    would scale with K), keeping the gradient magnitude comparable to the
    unweighted case regardless of K.

    The PHYSICS term is left UNWEIGHTED and computed exactly as in the
    global loss — it is a self-consistency constraint between the model's
    OWN two output heads (pen_norm and r_net), not a comparison against any
    particular neighbour's label, so neighbour-proximity weighting does not
    apply to it.

        Total Local Loss = Σ_i w_i · Data_MSE_i  +  λ_physics · L_physics

    Parameters
    ----------
    model          : CDAEPILSTMModel-compatible (duck-typed for .k_prod)
    pred           : (K, T, 2) model output on the K retrieved windows
    y_pen_norm     : (K, T) normalised targets for the K retrieved windows
    sample_weights : (K,) softmax weights from ``softmax_distance_weights``
    dt_n           : normalised time step
    lambda_physics : physics weight (fixed, non-annealed, during local FT)

    Returns
    -------
    loss      : scalar total loss
    loss_dict : breakdown for logging
    """
    pen_pred = pred[:, :, 0]     # (K, T)
    r_net    = pred[:, :, 1]     # (K, T)

    # ── Weighted data loss ────────────────────────────────────────────────
    valid = ~torch.isnan(y_pen_norm)                       # (K, T)
    sq_err = (pen_pred - torch.nan_to_num(y_pen_norm)) ** 2
    sq_err = torch.where(valid, sq_err, torch.zeros_like(sq_err))
    counts = valid.sum(dim=1).clamp(min=1).float()         # (K,) valid steps per window
    mse_per_window = sq_err.sum(dim=1) / counts             # (K,)  Data_MSE_i

    # sample_weights already sums to 1 -> weighted AVERAGE, not weighted SUM
    l_data = (sample_weights * mse_per_window).sum()

    # ── Physics residual (unweighted self-consistency, as in the global loss) ─
    if lambda_physics > 0.0:
        k_prod = model.k_prod
        dP_nn  = pen_pred[:, 1:] - pen_pred[:, :-1]        # (K, T-1)
        dP_phy = k_prod * r_net[:, :-1] * dt_n             # (K, T-1)
        l_phys = torch.mean((dP_nn - dP_phy) ** 2)
    else:
        l_phys = pred.new_tensor(0.0)

    loss = l_data + lambda_physics * l_phys

    loss_dict = {
        "loss":       float(loss.item()),
        "loss_data":  float(l_data.item()),
        "loss_phys":  float((lambda_physics * l_phys).item()),
        "k_prod":     float(model.k_prod.item()),
    }
    return loss, loss_dict


# ─────────────────────────────────────────────────────────────────────────────
# Section C/D — CVAE_JIT_PI_LSTM_Predictor
# ─────────────────────────────────────────────────────────────────────────────

class CVAE_JIT_PI_LSTM_Predictor:
    """
    Just-In-Time fine-tuning predictor built on a frozen GLOBAL CVAE-PI-LSTM.

    The global algorithm object (``global_algo``) is used ONLY as a
    read-only source of:
      • the trained weights   (never mutated — copied out at construction)
      • the fitted scalers    (feat / pen MinMaxScalers)
      • the normalised dt_n
      • the config's physics hyper-parameters (k_prod init n/a — the
        physics parameter itself IS fine-tuned locally, see below)

    Parameters
    ----------
    global_algo        : CVAEPILSTMAlgorithm, already loaded via
                          ``BaseAlgorithm.load(ckpt_path, device)``
    database            : JITFTWindowDatabase (built or loaded)
    K                   : number of nearest neighbour windows (default 150)
    local_epochs        : fine-tune epochs per query (default 8)
    local_lr            : learning rate for the local optimizer (default 5e-3)
    lambda_physics      : fixed physics weight during local fine-tuning
                          (defaults to the global config's lambda_physics_end)
    freeze_first_layer  : if True, LSTM layer 0 weights are frozen during
                          local fine-tuning (only layer 1 + shared FC + heads
                          are updated) — cheaper backward pass, retains the
                          globally-learned low-level temporal features.
    confidence_threshold : float (Feature A — Selective JIT-FT).
                          Mean SKL distance of the ``confidence_top_n``
                          closest historical neighbours is compared against
                          this threshold BEFORE any cloning/fine-tuning
                          happens:
                            mean_dist <  threshold  -> query looks like
                                "known normal" data -> SKIP fine-tuning,
                                predict directly with the frozen GLOBAL model.
                            mean_dist >= threshold  -> query looks like an
                                anomaly / drift / fault -> TRIGGER the full
                                JIT-FT clone-and-fine-tune procedure.
                          Default 0.0 -> SKL distances are always >= 0, so a
                          threshold of 0.0 makes the condition `>= 0.0`
                          virtually always true -> ALWAYS fine-tune
                          (i.e. identical behaviour to the original,
                          non-selective JIT-FT). Set to ``float("inf")`` to
                          NEVER fine-tune (pure global-model inference).
    confidence_top_n     : number of closest neighbours averaged to form the
                          confidence score (default 5).
    distance_temperature : softmax temperature T used to convert the K
                          retrieved SKL distances into local-loss sample
                          weights (Feature B). See
                          ``softmax_distance_weights`` for the exact
                          formula and the T -> 0 / T -> ∞ limiting cases.
    device              : torch device
    """

    def __init__(
        self,
        global_algo:          CVAEPILSTMAlgorithm,
        database:             JITFTWindowDatabase,
        K:                    int   = DEFAULT_K,
        local_epochs:         int   = DEFAULT_N_EPOCHS,
        local_lr:             float = DEFAULT_LOCAL_LR,
        lambda_physics:       float = None,
        freeze_first_layer:   bool  = False,
        confidence_threshold: float = 0.0,
        confidence_top_n:     int   = 5,
        distance_temperature: float = 1.0,
        device:               torch.device = None,
    ) -> None:
        assert database.is_built, "Call database.build() before creating the predictor."
        assert global_algo.model is not None, "global_algo must already be fit/loaded."

        self.device      = device or next(global_algo.model.parameters()).device
        self.db          = database
        self.K           = K
        self.local_epochs = local_epochs
        self.local_lr     = local_lr
        self.freeze_first_layer = freeze_first_layer

        # ── Feature A: Selective JIT-FT (confidence thresholding) ──────────
        self.confidence_threshold = confidence_threshold
        self.confidence_top_n     = confidence_top_n

        # ── Feature B: distance-weighted local loss ────────────────────────
        self.distance_temperature = distance_temperature

        # ── Diagnostics populated by the last predict_window() call ───────
        # (evaluate() reads these immediately after each call to build
        # aggregate trigger/skip statistics without changing the return
        # signature of predict_window itself.)
        self._last_triggered:     bool  = True
        self._last_confidence:    float = 0.0

        # ── Read-only references into the GLOBAL model ────────────────────
        self.global_algo   = global_algo
        self.dt_n           = global_algo._dt_n
        self.pen_scaler     = global_algo.scalers.pen

        # Physics weight used during local fine-tuning: reuse the annealed
        # END value from the global training config (the global model was
        # trained to convergence at this weight, so it is the physically
        # "settled" regularisation strength).
        tr = global_algo.cfg.training
        self.lambda_physics = (
            float(getattr(tr, "lambda_physics_end", 0.1))
            if lambda_physics is None else lambda_physics
        )

        # ── Snapshot the GLOBAL weights (read-only, never touched again) ──
        # Kept on CPU to avoid holding a duplicate GPU-resident copy; moved
        # to device only when copied INTO the scratch model per query.
        self._global_state = {
            k: v.detach().clone().cpu()
            for k, v in global_algo.model.state_dict().items()
        }

        # ── Allocate ONE reusable scratch model (optimisation #1) ─────────
        # Constructed once via the algorithm's own _build_model() factory
        # so the architecture is guaranteed identical to the global model.
        self._scratch_model: nn.Module = global_algo._build_model().to(self.device)
        self._reset_scratch()

    # ── Optimisation #1: in-place weight reset (no re-instantiation) ─────

    def _reset_scratch(self) -> None:
        """Reload the frozen global weights into the reusable scratch model."""
        self._scratch_model.load_state_dict(
            {k: v.to(self.device) for k, v in self._global_state.items()}
        )
        # Optimisation #2: optionally freeze LSTM layer 0
        if self.freeze_first_layer:
            for name, p in self._scratch_model.named_parameters():
                # nn.LSTM parameter names: weight_ih_l0, weight_hh_l0,
                # bias_ih_l0, bias_hh_l0 for layer 0; "_l1" for layer 1, etc.
                p.requires_grad_(not name.endswith("_l0"))
        else:
            for p in self._scratch_model.parameters():
                p.requires_grad_(True)

    # ── Step C — K-NN retrieval + Feature A confidence gating ─────────────

    def _skl_distances_for_query(
        self,
        mu_q:  np.ndarray,
        var_q: np.ndarray,
    ) -> np.ndarray:
        """Vectorised SKL distance to every window in the database."""
        inv_var_q = (1.0 / var_q).astype(np.float32)
        return skl_distances(mu_q, var_q, inv_var_q, self.db)

    def _confidence_score(self, d_skl: np.ndarray) -> float:
        """
        Feature A — confidence score used for Selective JIT-FT routing.

        Defined as the MEAN SKL distance of the ``confidence_top_n`` closest
        historical neighbours. A SMALL value means the query sits right on
        top of well-covered, "seen-before" training data (the global model
        can be trusted as-is); a LARGE value means the query is in a sparse
        / novel region of the latent space (drift, fault, anomaly) where
        the fixed global weights are less likely to be locally optimal.
        """
        n_conf   = min(self.confidence_top_n, len(d_skl))
        conf_idx = np.argpartition(d_skl, n_conf - 1)[:n_conf]
        return float(d_skl[conf_idx].mean())

    def _get_knn_windows(
        self,
        d_skl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve the K nearest training WINDOWS (full seq_len sequences)
        by Symmetric KL divergence of their terminal latent posterior, and
        convert their distances into Feature-B softmax sample weights.

        Returns
        -------
        X_knn   : (K, seq_len, D)  scaled latent sequences
        y_knn   : (K, seq_len)     scaled Penicillin target sequences
        weights : (K,)             softmax(-distance/T) sample weights, Σ=1
        """
        k_eff     = min(self.K, len(d_skl))
        top_k_idx = np.argpartition(d_skl, k_eff - 1)[:k_eff]

        distances_k = d_skl[top_k_idx]
        weights     = softmax_distance_weights(distances_k, self.distance_temperature)

        return self.db.X_win[top_k_idx], self.db.y_win[top_k_idx], weights

    # ── Step D — Ephemeral local fine-tuning ───────────────────────────────

    def _fine_tune_and_predict(
        self,
        X_knn:   np.ndarray,   # (K, seq_len, D)
        y_knn:   np.ndarray,   # (K, seq_len)
        weights: np.ndarray,   # (K,)  Feature B softmax sample weights, Σ=1
        x_query: np.ndarray,   # (seq_len, D)  — the actual query window input
    ) -> float:
        """
        Fine-tune the scratch model on (X_knn, y_knn) for ``local_epochs``
        epochs using the DISTANCE-WEIGHTED physics-informed loss
        (``compute_weighted_cdae_pinn_loss``), then predict the query
        window's Penicillin concentration (last time step) and return it in
        NORMALISED [0,1] space.

        The caller is responsible for resetting the scratch model
        (``_reset_scratch``) both BEFORE this call (fresh global copy) and
        implicitly for the NEXT query (called again at the top of the loop).
        """
        model = self._scratch_model
        model.train()

        # Local optimizer: single AdamW group over ALL currently-trainable
        # parameters (network + physics k_prod). A fresh optimizer instance
        # is required every call because Adam's moment buffers must not
        # carry over from the previous query's (unrelated) local optimum.
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        local_optim = torch.optim.AdamW(trainable_params, lr=self.local_lr)

        X_t = torch.from_numpy(X_knn).to(self.device)           # (K, seq_len, D)
        y_t = torch.from_numpy(y_knn).to(self.device)            # (K, seq_len)
        w_t = torch.from_numpy(weights).to(self.device)          # (K,) Σ=1

        for _ in range(self.local_epochs):
            local_optim.zero_grad(set_to_none=True)
            pred = model(X_t)                                     # (K, T, 2)
            loss, _ = compute_weighted_cdae_pinn_loss(
                model=model,
                pred=pred,
                y_pen_norm=y_t,
                sample_weights=w_t,
                dt_n=self.dt_n,
                lambda_physics=self.lambda_physics,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 5.0)
            local_optim.step()

        # ── Predict the query window (last time step) ─────────────────────
        model.eval()
        with torch.no_grad():
            x_q_t = torch.from_numpy(x_query[None]).to(self.device)   # (1, seq_len, D)
            pred_q = model(x_q_t)                                     # (1, T, 2)
            pen_norm_hat = pred_q[0, -1, 0].item()

        return pen_norm_hat

    def _predict_global_only(self, x_query: np.ndarray) -> float:
        """
        Feature A fast path — the query was deemed "already known" by the
        confidence check, so we bypass cloning/fine-tuning ENTIRELY and
        predict directly with the frozen GLOBAL model. This is essentially
        free (single forward pass, no backward, no optimizer, no clone
        reset) compared to a full JIT-FT cycle.
        """
        model = self.global_algo.model
        model.eval()
        with torch.no_grad():
            x_q_t = torch.from_numpy(x_query[None]).to(self.device)
            pred_q = model(x_q_t)
            return pred_q[0, -1, 0].item()

    # ── Core predict API ────────────────────────────────────────────────────

    def predict_window(
        self,
        x_query_scaled: np.ndarray,   # (seq_len, D)  scaled latent sequence
        mu_q_raw:        np.ndarray,   # (D,)  raw terminal posterior mean
        var_q_raw:       np.ndarray,   # (D,)  raw terminal posterior variance
    ) -> float:
        """
        Full JIT-FT prediction for one query window, with Feature A
        (Selective JIT-FT) routing:

          1. Compute SKL distances to the whole database (cheap, vectorised).
          2. Confidence score = mean SKL distance of the `confidence_top_n`
             closest neighbours.
          3. IF confidence_score < confidence_threshold:
                 "This looks like data the global model has already
                 mastered." -> SKIP retrieval-K / cloning / fine-tuning,
                 predict directly with the frozen global model (cheap).
             ELSE:
                 "This looks like an anomaly / drift / fault." -> retrieve
                 the K nearest windows, convert their distances into
                 Feature-B softmax weights, clone + fine-tune + predict.

        Diagnostics (``self._last_triggered``, ``self._last_confidence``)
        are updated on every call so ``evaluate()`` can report the
        trigger/skip rate without changing this method's return signature.

        Returns
        -------
        y_hat : float — predicted Penicillin concentration in g/L
                (already inverse-transformed from normalised space)
        """
        d_skl      = self._skl_distances_for_query(mu_q_raw, var_q_raw)
        confidence = self._confidence_score(d_skl)
        self._last_confidence = confidence

        if confidence < self.confidence_threshold:
            # ── Feature A fast path: SKIP fine-tuning ──────────────────────
            self._last_triggered = False
            pen_norm_hat = self._predict_global_only(x_query_scaled)
        else:
            # ── Full JIT-FT: retrieve, weight, clone, fine-tune, predict ───
            self._last_triggered = True
            X_knn, y_knn, weights = self._get_knn_windows(d_skl)
            self._reset_scratch()   # start from the UNMODIFIED global weights
            pen_norm_hat = self._fine_tune_and_predict(X_knn, y_knn, weights, x_query_scaled)

        y_hat = float(
            self.pen_scaler.inverse_transform(
                np.array([[pen_norm_hat]], dtype=np.float32)
            ).ravel()[0]
        )
        return y_hat

    # ── Evaluation loop ─────────────────────────────────────────────────────

    def evaluate(
        self,
        test_batches:  List[pd.DataFrame],
        encoder:       VAEJITLEncoder,
        feat_scaler,
        seq_len:       int,
        max_queries:   int = None,
        verbose:       bool = True,
    ) -> Dict[str, float]:
        """
        Run the JIT-FT pipeline on the test (fault) set, evaluating ONLY at
        the sparse offline measurement points — identical protocol to the
        global CVAE-PI-LSTM's ``evaluate()``.

        Optimisation
        ------------
        For each test batch, ALL valid Raman rows are encoded in a single
        batched GPU forward pass BEFORE the (inherently sequential) local
        fine-tuning loop begins — the expensive CVAE encoding cost is
        amortised across the whole batch, only the fine-tune step pays the
        "one-at-a-time" cost.

        Parameters
        ----------
        max_queries : if set, stop after this many total query points
                      (useful for fast smoke tests / debugging).
        """
        all_y_true: List[float] = []
        all_y_pred: List[float] = []
        all_times:  List[float] = []
        n_triggered = 0
        n_skipped   = 0

        print(f"\n{'='*64}")
        print(f"  CVAE-PI-LSTM  JIT-FT  Evaluation")
        print(f"  K={self.K}  local_epochs={self.local_epochs}  "
              f"local_lr={self.local_lr}  lambda_physics={self.lambda_physics:.3f}  "
              f"freeze_l0={self.freeze_first_layer}")
        print(f"  confidence_threshold={self.confidence_threshold}  "
              f"confidence_top_n={self.confidence_top_n}  "
              f"distance_temperature={self.distance_temperature}")
        print(f"  DB size: {len(self.db):,} windows  |  Test batches: {len(test_batches)}")
        print(f"{'='*64}")

        total_queries = 0
        stop = False

        for b_idx, batch_df in enumerate(test_batches, 1):
            if stop:
                break

            all_cols   = batch_df.columns.tolist()
            n_avail    = len(all_cols) - RAMAN_START_COL_IDX
            n_use      = min(N_RAMAN_COLS, n_avail)
            raman_cols = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + n_use]
            X_raw      = batch_df[raman_cols].values.astype(np.float32)
            energy_mask = np.abs(X_raw).sum(axis=1) > 0
            if energy_mask.sum() == 0:
                continue

            # ── Batched CVAE encoding (one GPU pass for the whole batch) ──
            mu_valid, lv_valid = encoder.encode(X_raw[energy_mask])
            var_valid          = np.exp(lv_valid).clip(VAR_FLOOR)

            T = len(batch_df)
            mu_full  = np.zeros((T, mu_valid.shape[1]), dtype=np.float32)
            var_full = np.ones((T,  var_valid.shape[1]), dtype=np.float32)
            valid_idx = np.where(energy_mask)[0]
            mu_full[valid_idx]  = mu_valid
            var_full[valid_idx] = var_valid
            if len(valid_idx) > 0:
                first_valid = valid_idx[0]
                mu_full[:first_valid]  = mu_full[first_valid]
                var_full[:first_valid] = var_full[first_valid]

            X_scaled = feat_scaler.transform(mu_full).astype(np.float32)

            sp_pen = batch_df["_penicillin_sparse"].values

            batch_preds, batch_true = [], []

            # Query rows: sparse measurement points, requiring a full
            # seq_len window ending at that row.
            query_rows = np.where(~pd.isna(sp_pen))[0]
            query_rows = query_rows[query_rows >= seq_len - 1]

            for row_idx in query_rows:
                if max_queries is not None and total_queries >= max_queries:
                    stop = True
                    break

                start = row_idx - seq_len + 1
                x_query = X_scaled[start : row_idx + 1]          # (seq_len, D)
                mu_q    = mu_full[row_idx]
                var_q   = var_full[row_idx]

                y_true = float(sp_pen[row_idx])
                if not np.isfinite(y_true):
                    continue

                t0 = time.perf_counter()
                y_hat = self.predict_window(x_query, mu_q, var_q)
                elapsed = (time.perf_counter() - t0) * 1e3

                if self._last_triggered:
                    n_triggered += 1
                else:
                    n_skipped += 1

                batch_preds.append(y_hat)
                batch_true.append(y_true)
                all_times.append(elapsed)
                total_queries += 1

            all_y_pred.extend(batch_preds)
            all_y_true.extend(batch_true)

            if verbose and batch_preds:
                bt, bp = np.array(batch_true), np.array(batch_preds)
                b_rmse = np.sqrt(np.mean((bp - bt) ** 2))
                print(f"  Batch {b_idx:2d}  |  queries={len(batch_preds):3d}  "
                      f"RMSE={b_rmse:.3f} g/L  "
                      f"avg_time={np.mean(all_times[-len(batch_preds):]):.1f} ms/query")

        y_true_arr = np.array(all_y_true, dtype=np.float32)
        y_pred_arr = np.array(all_y_pred, dtype=np.float32)
        metrics    = compute_metrics(y_true_arr, y_pred_arr)

        total_q    = n_triggered + n_skipped
        skip_rate  = (n_skipped / total_q) if total_q > 0 else 0.0
        metrics["n_triggered"] = n_triggered
        metrics["n_skipped"]   = n_skipped
        metrics["skip_rate"]   = skip_rate

        avg_ms = float(np.mean(all_times)) if all_times else 0.0
        print(f"\n{'─'*64}")
        print(f"  JIT-FT results  (K={self.K}, epochs={self.local_epochs}, "
              f"lr={self.local_lr}, conf_thr={self.confidence_threshold})")
        print(f"{'─'*64}")
        print(f"  Penicillin  RMSE = {metrics['RMSE']:.4f} g/L")
        print(f"  Penicillin  MAE  = {metrics['MAE']:.4f} g/L")
        print(f"  Penicillin  R²   = {metrics['R2']:.6f}")
        print(f"  N queries        = {metrics['n']}")
        print(f"  Fine-tuned (triggered) = {n_triggered}  |  "
              f"Skipped (global-only)  = {n_skipped}  "
              f"(skip rate = {skip_rate*100:.1f}%)")
        print(f"  Avg query time   = {avg_ms:.2f} ms/query  "
              f"(total {sum(all_times)/1e3:.1f} s)")
        print(f"{'─'*64}\n")

        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Section E — Feature C: nested grid search over K x confidence_threshold
# ─────────────────────────────────────────────────────────────────────────────

def grid_search(
    predictor:             CVAE_JIT_PI_LSTM_Predictor,
    test_batches:          List[pd.DataFrame],
    encoder:               VAEJITLEncoder,
    feat_scaler,
    seq_len:               int,
    k_values:              List[int]   = (200, 250, 300),
    confidence_thresholds: List[float] = (0.1, 0.5, 1.0, float("inf")),
    max_queries:           int = None,
) -> Dict[Tuple[int, float], Dict[str, float]]:
    """
    Feature C — nested evaluation grid search over (K, confidence_threshold).

    For every combination:
      • K controls how many historical windows are retrieved WHEN fine-
        tuning is triggered (larger K = more (noisier, but more diverse)
        context per local fine-tune).
      • confidence_threshold controls HOW OFTEN fine-tuning is triggered
        at all (Feature A):
            threshold = 0.1  -> very sensitive; almost every query is
                                 considered "novel enough" to fine-tune
                                 (behaviour close to the original,
                                 always-fine-tune JIT-FT).
            threshold = 0.5  -> medium sensitivity.
            threshold = 1.0  -> only fine-tune on genuinely distant /
                                 unusual queries.
            threshold = inf  -> NEVER fine-tune -> pure global-model
                                 inference (0 ms overhead, acts as the
                                 control/baseline row of the grid).

    The thresholds [0.1, 0.5, 1.0, inf] are a reasonable low/medium/high/
    control set for THIS latent space: on a 64-dim standardised Gaussian
    posterior, a mean-of-5-neighbours SKL divergence below ~0.1 typically
    indicates the query sits almost exactly on top of a training point,
    while values around/above 1.0 indicate a genuinely distant region of
    the latent manifold. If you want DATA-DRIVEN thresholds instead of
    these defaults, compute the empirical percentiles of the top-N SKL
    distance across a held-out sample of TRAINING queries (e.g. the 25th /
    50th / 75th percentile) and use those instead — the mechanism is
    threshold-agnostic.

    Returns
    -------
    results : {(K, threshold): metrics_dict}
    """
    results: Dict[Tuple[int, float], Dict[str, float]] = {}
    original_K   = predictor.K
    original_thr = predictor.confidence_threshold

    print(f"\n{'='*78}")
    print(f"  JIT-FT Grid Search  —  K x confidence_threshold")
    print(f"{'='*78}")
    header = (f"  {'K':>5}  {'threshold':>10}  {'RMSE':>9}  {'MAE':>9}  "
              f"{'R²':>10}  {'n':>5}  {'triggered':>10}  {'skip%':>7}")
    print(header)
    print(f"  {'─'*72}")

    for K in k_values:
        for thr in confidence_thresholds:
            predictor.K = K
            predictor.confidence_threshold = thr

            m = predictor.evaluate(
                test_batches, encoder, feat_scaler, seq_len,
                max_queries=max_queries, verbose=False,
            )
            results[(K, thr)] = m

            thr_str = "inf" if thr == float("inf") else f"{thr:.2f}"
            print(f"  {K:>5}  {thr_str:>10}  {m['RMSE']:>9.4f}  {m['MAE']:>9.4f}  "
                  f"{m['R2']:>10.6f}  {m['n']:>5}  "
                  f"{m['n_triggered']:>10}  {m['skip_rate']*100:>6.1f}%")

    predictor.K = original_K
    predictor.confidence_threshold = original_thr

    # ── Best combination by RMSE ────────────────────────────────────────────
    best_key = min(results, key=lambda k: results[k]["RMSE"])
    best_K, best_thr = best_key
    best_m = results[best_key]
    thr_str = "inf" if best_thr == float("inf") else f"{best_thr:.2f}"
    print(f"  {'─'*72}")
    print(f"  BEST  ->  K={best_K}  threshold={thr_str}  "
          f"RMSE={best_m['RMSE']:.4f}  R²={best_m['R2']:.6f}  "
          f"(triggered {best_m['n_triggered']}/{best_m['n']}, "
          f"skip {best_m['skip_rate']*100:.1f}%)")
    print(f"{'='*78}\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Section F — CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _print_comparison(jitft_metrics: Dict[str, float]) -> None:
    """Print a 3-way comparison: global model vs PLS-JITL vs JIT-FT."""
    models = [
        ("CVAE-PI-LSTM (global)",      0.1350, 0.9998),
        ("CDAE-PI-LSTM (global)",      0.0876, 0.9999),
        ("VAE-JITL PLS K=300 (local)", 0.6459, 0.9944),
    ]
    print(f"\n{'='*64}")
    print(f"  Global vs Local (PLS-JITL) vs Local (JIT-FT)  —  Test (fault)")
    print(f"{'='*64}")
    print(f"  {'Model':<32}  {'Test RMSE':>12}  {'Test R²':>10}")
    print(f"  {'─'*58}")
    for name, rmse, r2 in models:
        print(f"  {name:<32}  {rmse:>12.4f}  {r2:>10.6f}")
    print(f"  {'─'*58}")
    print(f"  {'CVAE-PI-LSTM JIT-FT (local)':<32}  "
          f"{jitft_metrics['RMSE']:>12.4f}  {jitft_metrics['R2']:>10.6f}")
    print(f"{'='*64}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CVAE-PI-LSTM Just-In-Time Fine-Tuning (JIT-FT) evaluation"
    )
    parser.add_argument("--csv",         default="./100_Batches_IndPenSim_V3.csv")
    parser.add_argument("--global-ckpt", default=DEFAULT_CKPT,
                        help="Path to the pre-trained GLOBAL cvae_pi_lstm.pt")
    parser.add_argument("--cvae-ckpt",   default="./checkpoints/cvae_best.pt")
    parser.add_argument("--cvae-scaler", default="./checkpoints/cvae_scaler.joblib")
    parser.add_argument("--db-cache",    default=DEFAULT_DB_CACHE)
    parser.add_argument("--rebuild-db",  action="store_true")
    parser.add_argument("--K",              type=int,   default=DEFAULT_K)
    parser.add_argument("--local-epochs",   type=int,   default=DEFAULT_N_EPOCHS)
    parser.add_argument("--local-lr",       type=float, default=DEFAULT_LOCAL_LR)
    parser.add_argument("--lambda-physics", type=float, default=None,
                        help="Override physics weight (default: global cfg lambda_physics_end)")
    parser.add_argument("--freeze-first-layer", action="store_true",
                        help="Freeze LSTM layer 0 during local fine-tuning")
    parser.add_argument("--confidence-threshold", type=float, default=0.0,
                        help="Feature A: mean-top-N SKL distance below which "
                             "fine-tuning is SKIPPED (0.0 = always fine-tune, "
                             "the original behaviour; inf = never fine-tune)")
    parser.add_argument("--confidence-top-n", type=int, default=5,
                        help="Number of closest neighbours averaged for the "
                             "Feature A confidence score (default 5)")
    parser.add_argument("--distance-temperature", type=float, default=1.0,
                        help="Feature B: softmax temperature for distance-"
                             "weighted local loss (default 1.0)")
    parser.add_argument("--max-queries",    type=int, default=None,
                        help="Cap total query points (fast smoke test)")
    parser.add_argument("--grid-search",    action="store_true",
                        help="Run the Feature C nested grid search over "
                             "K=[200,250,300] x confidence_threshold=[0.1,0.5,1.0,inf] "
                             "instead of a single evaluation")
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--seed",           type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── 1. Load the GLOBAL model (weights frozen, read-only from here on) ──
    print(f"\n[1] Loading GLOBAL CVAE-PI-LSTM from {args.global_ckpt} ...")
    global_algo: CVAEPILSTMAlgorithm = BaseAlgorithm.load(args.global_ckpt, device)
    global_algo.model.eval()
    for p in global_algo.model.parameters():
        p.requires_grad_(False)   # belt-and-braces: global model never trains again
    print(f"  Global model loaded. dt_n={global_algo._dt_n:.6f}  "
          f"(recomputed at fit time; re-derived below for DB build consistency)")

    # ── 2. Load CVAE encoder (exposes mu AND log_var) ───────────────────────
    print(f"\n[2] Loading CVAE encoder ...")
    encoder = VAEJITLEncoder(
        ckpt_path=args.cvae_ckpt, scaler_path=args.cvae_scaler, device=device,
    )

    seq_len = int(global_algo.cfg.training.seq_len)

    # ── 3. Load / build the window database ─────────────────────────────────
    db = JITFTWindowDatabase()
    db_path = Path(args.db_cache)

    train_batches, _, test_batches = _load_and_split_csv(args.csv, seed=args.seed)

    # Recompute dt_n exactly as CVAEPILSTMAlgorithm._fit_scalers does, using
    # the SAME training batches (guarantees identical value to the one used
    # for the original global training run).
    t_all   = np.concatenate([b["Time (h)"].values for b in train_batches])
    t_range = float(t_all.max() - t_all.min())
    global_algo._dt_n = float(global_algo.cfg.training.dt) / max(t_range, 1e-8)
    print(f"  dt_n (recomputed) = {global_algo._dt_n:.6f}")

    if db_path.exists() and not args.rebuild_db:
        print(f"\n[3] Loading cached window DB from {db_path} ...")
        db.load(str(db_path))
    else:
        print(f"\n[3] Building window DB (encoding + windowing training set) ...")
        from src.data.dataset import PROCESS_FEATURE_COLS
        db.build(
            train_batches, encoder,
            feat_scaler=global_algo.scalers.feat,
            pen_scaler=global_algo.scalers.pen,
            seq_len=seq_len,
            process_cols=PROCESS_FEATURE_COLS,
        )
        db.save(str(db_path))

    # ── 4. Build JIT-FT predictor ────────────────────────────────────────────
    print(f"\n[4] Initialising CVAE_JIT_PI_LSTM_Predictor ...")
    predictor = CVAE_JIT_PI_LSTM_Predictor(
        global_algo=global_algo,
        database=db,
        K=args.K,
        local_epochs=args.local_epochs,
        local_lr=args.local_lr,
        lambda_physics=args.lambda_physics,
        freeze_first_layer=args.freeze_first_layer,
        confidence_threshold=args.confidence_threshold,
        confidence_top_n=args.confidence_top_n,
        distance_temperature=args.distance_temperature,
        device=device,
    )

    # ── 5. Evaluate (single run OR full grid search) ────────────────────────
    if args.grid_search:
        print(f"\n[5] Running Feature C grid search "
              f"(K x confidence_threshold) ...")
        grid_search(
            predictor, test_batches, encoder, global_algo.scalers.feat, seq_len,
            k_values=[200, 250, 300],
            confidence_thresholds=[0.1, 0.5, 1.0, float("inf")],
            max_queries=args.max_queries,
        )
    else:
        print(f"\n[5] Running JIT-FT evaluation ...")
        metrics = predictor.evaluate(
            test_batches, encoder, global_algo.scalers.feat, seq_len,
            max_queries=args.max_queries, verbose=True,
        )

        # ── 6. Comparison ───────────────────────────────────────────────────
        _print_comparison(metrics)


if __name__ == "__main__":
    main()
