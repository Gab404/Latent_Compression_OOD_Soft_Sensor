#!/usr/bin/env python
"""
vae_jitl.py
===========
VAE Just-In-Time Learning (JITL) predictor for penicillin soft sensing.

Paradigm contrast
-----------------
  Global model  (CVAE-PI-LSTM)
    • Trained once on all 72 training batches.
    • A single fixed LSTM captures the *average* input-output relationship
      over the entire operating envelope.
    • Fast at inference, but may struggle when a new batch deviates from
      the training distribution (fault batches, edge conditions).

  JITL / local model  (this module)
    • No offline LSTM training.
    • For every new query point  q  at inference time:
        1. Retrieve the K most similar historical training samples.
        2. Fit an ephemeral local regression exclusively on those K points.
        3. Predict, then discard the local model.
    • The "similarity" is defined in the CVAE probabilistic latent space via
      the Symmetric KL divergence (see below), which accounts for both the
      *location*  (mu difference) and the *certainty*  (variance ratio) of
      each posterior estimate.

Mathematical background — SKL divergence on CVAE posteriors
------------------------------------------------------------
The CVAE encoder maps a preprocessed Raman spectrum  x  to a diagonal
Gaussian posterior:

    q_φ(z | x)  =  N(μ(x),  diag(σ²(x)))

where  μ ∈ ℝ^D  and  σ² = exp(log_var) ∈ ℝ^D₊.

The Symmetric KL divergence between two such distributions
(q_i = N(μ_i, diag(σ²_i))  and  q_q = N(μ_q, diag(σ²_q)))  is:

    D_SKL(q_i ‖ q_q)
        = ½ · [ KL(q_i ‖ q_q) + KL(q_q ‖ q_i) ]
        = ½ · Σ_d [
              σ²_i,d / σ²_q,d
            + σ²_q,d / σ²_i,d
            + (μ_q,d − μ_i,d)² · (1/σ²_q,d + 1/σ²_i,d)
            − 2
          ]

Properties:
  • Symmetric by construction: D_SKL(q_i, q_q) = D_SKL(q_q, q_i).
  • The log-ratio terms in the individual KL divergences cancel, leaving
    only variance-ratio and Mahalanobis-like terms.
  • Degenerates to a scaled squared Euclidean distance when σ²_i = σ²_q
    (constant-variance regime).
  • High encoder confidence (small σ²) amplifies the mu-difference term,
    making SKL more discriminative where the encoder is certain.

Pipeline per query point
------------------------
    raw Raman  (2001-D float32)
      → clamp NaN/neg → SG 1st-deriv (w=15, p=2) → StandardScaler
      → CVAE_Raman.encode()  →  (μ_q, log_var_q)  ∈ ℝ^64 × ℝ^64
      → σ²_q = exp(log_var_q).clip(VAR_FLOOR)
      → D_SKL vs every row in DB         [vectorised, O(N_db × D)]
      → argsort → top-K indices
      → X_knn = DB.mu[top_K]  (K × 64)  +  y_knn = DB.y[top_K]
      → local PLSRegression / GPR        (fit once, predict once, discard)
      → ŷ_q

Optimisation notes
------------------
  1. **Vectorised SKL** — all N_db distances are computed with a single
     NumPy broadcast pass; no Python loop over the database.
  2. **Pre-computed 1/σ²** — stored at DB build time; never recomputed.
  3. **Variance floor** — σ² ≥ VAR_FLOOR = 1e-6 prevents division-by-zero.
  4. **DB built once, cached as .npz** — encoding 80 K training spectra
     takes ~30 s; subsequent runs reload from disk in < 1 s.
  5. **Local PLS** — sklearn PLSRegression is O(K·D·n_comp) per fit,
     ~0.5 ms for K=150, D=64.  GPR is O(K³) per fit, ~5–20 ms for K=150;
     it is the slower option but provides prediction uncertainty.

Usage
-----
    python vae_jitl.py                          # defaults: K=150, PLS
    python vae_jitl.py --K 200 --local-model gpr
    python vae_jitl.py --K 100 --n-components 15 --rebuild-db
    python vae_jitl.py --device cpu --sparse-only
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

# ── Project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.autoencoder.model import CVAE_Raman
from src.data.dataset import (
    FAULT_COL, PROCESS_FEATURE_COLS, RAMAN_START_COL_IDX, TARGET_COLS,
)
from src.evaluation.metrics import compute_metrics

# ── Constants ─────────────────────────────────────────────────────────────────
N_RAMAN_COLS  = 2001
LATENT_DIM    = 64
SG_WINDOW     = 15
SG_POLY       = 2
SG_DERIV      = 1
VAR_FLOOR     = 1e-6      # numerical floor on σ² to prevent 1/0
TRAIN_FRAC    = 0.80
SEED          = 42
MIN_BATCH_LEN = 50

DEFAULT_K             = 150
DEFAULT_N_COMPONENTS  = 10   # PLS latent components for local model
DEFAULT_CKPT          = "./checkpoints/cvae_best.pt"
DEFAULT_SCALER        = "./checkpoints/cvae_scaler.joblib"
DEFAULT_DB_CACHE      = "./checkpoints/vae_jitl_db.npz"


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Low-level preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_sg(X: np.ndarray) -> np.ndarray:
    """Savitzky-Golay first-derivative filter, applied row-wise."""
    return savgol_filter(
        X.astype(np.float64),
        window_length=SG_WINDOW,
        polyorder=SG_POLY,
        deriv=SG_DERIV,
        axis=1,
    ).astype(np.float32)


def _clean_raman(X: np.ndarray) -> np.ndarray:
    """Clamp NaN / negative intensities to zero (instrument artefacts)."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, 0.0, None)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Raw data loading (independent of PenicillinDataModule)
# ─────────────────────────────────────────────────────────────────────────────

def _load_and_split_csv(
    csv_path: str,
    train_frac: float = TRAIN_FRAC,
    seed: int = SEED,
    min_batch_len: int = MIN_BATCH_LEN,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Load the IndPenSim V3 CSV, detect batch boundaries, apply the identical
    80/20 random split used by the global CVAE-PI-LSTM benchmark (seed=42).

    Returns
    -------
    train_batches, val_batches, test_batches
        Each element is a raw DataFrame slice for one batch (all columns
        including the 2001 Raman channels and the sparse target columns).

    Notes
    -----
    The split is reproduced bit-for-bit via ``np.random.default_rng(seed=42)``
    permutation of fault-free batches — the same RNG call sequence as
    ``PenicillinDataModule.get_splits()``.
    """
    print(f"Loading CSV: {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Detect batch boundaries (time resets) ─────────────────────────────
    t_vals    = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid
    df["_batch_id"] = batch_ids
    n_batches = bid + 1

    # ── Sparse targets (preserve before interpolation) ────────────────────
    df["_penicillin_sparse"] = df[TARGET_COLS[1]].copy()
    df["_biomass_sparse"]    = df[TARGET_COLS[0]].copy()

    # ── Dense interpolation (for DB y values) ────────────────────────────
    for tc in TARGET_COLS:
        df[tc] = df[tc].interpolate(method="linear").ffill().bfill()

    # ── Separate fault-free / faulty batches ─────────────────────────────
    clean_dfs: List[pd.DataFrame] = []
    fault_dfs: List[pd.DataFrame] = []

    for b_id in range(n_batches):
        b = df[df["_batch_id"] == b_id].copy().reset_index(drop=True)
        if len(b) < min_batch_len:
            continue
        if FAULT_COL in b.columns and bool(b[FAULT_COL].max() > 0):
            fault_dfs.append(b)
        else:
            clean_dfs.append(b)

    n_clean = len(clean_dfs)
    n_train = int(n_clean * train_frac)
    rng     = np.random.default_rng(seed)
    perm    = rng.permutation(n_clean)

    train_dfs = [clean_dfs[i] for i in perm[:n_train]]
    val_dfs   = [clean_dfs[i] for i in perm[n_train:]]

    print(f"  Batches — train: {len(train_dfs)}  val: {len(val_dfs)}  "
          f"test (fault): {len(fault_dfs)}")
    return train_dfs, val_dfs, fault_dfs


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — VAEJITLDatabase: encoded memory bank
# ─────────────────────────────────────────────────────────────────────────────

class VAEJITLDatabase:
    """
    Memory bank of CVAE-encoded training spectra.

    For every valid training time step we store:
      mu       (D,)  — posterior mean vector from CVAE encoder
      var      (D,)  — posterior variance  σ² = exp(log_var), clamped ≥ VAR_FLOOR
      inv_var  (D,)  — 1/σ², pre-computed for fast SKL computation
      y        ()    — Penicillin concentration (g/L)

    The database is built once and can be cached as a compressed .npz file.

    Parameters
    ----------
    use_sparse_only : bool
        If True  — only store rows where the *actual offline assay* was
                   recorded (_penicillin_sparse is not NaN).  DB size ≈ 1 400.
        If False — store every training time step with its *interpolated* y.
                   DB size ≈ 80 000.  Recommended: provides much denser
                   coverage of the latent space for K-NN retrieval.
    """

    def __init__(self, use_sparse_only: bool = False) -> None:
        self.use_sparse_only = use_sparse_only
        self.mu:      Optional[np.ndarray] = None   # (N, D)
        self.var:     Optional[np.ndarray] = None   # (N, D)
        self.inv_var: Optional[np.ndarray] = None   # (N, D)  pre-computed
        self.y:       Optional[np.ndarray] = None   # (N,)
        self._built   = False

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(
        self,
        train_batches: List[pd.DataFrame],
        encoder: "VAEJITLEncoder",
    ) -> "VAEJITLDatabase":
        """
        Encode all training-set Raman spectra and populate the memory bank.

        Parameters
        ----------
        train_batches : list of per-batch DataFrames (raw CSV slices)
        encoder       : VAEJITLEncoder (wraps CVAE model + scaler)
        """
        mu_parts, var_parts, y_parts = [], [], []

        print(f"Building JITL database ({len(train_batches)} training batches) ...")
        t0 = time.time()

        for batch_df in train_batches:
            # ── Extract Raman channels ─────────────────────────────────────
            all_cols   = batch_df.columns.tolist()
            n_avail    = len(all_cols) - RAMAN_START_COL_IDX
            n_use      = min(N_RAMAN_COLS, n_avail)
            raman_cols = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + n_use]
            X_raw      = batch_df[raman_cols].values.astype(np.float32)

            # ── Identify valid (non-zero energy) rows ──────────────────────
            energy_mask = np.abs(X_raw).sum(axis=1) > 0

            if self.use_sparse_only:
                sparse_mask = batch_df["_penicillin_sparse"].notna().values
                row_mask    = energy_mask & sparse_mask
            else:
                row_mask = energy_mask

            if row_mask.sum() == 0:
                continue

            X_valid = X_raw[row_mask]
            y_valid = batch_df[TARGET_COLS[1]].values[row_mask].astype(np.float32)

            # Skip rows with NaN y (should not happen with interpolation but be safe)
            valid_y = np.isfinite(y_valid)
            if valid_y.sum() == 0:
                continue
            X_valid = X_valid[valid_y]
            y_valid = y_valid[valid_y]

            # ── Encode to latent space ─────────────────────────────────────
            mu, log_var = encoder.encode(X_valid)   # (M, D), (M, D)

            var = np.exp(log_var).clip(VAR_FLOOR)   # σ² ≥ VAR_FLOOR

            mu_parts.append(mu)
            var_parts.append(var)
            y_parts.append(y_valid)

        self.mu      = np.concatenate(mu_parts,  axis=0).astype(np.float32)
        self.var     = np.concatenate(var_parts, axis=0).astype(np.float32)
        self.inv_var = (1.0 / self.var).astype(np.float32)   # cached reciprocal
        self.y       = np.concatenate(y_parts,   axis=0).astype(np.float32)
        self._built  = True

        elapsed = time.time() - t0
        print(f"  Database ready: {len(self.y):,} entries  "
              f"(D={self.mu.shape[1]})  [{elapsed:.1f} s]")
        return self

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the database to a compressed NumPy archive."""
        np.savez_compressed(
            path,
            mu=self.mu, var=self.var, inv_var=self.inv_var, y=self.y,
        )
        print(f"  Database cached → {path}")

    def load(self, path: str) -> "VAEJITLDatabase":
        """Load a previously saved database."""
        data = np.load(path)
        self.mu      = data["mu"]
        self.var     = data["var"]
        self.inv_var = data["inv_var"]
        self.y       = data["y"]
        self._built  = True
        print(f"  Database loaded ← {path}  ({len(self.y):,} entries)")
        return self

    # ── Properties ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.y) if self._built else 0

    @property
    def is_built(self) -> bool:
        return self._built


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — VAEJITLEncoder: CVAE wrapper that exposes (mu, log_var)
# ─────────────────────────────────────────────────────────────────────────────

class VAEJITLEncoder:
    """
    Thin wrapper around a frozen CVAE_Raman that exposes the full posterior
    (mu, log_var) rather than only mu.

    The standard CVAERamanEncoderV2 discards log_var at inference because the
    downstream PI-LSTM only needs mu.  JITL needs σ² for the SKL metric.

    Preprocessing pipeline
    ----------------------
    raw (N, 2001)
        → _clean_raman()           clamp NaN / negatives
        → _apply_sg()              Savitzky-Golay 1st derivative
        → StandardScaler.transform()
        → CVAE_Raman.encode()      → (mu, log_var)  each (N, D)
    """

    def __init__(
        self,
        ckpt_path:    str,
        scaler_path:  str,
        device:       torch.device,
        encode_batch: int = 1024,
    ) -> None:
        self.device       = device
        self.encode_batch = encode_batch

        # ── Load CVAE model (frozen) ───────────────────────────────────────
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        input_length = int(state.get("input_length", N_RAMAN_COLS))
        latent_dim   = int(state.get("latent_dim",   LATENT_DIM))

        self._model = CVAE_Raman(
            input_length=input_length,
            latent_dim=latent_dim,
        ).to(device)
        self._model.load_state_dict(state["model_state_dict"])
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._model.eval()
        self.latent_dim = latent_dim

        # ── Load StandardScaler ────────────────────────────────────────────
        self._scaler: StandardScaler = joblib.load(scaler_path)

        print(f"  VAEJITLEncoder loaded | "
              f"input={input_length}  latent={latent_dim}  device={device}")

    @torch.no_grad()
    def encode(self, X_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full preprocessing + CVAE encoding pipeline.

        Parameters
        ----------
        X_raw : (N, 2001) float32  — raw Raman intensities

        Returns
        -------
        mu      : (N, D) float32  — posterior means
        log_var : (N, D) float32  — posterior log-variances
        """
        # ── Preprocess ────────────────────────────────────────────────────
        X = _clean_raman(X_raw)
        X = _apply_sg(X)
        X = self._scaler.transform(X).astype(np.float32)

        # ── Encode in mini-batches ─────────────────────────────────────────
        mu_parts, lv_parts = [], []
        for start in range(0, len(X), self.encode_batch):
            chunk = torch.tensor(
                X[start : start + self.encode_batch],
                dtype=torch.float32, device=self.device,
            )
            mu_c, lv_c = self._model.encode(chunk)   # (B, D) each
            mu_parts.append(mu_c.cpu().numpy())
            lv_parts.append(lv_c.cpu().numpy())

        return (
            np.concatenate(mu_parts, axis=0).astype(np.float32),
            np.concatenate(lv_parts, axis=0).astype(np.float32),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — VAE_JITL_Predictor: the core engine
# ─────────────────────────────────────────────────────────────────────────────

class VAE_JITL_Predictor:
    """
    VAE Just-In-Time Learning predictor for online soft sensing.

    For every new query spectrum the predictor:
      1. Encodes it to (μ_q, σ²_q) via the frozen CVAE encoder.
      2. Computes the vectorised SKL divergence to all N_db database entries.
      3. Selects the K nearest neighbours.
      4. Fits an ephemeral local regression (PLS or GPR) on (μ_knn, y_knn).
      5. Predicts ŷ_q using the local model, then discards it.

    Parameters
    ----------
    encoder      : VAEJITLEncoder
    database     : VAEJITLDatabase (must be built before calling predict)
    K            : number of nearest neighbours (default 150)
    local_model  : "pls" | "gpr"
    n_components : PLS latent components (ignored for GPR)
    """

    def __init__(
        self,
        encoder:     VAEJITLEncoder,
        database:    VAEJITLDatabase,
        K:           int   = DEFAULT_K,
        local_model: str   = "pls",
        n_components: int  = DEFAULT_N_COMPONENTS,
    ) -> None:
        assert database.is_built, "Call database.build() before creating the predictor."
        self.encoder      = encoder
        self.db           = database
        self.K            = K
        self.local_model  = local_model.lower()
        # Ensure n_components does not exceed K (PLS constraint)
        self.n_components = min(n_components, K - 1)

    # ── Step B — Symmetric KL divergence (fully vectorised) ──────────────────

    def skl_distances(
        self,
        mu_q:      np.ndarray,   # (D,)
        var_q:     np.ndarray,   # (D,)  already clamped
        inv_var_q: np.ndarray,   # (D,)  precomputed 1/var_q
    ) -> np.ndarray:
        """
        Compute D_SKL between query q and every row in the database.

        Formula (all operations are element-wise, sum over latent dim D):
            D_SKL = 0.5 · Σ_d [
                var_i / var_q + var_q / var_i
                + (μ_q - μ_i)² · (1/var_q + 1/var_i)
                - 2
            ]

        Returns
        -------
        distances : (N_db,) float32  — SKL divergence to each DB entry
        """
        # Each variable below is (N_db, D) via broadcasting from (D,)
        #   self.db.var     : (N_db, D)  σ²_i
        #   self.db.inv_var : (N_db, D)  1/σ²_i
        #   self.db.mu      : (N_db, D)  μ_i

        var_ratio_fwd = self.db.var     * inv_var_q[None]   # σ²_i / σ²_q
        var_ratio_rev = var_q[None]     * self.db.inv_var   # σ²_q / σ²_i
        delta_mu_sq   = (mu_q[None] - self.db.mu) ** 2
        mahalanobis   = delta_mu_sq * (inv_var_q[None] + self.db.inv_var)

        # Sum over latent dimension → (N_db,)
        d_skl = 0.5 * (var_ratio_fwd + var_ratio_rev + mahalanobis - 2.0).sum(axis=1)
        return d_skl

    # ── Step C — K-Nearest Neighbours selection ───────────────────────────────

    def _get_knn(
        self,
        mu_q:  np.ndarray,   # (D,)
        var_q: np.ndarray,   # (D,)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (X_knn, y_knn) for the K most similar database entries.

        X_knn : (K, D)  — mu vectors of nearest neighbours
        y_knn : (K,)    — corresponding Penicillin concentrations
        """
        inv_var_q = (1.0 / var_q).astype(np.float32)
        d_skl     = self.skl_distances(mu_q, var_q, inv_var_q)

        k_eff     = min(self.K, len(d_skl))
        top_k_idx = np.argpartition(d_skl, k_eff)[:k_eff]   # O(N) partial sort

        return self.db.mu[top_k_idx], self.db.y[top_k_idx]

    # ── Step D — Ephemeral local model ────────────────────────────────────────

    def _local_predict_pls(
        self,
        X_knn:   np.ndarray,   # (K, D)  — neighbour mu vectors
        y_knn:   np.ndarray,   # (K,)
        x_query: np.ndarray,   # (D,)    — query mu vector
    ) -> float:
        """
        Fit PLSRegression on the K neighbours and predict for x_query.

        The local model operates in a standardised version of the K-NN
        neighbourhood: centering and scaling are estimated from the K
        neighbours, then applied to both X_knn and x_query before PLS.
        This removes the global mean offset (which varies batch-to-batch)
        and ensures PLS captures *local* spectral variation.

        n_components is clamped to min(n_components, K-1, D) to satisfy
        sklearn's constraint n_components ≤ min(n_samples, n_features, n_targets).
        """
        # Local standardisation within the neighbourhood
        scaler_local = StandardScaler()
        X_scaled     = scaler_local.fit_transform(X_knn)
        x_q_scaled   = scaler_local.transform(x_query[None])   # (1, D)

        n_comp = min(self.n_components, X_scaled.shape[0] - 1, X_scaled.shape[1])
        pls    = PLSRegression(n_components=n_comp, max_iter=500)
        pls.fit(X_scaled, y_knn)

        return float(pls.predict(x_q_scaled).ravel()[0])

    def _local_predict_gpr(
        self,
        X_knn:   np.ndarray,
        y_knn:   np.ndarray,
        x_query: np.ndarray,
    ) -> float:
        """
        Fit a Gaussian Process Regressor on the K neighbours and predict.

        Kernel: ConstantKernel × RBF + WhiteKernel (noise).
        GPR scales as O(K³) at fit time — acceptable for K ≤ 300.
        At inference it provides a predictive mean AND standard deviation,
        though this wrapper returns only the mean for the benchmark comparison.
        """
        scaler_local = StandardScaler()
        X_scaled     = scaler_local.fit_transform(X_knn)
        x_q_scaled   = scaler_local.transform(x_query[None])

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) \
               + WhiteKernel(1e-3, (1e-6, 1.0))
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,
            normalize_y=True,
        )
        gpr.fit(X_scaled, y_knn)

        return float(gpr.predict(x_q_scaled).ravel()[0])

    # ── Core predict API ──────────────────────────────────────────────────────

    def predict_from_latent(
        self,
        mu_q:    np.ndarray,   # (D,)  — already encoded and pre-processed
        var_q:   np.ndarray,   # (D,)  — already clamped ≥ VAR_FLOOR
    ) -> float:
        """
        JITL prediction given pre-computed latent statistics.

        Prefer this method when encoding the full test batch at once (see
        evaluate()) to amortise the GPU encoding cost across all query points.
        """
        X_knn, y_knn = self._get_knn(mu_q, var_q)

        if self.local_model == "pls":
            return self._local_predict_pls(X_knn, y_knn, mu_q)
        elif self.local_model == "gpr":
            return self._local_predict_gpr(X_knn, y_knn, mu_q)
        else:
            raise ValueError(f"Unknown local_model '{self.local_model}'. "
                             "Choose 'pls' or 'gpr'.")

    def predict_from_spectrum(self, spectrum_raw: np.ndarray) -> float:
        """
        End-to-end JITL prediction from one raw Raman spectrum.

        Parameters
        ----------
        spectrum_raw : (2001,) or (1, 2001) float32  — raw Raman intensities

        Returns
        -------
        y_hat : float  — predicted Penicillin concentration (g/L)
        """
        spec = np.atleast_2d(spectrum_raw).astype(np.float32)
        mu, log_var = self.encoder.encode(spec)                  # (1, D)
        var         = np.exp(log_var[0]).clip(VAR_FLOOR)
        return self.predict_from_latent(mu[0], var)

    # ── Evaluation loop ───────────────────────────────────────────────────────

    def evaluate(
        self,
        test_batches:  List[pd.DataFrame],
        sparse_only:   bool = True,
        verbose:       bool = True,
    ) -> Dict[str, float]:
        """
        Run the full JITL pipeline on the test set and compute metrics.

        Protocol
        --------
        • If sparse_only=True  (default / benchmark-compatible):
            predictions are made only at the time steps where an actual
            offline Penicillin assay was recorded (_penicillin_sparse ≠ NaN).
            This exactly matches the evaluation protocol of the global
            CVAE-PI-LSTM.
        • If sparse_only=False:
            every time step with a valid Raman spectrum is predicted and
            evaluated against the interpolated target.  This is useful for
            visualising the full predicted trajectory.

        Optimisation
        ------------
        For each test batch all valid spectra are encoded in a single GPU
        forward pass (amortising the CVAE overhead), then only the sparse
        measurement rows are used for JITL prediction and scoring.
        GPU encoding is separated from CPU JITL search + fit to maximise
        resource utilisation.

        Returns
        -------
        metrics : {"RMSE", "MAE", "R2", "n"}
        """
        all_y_true: List[float] = []
        all_y_pred: List[float] = []
        all_times:  List[float] = []   # wall-clock time per query (ms)

        print(f"\n{'='*60}")
        print(f"  VAE-JITL Evaluation")
        print(f"  K={self.K}  local_model={self.local_model.upper()}  "
              f"n_components={self.n_components}  "
              f"sparse_only={sparse_only}")
        print(f"  DB size: {len(self.db):,}  |  "
              f"Test batches: {len(test_batches)}")
        print(f"{'='*60}")

        total_queries = 0

        for b_idx, batch_df in enumerate(test_batches, 1):
            # ── Extract raw Raman for this batch ──────────────────────────
            all_cols   = batch_df.columns.tolist()
            n_avail    = len(all_cols) - RAMAN_START_COL_IDX
            n_use      = min(N_RAMAN_COLS, n_avail)
            raman_cols = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + n_use]
            X_raw      = batch_df[raman_cols].values.astype(np.float32)

            # ── Identify valid (non-zero energy) rows ─────────────────────
            energy_mask = np.abs(X_raw).sum(axis=1) > 0

            # ── Encode ALL valid rows in one GPU pass ─────────────────────
            if energy_mask.sum() == 0:
                continue
            mu_all, lv_all = self.encoder.encode(X_raw[energy_mask])   # (M, D)
            var_all        = np.exp(lv_all).clip(VAR_FLOOR)             # (M, D)

            # Map back to full-batch row indices
            valid_indices = np.where(energy_mask)[0]   # positions in batch_df

            # ── Identify query rows (sparse measurement points or all valid) ─
            if sparse_only:
                sparse_mask = batch_df["_penicillin_sparse"].notna().values
                query_rows  = np.where(energy_mask & sparse_mask)[0]
            else:
                query_rows  = valid_indices

            if len(query_rows) == 0:
                continue

            # ── Run JITL for each query row ───────────────────────────────
            batch_preds = []
            batch_true  = []

            for row_idx in query_rows:
                # Locate this row's position within valid_indices
                pos = np.searchsorted(valid_indices, row_idx)

                mu_q  = mu_all[pos]   # (D,)
                var_q = var_all[pos]  # (D,)

                y_true = float(
                    batch_df["_penicillin_sparse"].iloc[row_idx]
                    if sparse_only
                    else batch_df[TARGET_COLS[1]].iloc[row_idx]
                )
                if not np.isfinite(y_true):
                    continue

                t_start = time.perf_counter()
                y_hat   = self.predict_from_latent(mu_q, var_q)
                elapsed = (time.perf_counter() - t_start) * 1e3   # ms

                batch_preds.append(y_hat)
                batch_true.append(y_true)
                all_times.append(elapsed)

            all_y_pred.extend(batch_preds)
            all_y_true.extend(batch_true)
            total_queries += len(batch_preds)

            if verbose and len(batch_preds) > 0:
                bt = np.array(batch_true)
                bp = np.array(batch_preds)
                b_rmse = np.sqrt(np.mean((bp - bt) ** 2))
                print(f"  Batch {b_idx:2d}  |  queries={len(batch_preds):3d}  "
                      f"RMSE={b_rmse:.3f} g/L  "
                      f"avg_time={np.mean(all_times[-len(batch_preds):]):.1f} ms/query")

        # ── Aggregate metrics ─────────────────────────────────────────────
        y_true_arr = np.array(all_y_true, dtype=np.float32)
        y_pred_arr = np.array(all_y_pred, dtype=np.float32)
        metrics    = compute_metrics(y_true_arr, y_pred_arr)

        avg_ms = float(np.mean(all_times)) if all_times else 0.0
        print(f"\n{'─'*60}")
        print(f"  VAE-JITL results  (K={self.K}, {self.local_model.upper()})")
        print(f"{'─'*60}")
        print(f"  Penicillin  RMSE = {metrics['RMSE']:.4f} g/L")
        print(f"  Penicillin  MAE  = {metrics['MAE']:.4f} g/L")
        print(f"  Penicillin  R²   = {metrics['R2']:.6f}")
        print(f"  N queries        = {metrics['n']}")
        print(f"  Avg query time   = {avg_ms:.2f} ms/query  "
              f"(total {sum(all_times)/1e3:.1f} s)")
        print(f"{'─'*60}\n")

        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — K sensitivity sweep (optional utility)
# ─────────────────────────────────────────────────────────────────────────────

def sweep_K(
    predictor:    VAE_JITL_Predictor,
    test_batches: List[pd.DataFrame],
    k_values:     List[int],
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate the JITL predictor at multiple K values and print a comparison
    table.  Database and encoder are reused; only K changes.

    Returns
    -------
    results : {K: metrics_dict}
    """
    results: Dict[int, Dict[str, float]] = {}
    original_K = predictor.K

    print(f"\n{'='*60}")
    print(f"  K Sensitivity Sweep  ({predictor.local_model.upper()})")
    print(f"{'='*60}")
    print(f"  {'K':>6}  {'RMSE':>10}  {'MAE':>10}  {'R²':>10}  {'n':>6}")
    print(f"  {'─'*50}")

    for K in k_values:
        predictor.K            = K
        predictor.n_components = min(predictor.n_components, K - 1)
        m = predictor.evaluate(test_batches, sparse_only=True, verbose=False)
        results[K] = m
        print(f"  {K:>6}  {m['RMSE']:>10.4f}  {m['MAE']:>10.4f}  "
              f"{m['R2']:>10.6f}  {m['n']:>6}")

    predictor.K = original_K
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _print_comparison(jitl_metrics: Dict[str, float]) -> None:
    """Print a comparison table against the known global model results."""
    global_models = [
        ("CVAE-PI-LSTM (global)",   0.1190, 0.9999),
        ("CDAE-PI-LSTM (global)",   0.0876, 0.9999),
        ("cdae_process (global)",   0.2270, 0.9993),
    ]
    jitl_label = f"VAE-JITL (local)"

    print(f"\n{'='*64}")
    print(f"  Global vs Local Model Comparison  —  Test (fault) batches")
    print(f"{'='*64}")
    print(f"  {'Model':<30}  {'Test RMSE':>12}  {'Test R²':>10}")
    print(f"  {'─'*56}")
    for name, rmse, r2 in global_models:
        print(f"  {name:<30}  {rmse:>12.4f}  {r2:>10.6f}")
    print(f"  {'─'*56}")
    print(f"  {jitl_label:<30}  "
          f"{jitl_metrics['RMSE']:>12.4f}  "
          f"{jitl_metrics['R2']:>10.6f}")
    print(f"{'='*64}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VAE-JITL inference pipeline for penicillin soft sensing"
    )
    parser.add_argument("--csv",          default="./100_Batches_IndPenSim_V3.csv")
    parser.add_argument("--ckpt",         default=DEFAULT_CKPT)
    parser.add_argument("--scaler",       default=DEFAULT_SCALER)
    parser.add_argument("--db-cache",     default=DEFAULT_DB_CACHE,
                        help="Path to .npz DB cache. Built on first run, reloaded after.")
    parser.add_argument("--rebuild-db",   action="store_true",
                        help="Force rebuild the database even if cache exists.")
    parser.add_argument("--K",            type=int,   default=DEFAULT_K,
                        help="Number of nearest neighbours (default 150)")
    parser.add_argument("--local-model",  default="pls", choices=["pls", "gpr"],
                        help="Local regression model (default: pls)")
    parser.add_argument("--n-components", type=int,   default=DEFAULT_N_COMPONENTS,
                        help="PLS latent components (default 10, ignored for GPR)")
    parser.add_argument("--sparse-only",  action="store_true", default=True,
                        help="Evaluate only at sparse offline assay points (default: on)")
    parser.add_argument("--sweep-K",      action="store_true",
                        help="Run a K sensitivity sweep after main evaluation")
    parser.add_argument("--sparse-db",    action="store_true",
                        help="Build DB with sparse-only y values (default: use dense)")
    parser.add_argument("--device",       default="cuda",
                        help="Torch device for CVAE encoding (default: cuda)")
    parser.add_argument("--seed",         type=int,   default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── 1. Load encoder ───────────────────────────────────────────────────
    print("\n[1] Loading CVAE encoder ...")
    encoder = VAEJITLEncoder(
        ckpt_path   = args.ckpt,
        scaler_path = args.scaler,
        device      = device,
    )

    # ── 2. Load / build database ──────────────────────────────────────────
    db = VAEJITLDatabase(use_sparse_only=args.sparse_db)
    db_path = Path(args.db_cache)

    if db_path.exists() and not args.rebuild_db:
        print(f"\n[2] Loading cached DB from {db_path} ...")
        db.load(str(db_path))
    else:
        print(f"\n[2] Building DB (encoding training set) ...")
        train_batches, _, _ = _load_and_split_csv(
            args.csv, seed=args.seed
        )
        db.build(train_batches, encoder)
        db.save(str(db_path))

    # ── 3. Create predictor ───────────────────────────────────────────────
    print(f"\n[3] Initialising VAE-JITL predictor ...")
    predictor = VAE_JITL_Predictor(
        encoder      = encoder,
        database     = db,
        K            = args.K,
        local_model  = args.local_model,
        n_components = args.n_components,
    )

    # ── 4. Load test batches ──────────────────────────────────────────────
    print(f"\n[4] Loading test batches ...")
    _, _, test_batches = _load_and_split_csv(args.csv, seed=args.seed)

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    print(f"\n[5] Running JITL evaluation ...")
    metrics = predictor.evaluate(
        test_batches,
        sparse_only=args.sparse_only,
        verbose=True,
    )

    # ── 6. Print comparison ───────────────────────────────────────────────
    _print_comparison(metrics)

    # ── 7. Optional K sweep ───────────────────────────────────────────────
    if args.sweep_K:
        sweep_K(predictor, test_batches, k_values=[25, 50, 100, 150, 200, 300, 500])


if __name__ == "__main__":
    main()
