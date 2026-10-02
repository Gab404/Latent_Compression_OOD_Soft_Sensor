"""
src/data/dataset.py
===================
IndPenSim V3 data loading and preprocessing.

Responsibilities:
  - Column name constants (feature presets, targets, fault flag)
  - Batch detection from the time-reset convention
  - Sparse label preservation and dense-target interpolation
  - Train / val / test splitting (random or sequential)

Usage
-----
    from src.data.dataset import PenicillinDataModule, RAMAN_LATENT_COLS

    dm = PenicillinDataModule(cfg.data, raman_encoder=encoder)
    dm.load()
    train_batches, val_batches, test_batches = dm.get_splits()
"""

from __future__ import annotations

import ctypes
import gc
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _trim_system_memory() -> None:
    """
    Ask glibc to release free'd heap pages back to the OS (Linux only).

    After a large pd.read_csv + Raman encoding, Python's allocator holds
    onto the freed pages in its internal pool rather than returning them.
    This call shrinks the process RSS to match what is actually live,
    preventing accumulated memory pressure when multiple train.py subprocesses
    run in parallel.
    """
    try:
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass  # non-Linux or libc not available — silently skip


# ── Column definitions ────────────────────────────────────────────────────────

PROCESS_FEATURE_COLS: List[str] = [
    "Time (h)",
    "Aeration rate(Fg:L/h)",
    "Sugar feed rate(Fs:L/h)",
    "Acid flow rate(Fa:L/h)",
    "Base flow rate(Fb:L/h)",
    "Heating/cooling water flow rate(Fc:L/h)",
    "Heating water flow rate(Fh:L/h)",
    "Water for injection/dilution(Fw:L/h)",
    "Air head pressure(pressure:bar)",
    "Dumped broth flow(Fremoved:L/h)",
    "Substrate concentration(S:g/L)",
    "Dissolved oxygen concentration(DO2:mg/L)",
    "Vessel Volume(V:L)",
    "Vessel Weight(Wt:Kg)",
    "pH(pH:pH)",
    "Temperature(T:K)",
    "Generated heat(Q:kJ)",
    "carbon dioxide percent in off-gas(CO2outgas:%)",
    "PAA flow(Fpaa:PAA flow (L/h))",
    "Oil flow(Foil:L/hr)",
    "Oxygen Uptake Rate(OUR:(g min^{-1}))",
    "Oxygen in percent in off-gas(O2:O2  (%))",
    "Carbon evolution rate(CER:g/h)",
]

# Les cibles augmentées pour le calcul du SAP Score
TARGET_COLS: List[str] = [
    "Offline Biomass concentratio(X_offline:X(g L^{-1}))",
    "Offline Penicillin concentration(P_offline:P(g L^{-1}))",
]

FAULT_COL: str = "Fault reference(Fault_ref:Fault ref)"
RAMAN_FLAG_COL: str = "1- No Raman spec"
RAMAN_START_COL_IDX: int = 39
RAMAN_N_COLS: int = 2200
RAMAN_LATENT_DIM: int = 64
RAMAN_LATENT_COLS: List[str] = [f"_raman_{i}" for i in range(RAMAN_LATENT_DIM)]

# Short aliases that configs can reference
FEATURE_PRESETS = {
    "process_23": PROCESS_FEATURE_COLS,
}


# ── PenicillinDataModule ─────────────────────────────────────────────────────

class PenicillinDataModule:
    """
    Loads and preprocesses the IndPenSim V3 CSV into per-batch DataFrames.

    Parameters
    ----------
    cfg : dict | Config
        Must contain:
          csv_path      : path to the CSV file
          [use_raman]   : bool, default False
          [min_len]     : int, minimum rows per batch to keep (default 50)
    raman_encoder : object, optional
        Any object exposing ``encode_dataframe(df) -> np.ndarray[N, latent_dim]``
        (CDAERamanEncoderV2, CVAERamanEncoderV2, PCARamanEncoderV1,
        PLSRamanEncoderV1).  If provided and ``cfg.use_raman`` is True,
        Raman latents are encoded once and attached as columns
        ``_raman_0`` … ``_raman_{latent_dim-1}``.
    """

    def __init__(self, cfg, raman_encoder=None):
        self._cfg           = cfg
        self._raman_encoder = raman_encoder
        self._batches_clean: List[pd.DataFrame] = []
        self._batches_fault: List[pd.DataFrame] = []
        self._loaded        = False
        # Updated in load() to reflect the actual encoder output dimensionality.
        # Defaults to the 64-d CDAE names for backward compatibility.
        self._raman_latent_cols: List[str] = list(RAMAN_LATENT_COLS)

    # ── Public interface ──────────────────────────────────────────────────────

    def load(self) -> "PenicillinDataModule":
        """Read CSV, encode Raman (optional), split into clean / faulty batches."""
        csv_path = self._cfg["csv_path"] if isinstance(self._cfg, dict) \
                   else self._cfg.csv_path

        use_raman = (self._cfg.get("use_raman", False)
                     if isinstance(self._cfg, dict)
                     else getattr(self._cfg, "use_raman", False))

        print(f"  Reading {csv_path} ...")
        # Load all columns when Raman encoding is needed; otherwise only required
        if use_raman and self._raman_encoder is not None:
            df = pd.read_csv(csv_path)
        else:
            needed = PROCESS_FEATURE_COLS + TARGET_COLS + [FAULT_COL]
            df = pd.read_csv(csv_path, usecols=needed)

        print(f"  Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

        # Optional Raman encoding
        if use_raman and self._raman_encoder is not None:
            print("  Encoding Raman spectra ...")
            latents = self._raman_encoder.encode_dataframe(df)   # (N, latent_dim)
            # Derive column names from the actual encoder output dimension so that
            # all encoders (PCA / PLS K-d, CDAE / CVAE 64-d, ...) work without
            # any hardcoded constant.
            n_latent = latents.shape[1]
            self._raman_latent_cols = [f"_raman_{i}" for i in range(n_latent)]
            for i, col in enumerate(self._raman_latent_cols):
                df[col] = latents[:, i]

            # ── Free ~2 GB of raw Raman channels immediately ──────────────────
            # The raw wavenumber columns are no longer needed after encoding.
            # Dropping them now prevents each batch slice from carrying redundant data.
            _keep = set(PROCESS_FEATURE_COLS + TARGET_COLS + [FAULT_COL]
                        + self._raman_latent_cols)
            _drop = [c for c in df.columns if c not in _keep]
            if _drop:
                df.drop(columns=_drop, inplace=True)
            del latents, _drop, _keep
            gc.collect()
            _trim_system_memory()
            print(f"  Raw Raman dropped  →  df now {df.shape[1]} columns")

        # Detect batch boundaries (time resets)
        t_vals    = df["Time (h)"].values
        batch_ids = np.zeros(len(df), dtype=np.int32)
        bid       = 0
        for i in range(1, len(df)):
            if t_vals[i] < t_vals[i - 1]:
                bid += 1
            batch_ids[i] = bid
        df["_batch_id"] = batch_ids
        n_batches = int(bid + 1)
        print(f"  Detected {n_batches} batches")

        # Process each batch
        min_len   = self._cfg.get("min_len", 50) if isinstance(self._cfg, dict) \
                    else getattr(self._cfg, "min_len", 50)
        clean, fault = [], []

        for b_id in range(n_batches):
            b = df[df["_batch_id"] == b_id].copy().reset_index(drop=True)

            # Preserve sparse targets before interpolation
            b["_biomass_sparse"]    = b[TARGET_COLS[0]].copy()
            b["_penicillin_sparse"] = b[TARGET_COLS[1]].copy()

            # Dense interpolation for supervision
            for tc in TARGET_COLS:
                b[tc] = b[tc].interpolate(method="linear").ffill().bfill()

            # Forward-fill Raman latents (first ~10 rows have zero Raman signal).
            # We only replace rows where ALL latent columns are exactly 0 — these
            # are the rows where the encoder received no input spectrum and returned
            # an all-zero vector.
            if use_raman and self._raman_latent_cols:
                raman_vals   = b[self._raman_latent_cols].values
                no_signal    = (raman_vals == 0).all(axis=1)  # rows with no Raman input
                if no_signal.any():
                    for col in self._raman_latent_cols:
                        b[col] = b[col].where(~no_signal, other=np.nan).ffill().bfill()

            # Drop rows still missing required columns
            required = PROCESS_FEATURE_COLS + TARGET_COLS
            b = b.dropna(subset=required).reset_index(drop=True)

            if len(b) < min_len:
                continue

            is_faulty = bool(b[FAULT_COL].max() > 0)
            (fault if is_faulty else clean).append(b)

        self._batches_clean = clean
        self._batches_fault = fault
        self._loaded        = True

        # Release the full DataFrame — all needed data is now in the batch lists
        del df
        gc.collect()
        _trim_system_memory()

        print(f"  Fault-free: {len(clean)}  |  Faulty: {len(fault)}")
        return self

    def get_splits(
        self,
        train_frac: float  = 0.80,
        seed:       int    = 42,
        strategy:   str    = "random",
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
        """
        Split fault-free batches into train / val.  Faulty batches = test.

        Parameters
        ----------
        train_frac : fraction of fault-free batches used for training
        seed       : random seed (used when strategy == "random")
        strategy   : "random" or "sequential"

        Returns
        -------
        train_batches, val_batches, test_batches
        """
        if not self._loaded:
            raise RuntimeError("Call .load() before .get_splits()")

        n_clean = len(self._batches_clean)
        n_train = int(n_clean * train_frac)

        if strategy == "random":
            rng  = np.random.default_rng(seed)
            perm = rng.permutation(n_clean)
            train_b = [self._batches_clean[i] for i in perm[:n_train]]
            val_b   = [self._batches_clean[i] for i in perm[n_train:]]
        elif strategy == "sequential":
            train_b = self._batches_clean[:n_train]
            val_b   = self._batches_clean[n_train:]
        else:
            raise ValueError(f"Unknown split strategy '{strategy}'. "
                             "Choose 'random' or 'sequential'.")

        return train_b, val_b, self._batches_fault

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def batches_clean(self) -> List[pd.DataFrame]:
        return self._batches_clean

    @property
    def batches_fault(self) -> List[pd.DataFrame]:
        return self._batches_fault