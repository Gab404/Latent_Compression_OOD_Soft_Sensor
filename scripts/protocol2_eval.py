#!/usr/bin/env python
"""
protocol2_eval.py
==================
Fill Protocol 2 column: Train R1 only, Val = R1 holdout, Test = R4 (Fault).

For each model:
  - If a regime_ablation checkpoint exists, load it
  - If not, train from scratch (static baselines) or skip
  - Evaluate on R1 val (20%) and R4 (fault batches)
  - Print formatted table

Usage
-----
    PYTHONPATH=. python scripts/protocol2_eval.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.algorithms  # noqa — populate REGISTRY
from src.algorithms.base import BaseAlgorithm, REGISTRY
from src.data.dataset import (
    PenicillinDataModule, RAMAN_START_COL_IDX, TARGET_COLS,
)

# ── Constants ─────────────────────────────────────────────────────────────────
CSV_PATH   = "./100_Batches_IndPenSim_V3.csv"
SEED       = 42
N_RAMAN    = 2001
SG_W, SG_P, SG_D = 15, 2, 1

REGIME_IDS = {
    "R1":    set(range(0,  30)),
    "R2":    set(range(30, 60)),
    "R3":    set(range(60, 90)),
    "Fault": set(range(90, 100)),
}

# Regime ablation checkpoints (train on R1 only)
REGIME_CKPTS = {
    "pi_lstm":          "checkpoints/pi_lstm_regime_ablation.pt",
    "pca_pi_lstm":      "checkpoints/pca_pi_lstm_regime_ablation.pt",
    "pls_pi_lstm":      "checkpoints/pls_pi_lstm_regime_ablation.pt",
    "cdae_pi_lstm":     "checkpoints/cdae_pi_lstm_regime_ablation.pt",
    "cvae_pi_lstm":     "checkpoints/cvae_pi_lstm_regime_ablation.pt",
    "beta_vae_pi_lstm": "checkpoints/beta_vae_pi_lstm_regime_ablation.pt",
    "neural_ode":       "checkpoints/neural_ode_regime_ablation.pt",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_all_batches(csv_path: str = CSV_PATH) -> Dict[str, List[pd.DataFrame]]:
    """Load CSV, detect batches, split by regime, add helper cols."""
    print(f"[DATA] Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    t_vals = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid
    df["_batch_id"] = batch_ids

    # Sparse columns before interpolation
    df["_penicillin_sparse"] = df[TARGET_COLS[1]].copy()
    df["_biomass_sparse"]    = df[TARGET_COLS[0]].copy()

    # Dense interpolation
    for tc in TARGET_COLS:
        df[tc] = df[tc].interpolate(method="linear").ffill().bfill()

    buckets: Dict[str, List[pd.DataFrame]] = {k: [] for k in REGIME_IDS}
    for b_id in range(int(df["_batch_id"].max()) + 1):
        b = df[df["_batch_id"] == b_id].copy().reset_index(drop=True)
        if len(b) < 50:
            continue
        for regime, ids in REGIME_IDS.items():
            if b_id in ids:
                buckets[regime].append(b)
                break

    for r, bl in buckets.items():
        print(f"  {r}: {len(bl)} batches")
    return buckets


def split_r1(
    r1_batches: List[pd.DataFrame],
    seed: int = SEED,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """80/20 split of R1, matching train.py regime_ablation."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(r1_batches))
    n_tr = int(len(r1_batches) * 0.80)
    return (
        [r1_batches[i] for i in perm[:n_tr]],
        [r1_batches[i] for i in perm[n_tr:]],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Raman extraction + SG preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def extract_raman_pen(
    batches: List[pd.DataFrame],
    pen_col: str = "Penicillin concentration(P:g/L)",
    sparse: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract SG-preprocessed Raman spectra and penicillin targets."""
    x_parts, y_parts = [], []
    for b in batches:
        cols = b.columns.tolist()
        rc = cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + N_RAMAN]
        X = b[rc].values.astype(np.float32)
        energy = np.abs(X).sum(axis=1) > 0

        if sparse:
            sp = b["_penicillin_sparse"].values
            valid = energy & np.isfinite(sp) & (sp != 0)
        else:
            valid = energy

        if not valid.any():
            continue

        X_v = np.clip(np.nan_to_num(X[valid], 0.0), 0.0, None)
        X_sg = savgol_filter(
            X_v.astype(np.float64), window_length=SG_W,
            polyorder=SG_P, deriv=SG_D, axis=1,
        ).astype(np.float32)
        y = b[pen_col].values[valid].astype(np.float64)
        x_parts.append(X_sg)
        y_parts.append(y)

    return np.concatenate(x_parts), np.concatenate(y_parts)


def extract_time_pen(
    batches: List[pd.DataFrame],
    pen_col: str = "Penicillin concentration(P:g/L)",
    sparse: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (batch_time_norm, penicillin, batch_id) arrays."""
    t_parts, y_parts, bid_parts = [], [], []
    for b in batches:
        t = b["Time (h)"].values.astype(np.float64)
        y = b[pen_col].values.astype(np.float64)
        bid = int(b["_batch_id"].iloc[0])

        if sparse:
            sp = b["_penicillin_sparse"].values
            valid = np.isfinite(sp) & (sp != 0)
            t, y = t[valid], y[valid]
            n = valid.sum()
        else:
            n = len(t)

        if n < 2:
            continue

        # Normalise time per batch
        t_norm = (t - t.min()) / (t.max() - t.min() + 1e-12)
        t_parts.append(t_norm)
        y_parts.append(y)
        bid_parts.append(np.full(n, bid, dtype=np.int32))

    return np.concatenate(t_parts), np.concatenate(y_parts), np.concatenate(bid_parts)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Time-only baselines
# ─────────────────────────────────────────────────────────────────────────────

def eval_time_baselines(
    r1_train: List[pd.DataFrame],
    r1_val:   List[pd.DataFrame],
    fault:    List[pd.DataFrame],
) -> List[Dict]:
    """Cubic polynomial and mean trajectory on batch-normalised time."""
    results = []

    # --- Cubic polynomial ---
    t_tr, y_tr, _ = extract_time_pen(r1_train, sparse=True)
    t_va, y_va, _ = extract_time_pen(r1_val, sparse=True)
    t_te, y_te, _ = extract_time_pen(fault, sparse=True)

    poly = Pipeline([
        ("poly", PolynomialFeatures(degree=3, include_bias=True)),
        ("lr", LinearRegression()),
    ])
    poly.fit(t_tr.reshape(-1, 1), y_tr)
    val_rmse = rmse(y_va, poly.predict(t_va.reshape(-1, 1)))
    test_rmse = rmse(y_te, poly.predict(t_te.reshape(-1, 1)))
    results.append({
        "model": "Cubic polynomial", "val": val_rmse, "test": test_rmse,
    })

    # --- Mean trajectory ---
    # Build mean trajectory from R1 train: bin time into 200 bins, average pen
    n_bins = 200
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx_tr = np.digitize(t_tr, bins) - 1
    bin_idx_tr = np.clip(bin_idx_tr, 0, n_bins - 1)
    mean_traj = np.zeros(n_bins)
    for i in range(n_bins):
        mask = bin_idx_tr == i
        mean_traj[i] = y_tr[mask].mean() if mask.any() else 0.0

    # Fill gaps via interpolation
    from scipy.interpolate import interp1d
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    valid_bins = mean_traj > 0
    if valid_bins.sum() >= 2:
        interp = interp1d(bin_centers[valid_bins], mean_traj[valid_bins],
                          kind="linear", fill_value="extrapolate")
        mean_traj = interp(bin_centers)

    def predict_mean_traj(t_norm):
        idx = np.digitize(t_norm, bins) - 1
        idx = np.clip(idx, 0, n_bins - 1)
        return mean_traj[idx]

    val_rmse_mt = rmse(y_va, predict_mean_traj(t_va))
    test_rmse_mt = rmse(y_te, predict_mean_traj(t_te))
    results.append({
        "model": "Mean trajectory", "val": val_rmse_mt, "test": test_rmse_mt,
    })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Static chemometric baselines (PCA-PCR, PLS, SVR)
# ─────────────────────────────────────────────────────────────────────────────

def eval_static_baselines(
    r1_train: List[pd.DataFrame],
    r1_val:   List[pd.DataFrame],
    fault:    List[pd.DataFrame],
) -> List[Dict]:
    """Train PCA-PCR, PLS, SVR on R1 SG spectra → Penicillin, test on R4."""
    results = []

    X_tr, y_tr = extract_raman_pen(r1_train, sparse=True)
    X_va, y_va = extract_raman_pen(r1_val, sparse=True)
    X_te, y_te = extract_raman_pen(fault, sparse=True)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    print(f"  Static baselines: train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}")

    # --- PCA-PCR (4 components, matching Protocol 1) ---
    pca = PCA(n_components=4)
    Z_tr = pca.fit_transform(X_tr_s)
    Z_va = pca.transform(X_va_s)
    Z_te = pca.transform(X_te_s)
    lr = LinearRegression()
    lr.fit(Z_tr, y_tr)
    results.append({
        "model": "PCA-PCR",
        "val": rmse(y_va, lr.predict(Z_va)),
        "test": rmse(y_te, lr.predict(Z_te)),
    })

    # --- PLS (4 components) ---
    pls = PLSRegression(n_components=4, scale=False)  # already scaled
    pls.fit(X_tr_s, y_tr)
    results.append({
        "model": "PLS",
        "val": rmse(y_va, pls.predict(X_va_s).ravel()),
        "test": rmse(y_te, pls.predict(X_te_s).ravel()),
    })

    # --- SVR (RBF) ---
    from sklearn.model_selection import RandomizedSearchCV
    svr = SVR(kernel="rbf")
    param_dist = {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.001, 0.01],
                  "epsilon": [0.01, 0.05, 0.1, 0.5]}
    search = RandomizedSearchCV(
        svr, param_dist, n_iter=16, cv=3, scoring="neg_root_mean_squared_error",
        random_state=SEED, n_jobs=-1,
    )
    # Subsample for speed (SVR is O(n^2))
    if len(X_tr_s) > 3000:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X_tr_s), 3000, replace=False)
        search.fit(X_tr_s[idx], y_tr[idx])
    else:
        search.fit(X_tr_s, y_tr)
    best_svr = search.best_estimator_
    print(f"    SVR best: C={best_svr.C}, gamma={best_svr.gamma}, eps={best_svr.epsilon}")
    results.append({
        "model": "SVR (RBF)",
        "val": rmse(y_va, best_svr.predict(X_va_s)),
        "test": rmse(y_te, best_svr.predict(X_te_s)),
    })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. LSTM-based models (load regime_ablation checkpoints)
# ─────────────────────────────────────────────────────────────────────────────

def encode_batches_for_alg(
    batches: List[pd.DataFrame],
    alg_name: str,
) -> List[pd.DataFrame]:
    """Pre-encode Raman columns for encoder-based algorithms."""
    encoder_type = None
    if "cdae" in alg_name:
        from src.data.cdae_encoder import CDAERamanEncoderV2
        encoder_type = "cdae"
    elif "cvae" in alg_name and "beta" not in alg_name:
        from src.data.cvae_encoder import CVAERamanEncoderV2
        encoder_type = "cvae"
    elif "beta_vae" in alg_name:
        from src.data.beta_vae_encoder import BetaVAERamanEncoderV2
        encoder_type = "beta_vae"
    elif "pca" in alg_name:
        from src.data.pca_encoder import PCARamanEncoderV1
        encoder_type = "pca"
    elif "pls" in alg_name:
        from src.data.pls_encoder import PLSRamanEncoderV1
        encoder_type = "pls"

    if encoder_type is None:
        return batches  # pi_lstm / neural_ode use raw features

    device = torch.device("cpu")
    encoder_map = {
        "cdae":     ("CDAERamanEncoderV2", "src.data.cdae_encoder",
                     "checkpoints/cdae_best.pt", "checkpoints/cdae_scaler.joblib"),
        "cvae":     ("CVAERamanEncoderV2", "src.data.cvae_encoder",
                     "checkpoints/cvae_best.pt", "checkpoints/cvae_scaler.joblib"),
        "beta_vae": ("BetaVAERamanEncoderV2", "src.data.beta_vae_encoder",
                     "checkpoints/beta_vae_best.pt", "checkpoints/beta_vae_scaler.joblib"),
        "pca":      ("PCARamanEncoderV1", "src.data.pca_encoder",
                     "checkpoints/pca_best.joblib", "checkpoints/pca_scaler.joblib"),
        "pls":      ("PLSRamanEncoderV1", "src.data.pls_encoder",
                     "checkpoints/pls_best_64.joblib", "checkpoints/pls_scaler_64.joblib"),
    }
    cls_name, mod_name, ckpt, scaler = encoder_map[encoder_type]
    import importlib
    mod = importlib.import_module(mod_name)
    EncoderCls = getattr(mod, cls_name)

    if encoder_type == "pca":
        enc = EncoderCls(pca_model_path=ckpt, scaler_path=scaler)
    elif encoder_type == "pls":
        enc = EncoderCls(pls_model_path=ckpt, scaler_path=scaler,
                         n_components=64)
    else:
        enc = EncoderCls(ckpt_path=ckpt, scaler_path=scaler, device=device)

    encoded = []
    for b in batches:
        b2 = b.copy()
        latents = enc.encode_dataframe(b2)
        for d in range(latents.shape[1]):
            b2[f"_raman_{d}"] = latents[:, d]
        encoded.append(b2)
    return encoded


def eval_lstm_model(
    alg_name:  str,
    r1_val:    List[pd.DataFrame],
    fault:     List[pd.DataFrame],
    device:    torch.device,
) -> Dict:
    """Load regime_ablation checkpoint, evaluate on R1 val and R4."""
    ckpt_path = REGIME_CKPTS.get(alg_name, "")
    if not Path(ckpt_path).exists():
        return {"model": alg_name, "val": float("nan"), "test": float("nan"),
                "note": "no checkpoint"}

    print(f"  Loading {ckpt_path} ...")
    alg = BaseAlgorithm.load(ckpt_path, device)

    # Pre-encode if needed
    needs_encoder = any(x in alg_name for x in
                        ("cdae", "cvae", "beta_vae", "pca", "pls"))
    if needs_encoder:
        print(f"    Encoding R1 val + R4 ...")
        r1_val_enc = encode_batches_for_alg(r1_val, alg_name)
        fault_enc = encode_batches_for_alg(fault, alg_name)
    else:
        r1_val_enc = r1_val
        fault_enc = fault

    val_metrics  = alg.evaluate(r1_val_enc)
    test_metrics = alg.evaluate(fault_enc)

    return {
        "model": alg_name,
        "val":  float(val_metrics["pen"]["RMSE"]),
        "test": float(test_metrics["pen"]["RMSE"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. JIT-FT variants
# ─────────────────────────────────────────────────────────────────────────────

def eval_jit_ft(
    variant:   str,   # "beta_vae" or "cvae"
    r1_train:  List[pd.DataFrame],
    r1_val:    List[pd.DataFrame],
    fault:     List[pd.DataFrame],
    device:    torch.device,
) -> Dict:
    """Run JIT-FT evaluation with R1-only database."""
    script = Path("scripts/cvae_jit_ft.py")
    if not script.exists():
        return {"model": f"{variant}-LSTM JIT-FT", "val": float("nan"),
                "test": float("nan"), "note": "script not found"}

    # Import the JIT-FT module
    import importlib.util
    spec = importlib.util.spec_from_file_location("cvae_jit_ft", str(script))
    jit_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jit_mod)

    from src.autoencoder.model import BetaVAE_Raman, CVAE_Raman

    # Determine config
    if variant == "beta_vae":
        db_path = "checkpoints/beta_vae_jit_ft_db_r1.npz"
        global_ckpt = "checkpoints/beta_vae_pi_lstm_regime_ablation.pt"
        enc_ckpt_default = "checkpoints/beta_vae_best.pt"
        enc_scaler_default = "checkpoints/beta_vae_scaler.joblib"
        model_class = BetaVAE_Raman
        K = 150
        label = "β-VAE-LSTM JIT-FT"
    else:  # cvae
        db_path = "checkpoints/cvae_jit_ft_db_r1.npz"
        global_ckpt = "checkpoints/cvae_pi_lstm_regime_ablation.pt"
        enc_ckpt_default = "checkpoints/cvae_best.pt"
        enc_scaler_default = "checkpoints/cvae_scaler.joblib"
        model_class = CVAE_Raman
        K = 200
        label = "CVAE-LSTM JIT-FT"

    if not Path(db_path).exists() or not Path(global_ckpt).exists():
        return {"model": f"{label} (K={K})", "val": float("nan"),
                "test": float("nan"), "note": f"missing {db_path} or {global_ckpt}"}

    # Load global algorithm
    global_algo = BaseAlgorithm.load(global_ckpt, device)
    global_algo.model.eval()
    for p in global_algo.model.parameters():
        p.requires_grad_(False)

    # BUGFIX: the Raman encoder used here MUST match the one `global_ckpt`
    # was actually trained with (its `feat` MinMaxScaler is calibrated to
    # that encoder's latent range) — derive it from the checkpoint's own
    # cfg instead of hardcoding a fixed default, to avoid silent
    # encoder/LSTM mismatches that inflate RMSE by 50-100x.
    enc_ckpt   = getattr(global_algo.cfg.data, "raman_ckpt", None)   or enc_ckpt_default
    enc_scaler = getattr(global_algo.cfg.data, "raman_scaler", None) or enc_scaler_default
    if (enc_ckpt, enc_scaler) != (enc_ckpt_default, enc_scaler_default):
        print(f"  [{label}] encoder resolved from {global_ckpt}'s cfg: {enc_ckpt}")


    # Recompute dt_n from R1 train
    t_all = np.concatenate([b["Time (h)"].values for b in r1_train])
    t_range = float(t_all.max() - t_all.min())
    global_algo._dt_n = float(global_algo.cfg.training.dt) / max(t_range, 1e-8)

    # Load encoder
    encoder = jit_mod.VAEJITLEncoder(
        ckpt_path=enc_ckpt, scaler_path=enc_scaler, device=device,
        model_class=model_class,
    )

    # Load DB
    db = jit_mod.JITFTWindowDatabase()
    db.load(db_path)

    seq_len = int(global_algo.cfg.training.seq_len)

    # Build predictor
    predictor = jit_mod.CVAE_JIT_PI_LSTM_Predictor(
        global_algo=global_algo,
        database=db,
        K=K,
        device=device,
        model_label=label,
    )

    # Evaluate
    val_metrics = predictor.evaluate(
        r1_val, encoder, global_algo.scalers.feat, seq_len, verbose=False,
    )
    test_metrics = predictor.evaluate(
        fault, encoder, global_algo.scalers.feat, seq_len, verbose=False,
    )

    return {
        "model": f"{label} (K={K})",
        "val": float(val_metrics["RMSE"]),
        "test": float(test_metrics["RMSE"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. VAE-JITL
# ─────────────────────────────────────────────────────────────────────────────

def eval_vae_jitl(
    r1_train: List[pd.DataFrame],
    r1_val:   List[pd.DataFrame],
    fault:    List[pd.DataFrame],
    device:   torch.device,
    K: int = 500,
) -> Dict:
    """VAE-JITL: build R1-only DB, evaluate with local PLS."""
    script = Path("scripts/vae_jitl.py")
    if not script.exists():
        return {"model": "VAE-JITL", "val": float("nan"),
                "test": float("nan"), "note": "script not found"}

    import importlib.util
    spec = importlib.util.spec_from_file_location("vae_jitl", str(script))
    jitl_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jitl_mod)

    from src.autoencoder.model import BetaVAE_Raman

    # Build or load DB from R1 train
    db_path = "checkpoints/vae_jitl_db_r1.npz"
    encoder = jitl_mod.VAEJITLEncoder(
        ckpt_path="checkpoints/beta_vae_best.pt",
        scaler_path="checkpoints/beta_vae_scaler.joblib",
        device=device,
        model_class=BetaVAE_Raman,
    )

    db = jitl_mod.VAEJITLDatabase()
    if Path(db_path).exists():
        print("    Loading VAE-JITL R1 database ...")
        db.load(db_path)
    else:
        print("    Building VAE-JITL R1 database ...")
        db.build(r1_train, encoder)
        db.save(db_path)

    predictor = jitl_mod.VAE_JITL_Predictor(
        encoder=encoder, database=db, K=K,
    )

    val_metrics = predictor.evaluate(r1_val, sparse_only=True, verbose=False)
    test_metrics = predictor.evaluate(fault, sparse_only=True, verbose=False)

    return {
        "model": f"VAE-JITL (K={K})",
        "val": float(val_metrics["RMSE"]),
        "test": float(test_metrics["RMSE"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cpu")
    print(f"[EVAL] Protocol 2: Train R1, Test R4  |  device={device}")

    buckets = load_all_batches()
    r1_train, r1_val = split_r1(buckets["R1"])
    fault = buckets["Fault"]
    print(f"  R1 train: {len(r1_train)}  R1 val: {len(r1_val)}  R4: {len(fault)}")

    all_results = []

    # ── 1. Time-only baselines ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Time-only baselines")
    print(f"{'─'*60}")
    time_results = eval_time_baselines(r1_train, r1_val, fault)
    all_results.extend(time_results)
    for r in time_results:
        print(f"    {r['model']:<30}  Val={r['val']:.3f}  Test={r['test']:.3f}")

    # ── 2. Static chemometric baselines ───────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Static chemometric baselines")
    print(f"{'─'*60}")
    static_results = eval_static_baselines(r1_train, r1_val, fault)
    all_results.extend(static_results)
    for r in static_results:
        print(f"    {r['model']:<30}  Val={r['val']:.3f}  Test={r['test']:.3f}")

    # ── 3. LSTM-based models (regime ablation checkpoints) ────────────────────
    print(f"\n{'─'*60}")
    print("  LSTM-based models (regime ablation checkpoints)")
    print(f"{'─'*60}")
    lstm_models = [
        "pca_pi_lstm", "pls_pi_lstm",
        "cdae_pi_lstm", "cvae_pi_lstm", "beta_vae_pi_lstm",
        "pi_lstm", "neural_ode",
    ]
    for name in lstm_models:
        print(f"\n  [{name}]")
        r = eval_lstm_model(name, r1_val, fault, device)
        all_results.append(r)
        note = r.get("note", "")
        if note:
            print(f"    SKIPPED: {note}")
        else:
            print(f"    Val={r['val']:.3f}  Test={r['test']:.3f}")

    # ── 4. JIT-FT variants ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  JIT-FT variants")
    print(f"{'─'*60}")
    for variant in ("beta_vae", "cvae"):
        print(f"\n  [{variant}-LSTM JIT-FT]")
        try:
            r = eval_jit_ft(variant, r1_train, r1_val, fault, device)
            all_results.append(r)
            print(f"    Val={r['val']:.3f}  Test={r['test']:.3f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            all_results.append({
                "model": f"{variant}-LSTM JIT-FT", "val": float("nan"),
                "test": float("nan"), "note": str(e),
            })

    # ── 5. VAE-JITL ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  VAE-JITL")
    print(f"{'─'*60}")
    try:
        r = eval_vae_jitl(r1_train, r1_val, fault, device, K=500)
        all_results.append(r)
        print(f"    Val={r['val']:.3f}  Test={r['test']:.3f}")
    except Exception as e:
        print(f"    ERROR: {e}")
        all_results.append({
            "model": "VAE-JITL (K=500)", "val": float("nan"),
            "test": float("nan"), "note": str(e),
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  PROTOCOL 2 RESULTS — Train R1, Test R4 (Fault)")
    print(f"{'='*75}")
    print(f"  {'Model':<35} {'Val':>8} {'Test':>8} {'d%':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    for r in all_results:
        v, t = r["val"], r["test"]
        if np.isfinite(v) and np.isfinite(t) and v > 0:
            delta = (t - v) / v * 100
            delta_s = f"{delta:>+7.0f}%"
        else:
            delta_s = "    N/A"
        vs = f"{v:.3f}" if np.isfinite(v) else "  N/A"
        ts = f"{t:.3f}" if np.isfinite(t) else "  N/A"
        print(f"  {r['model']:<35} {vs:>8} {ts:>8} {delta_s:>8}")
    print(f"{'='*75}")

    # Save
    out = Path("outputs/protocol2_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2, default=float))
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
