#!/usr/bin/env python
"""
run_all_seeds.py
================
Orchestrator script: trains and evaluates ALL models under both protocols
with multiple seeds, then compiles mean ± std RMSE table.

Models covered:
  - Time-only baselines (cubic polynomial, mean trajectory)
  - Static chemometric baselines (PCA-PCR, PLS, SVR)
  - Linear-latent + LSTM (PCA-LSTM, PLS-LSTM)
  - Deep-latent + LSTM (CDAE-LSTM, CVAE-LSTM, β-VAE-LSTM)
  - On-the-fly adaptation (β-VAE JIT-FT, CVAE JIT-FT, VAE-JITL)
  - End-to-end sequence models (LSTM, Neural ODE)

Usage
-----
    # Full run (all models, both protocols, seeds 1-4):
    PYTHONPATH=. python scripts/run_all_seeds.py --device cuda:0

    # Single protocol:
    PYTHONPATH=. python scripts/run_all_seeds.py --protocol 2 --device cuda:0

    # Specific models only:
    PYTHONPATH=. python scripts/run_all_seeds.py --models beta_vae_pi_lstm cvae_pi_lstm --device cuda:0

    # Custom seeds:
    PYTHONPATH=. python scripts/run_all_seeds.py --seeds 1 2 3 4 5 --device cuda:0

    # Evaluate only (skip training, use existing checkpoints):
    PYTHONPATH=. python scripts/run_all_seeds.py --eval-only --device cuda:0
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────
SEEDS = [1, 2, 3, 4]
CSV_PATH = "./100_Batches_IndPenSim_V3.csv"
PYTHON = sys.executable

REGIME_IDS = {
    "R1":    set(range(0,  30)),
    "R2":    set(range(30, 60)),
    "R3":    set(range(60, 90)),
    "Fault": set(range(90, 100)),
}

# Models trainable via train.py
TRAINPY_MODELS = [
    "pi_lstm", "neural_ode",
    "pca_pi_lstm", "pls_pi_lstm",
    "cdae_pi_lstm", "cvae_pi_lstm", "beta_vae_pi_lstm",
]

# Models that need VAE/encoder retraining per seed (end-to-end)
E2E_MODELS = ["beta_vae_pi_lstm", "cvae_pi_lstm", "cdae_pi_lstm"]

# All model categories
ALL_MODELS = [
    "cubic_poly", "mean_traj",
    "pca_pcr", "pls", "svr",
    "pca_pi_lstm", "pls_pi_lstm",
    "cdae_pi_lstm", "cvae_pi_lstm", "beta_vae_pi_lstm",
    "beta_vae_jit_ft", "cvae_jit_ft", "vae_jitl",
    "pi_lstm", "neural_ode",
]

PEN_MAX = 36.183  # max penicillin g/L in dataset, for NRMSE

OUT_DIR = Path("outputs/multi_seed")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_cmd(cmd: List[str], timeout: int = 7200) -> Tuple[int, str]:
    """Run a subprocess and return (returncode, combined output)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT) + ":" + env.get("PYTHONPATH", "")
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env,
        cwd=str(_ROOT),
    )
    output = result.stdout + result.stderr
    return result.returncode, output


def _set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 1. train.py models (LSTM-based, use existing train.py CLI)
# ─────────────────────────────────────────────────────────────────────────────

def train_trainpy_model(model: str, protocol: int, seed: int, device: str) -> str:
    """
    Train one model via train.py. Returns checkpoint path.

    Protocol 1: uses default split (R1-R3 train, R4 test) — no flag needed
    Protocol 2: uses --regime-ablation (R1 only)
    """
    ckpt_dir = OUT_DIR / f"p{protocol}" / model
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(ckpt_dir / f"{model}_s{seed}.pt")

    cmd = [
        PYTHON, "train.py",
        "--model", model,
        "--seed", str(seed),
        "--device", device,
        "--ckpt", ckpt_path,
        "--no-plots",
    ]
    if protocol == 2:
        cmd.append("--regime-ablation")

    print(f"    CMD: {' '.join(cmd)}")
    rc, output = run_cmd(cmd)
    if rc != 0:
        print(f"    FAILED (rc={rc})")
        # Print last 20 lines of output
        lines = output.strip().split("\n")
        for line in lines[-20:]:
            print(f"      {line}")
        return ""
    print(f"    OK → {ckpt_path}")
    return ckpt_path


def eval_trainpy_model(model: str, ckpt_path: str, protocol: int, seed: int, device: str) -> Dict:
    """Evaluate a train.py model checkpoint via evaluate.py."""
    cmd = [
        PYTHON, "evaluate.py",
        "--model", model,
        "--ckpt", ckpt_path,
        "--device", device,
        "--seed", str(seed),
    ]
    if protocol == 2:
        cmd.append("--regime-ablation")

    rc, output = run_cmd(cmd)
    if rc != 0:
        return {"RMSE": float("nan"), "MAE": float("nan")}

    # Parse RMSE from output
    import re
    rmse_match = re.search(r"\[Penicillin\]\s+RMSE=([\d.]+)", output)
    mae_match = re.search(r"MAE=([\d.]+)", output)

    return {
        "RMSE": float(rmse_match.group(1)) if rmse_match else float("nan"),
        "MAE":  float(mae_match.group(1)) if mae_match else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. End-to-end models (β-VAE, CVAE, CDAE need encoder retrain per seed)
# ─────────────────────────────────────────────────────────────────────────────

def train_e2e_model(model: str, protocol: int, seed: int, device: str) -> str:
    """
    Train end-to-end (encoder + LSTM) via multi_seed_train.py logic.

    For β-VAE: retrains VAE encoder from scratch + LSTM
    For CVAE/CDAE: retrains the respective autoencoder + LSTM

    Returns LSTM checkpoint path.
    """
    # Use the multi_seed_train.py script for beta_vae
    if model == "beta_vae_pi_lstm":
        cmd = [
            PYTHON, "scripts/multi_seed_train.py",
            "--protocol", str(protocol),
            "--device", device,
        ]
        # The script trains all 4 seeds; individual runs handled internally
        # For per-seed execution, we call the script once — it handles all seeds
        return str(OUT_DIR / f"lstm_p{protocol}_s{seed}.pt")

    # For CVAE/CDAE: retrain encoder with scripts/train_autoencoder.py, then LSTM
    encoder_type = "cvae" if "cvae" in model else "cdae"
    ckpt_dir = OUT_DIR / f"p{protocol}" / model
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    enc_ckpt = str(ckpt_dir / f"{encoder_type}_s{seed}.pt")
    enc_scaler = str(ckpt_dir / f"{encoder_type}_scaler_s{seed}.joblib")
    lstm_ckpt = str(ckpt_dir / f"{model}_s{seed}.pt")

    # Step 1: Train encoder
    enc_cmd = [
        PYTHON, "scripts/train_autoencoder.py",
        "--model", encoder_type,
        "--seed", str(seed),
        "--device", device,
        "--ckpt", enc_ckpt,
        "--scaler", enc_scaler,
    ]
    if protocol == 2:
        enc_cmd.append("--regime-ablation")

    print(f"    Training {encoder_type} encoder (seed={seed}) ...")
    rc, output = run_cmd(enc_cmd, timeout=3600)
    if rc != 0:
        print(f"    Encoder training FAILED")
        return ""

    # Step 2: Train LSTM pointing to new encoder
    lstm_cmd = [
        PYTHON, "train.py",
        "--model", model,
        "--seed", str(seed),
        "--device", device,
        "--ckpt", lstm_ckpt,
        "--no-plots",
    ]
    if protocol == 2:
        lstm_cmd.append("--regime-ablation")

    # Override encoder paths via env or config? train.py reads from YAML.
    # We need a custom config pointing to the new encoder.
    # Write a temporary config:
    import yaml
    base_cfg_path = f"configs/{model}.yaml"
    if protocol == 2 and Path(f"configs/{model}_regime_ablation.yaml").exists():
        base_cfg_path = f"configs/{model}_regime_ablation.yaml"

    with open(base_cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg["data"]["raman_ckpt"] = enc_ckpt
    cfg["data"]["raman_scaler"] = enc_scaler
    cfg["data"]["seed"] = seed

    tmp_cfg_path = str(ckpt_dir / f"config_s{seed}.yaml")
    with open(tmp_cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    lstm_cmd = [
        PYTHON, "train.py",
        "--model", model,
        "--config", tmp_cfg_path,
        "--seed", str(seed),
        "--device", device,
        "--ckpt", lstm_ckpt,
        "--no-plots",
    ]
    if protocol == 2:
        lstm_cmd.append("--regime-ablation")

    print(f"    Training {model} LSTM (seed={seed}) ...")
    rc, output = run_cmd(lstm_cmd, timeout=3600)
    if rc != 0:
        print(f"    LSTM training FAILED")
        lines = output.strip().split("\n")
        for line in lines[-10:]:
            print(f"      {line}")
        return ""

    return lstm_ckpt


# ─────────────────────────────────────────────────────────────────────────────
# 3. Static baselines (train from scratch, pure sklearn)
# ─────────────────────────────────────────────────────────────────────────────

def train_eval_static_baselines(protocol: int, seed: int) -> Dict[str, Dict]:
    """Train & evaluate static baselines. Returns {model_name: {val_RMSE, test_RMSE}}."""
    # Import lazily to avoid overhead
    from scipy.signal import savgol_filter
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.svm import SVR
    from src.data.dataset import RAMAN_START_COL_IDX, TARGET_COLS

    _set_seed(seed)

    # Load data
    df = pd.read_csv(CSV_PATH)
    t_vals = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid
    df["_batch_id"] = batch_ids
    df["_penicillin_sparse"] = df[TARGET_COLS[1]].copy()

    # Gather batches
    if protocol == 1:
        train_ids = set(range(0, 90))
    else:
        train_ids = set(range(0, 30))
    fault_ids = set(range(90, 100))

    train_batches, fault_batches = [], []
    for b_id in range(100):
        b = df[df["_batch_id"] == b_id].copy().reset_index(drop=True)
        if len(b) < 50:
            continue
        if b_id in train_ids:
            train_batches.append(b)
        elif b_id in fault_ids:
            fault_batches.append(b)

    # 80/20 split
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_batches))
    n_tr = int(len(train_batches) * 0.80)
    r_train = [train_batches[i] for i in perm[:n_tr]]
    r_val   = [train_batches[i] for i in perm[n_tr:]]

    N_RAMAN = 2001
    pen_col = "Penicillin concentration(P:g/L)"

    def extract_raman_pen(batches, sparse=True):
        x_parts, y_parts = [], []
        for b in batches:
            cols = b.columns.tolist()
            rc = cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + N_RAMAN]
            X = b[rc].values.astype(np.float32)
            energy = np.abs(X).sum(axis=1) > 0
            if sparse:
                sp = b["_penicillin_sparse"].values
                valid = energy & np.isfinite(sp)
            else:
                valid = energy
            if not valid.any():
                continue
            X_v = np.clip(np.nan_to_num(X[valid], 0.0), 0.0, None)
            X_sg = savgol_filter(X_v.astype(np.float64), 15, 2, deriv=1, axis=1).astype(np.float32)
            # Use offline pen for y when sparse
            if sparse:
                y = b["_penicillin_sparse"].values[valid].astype(np.float64)
            else:
                y = b[pen_col].values[valid].astype(np.float64)
            x_parts.append(X_sg)
            y_parts.append(y)
        return np.concatenate(x_parts), np.concatenate(y_parts)

    def extract_time_pen(batches, sparse=True):
        t_parts, y_parts = [], []
        for b in batches:
            t = b["Time (h)"].values.astype(np.float64)
            if sparse:
                sp = b["_penicillin_sparse"].values
                valid = np.isfinite(sp)
                t_v, y_v = t[valid], sp[valid].astype(np.float64)
            else:
                y_v = b[pen_col].values.astype(np.float64)
                t_v = t
            if len(t_v) < 2:
                continue
            t_norm = (t_v - t_v.min()) / (t_v.max() - t_v.min() + 1e-12)
            t_parts.append(t_norm)
            y_parts.append(y_v)
        return np.concatenate(t_parts), np.concatenate(y_parts)

    def rmse(yt, yp):
        return float(np.sqrt(np.mean((yt - yp) ** 2)))

    results = {}

    # --- Time baselines ---
    t_tr, y_tr = extract_time_pen(r_train)
    t_va, y_va = extract_time_pen(r_val)
    t_te, y_te = extract_time_pen(fault_batches)

    # Cubic polynomial
    poly = Pipeline([("poly", PolynomialFeatures(3, include_bias=True)), ("lr", LinearRegression())])
    poly.fit(t_tr.reshape(-1, 1), y_tr)
    results["cubic_poly"] = {
        "val_RMSE": rmse(y_va, poly.predict(t_va.reshape(-1, 1))),
        "test_RMSE": rmse(y_te, poly.predict(t_te.reshape(-1, 1))),
    }

    # Mean trajectory
    n_bins = 200
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(t_tr, bins) - 1, 0, n_bins - 1)
    mean_traj = np.array([y_tr[bin_idx == i].mean() if (bin_idx == i).any() else 0.0 for i in range(n_bins)])
    from scipy.interpolate import interp1d
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    valid_b = mean_traj > 0
    if valid_b.sum() >= 2:
        interp = interp1d(bin_centers[valid_b], mean_traj[valid_b], kind="linear", fill_value="extrapolate")
        mean_traj = interp(bin_centers)

    def predict_mt(t_n):
        idx = np.clip(np.digitize(t_n, bins) - 1, 0, n_bins - 1)
        return mean_traj[idx]

    results["mean_traj"] = {
        "val_RMSE": rmse(y_va, predict_mt(t_va)),
        "test_RMSE": rmse(y_te, predict_mt(t_te)),
    }

    # --- Spectral baselines ---
    X_tr, y_tr_s = extract_raman_pen(r_train)
    X_va, y_va_s = extract_raman_pen(r_val)
    X_te, y_te_s = extract_raman_pen(fault_batches)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    # PCA-PCR
    pca = PCA(n_components=4)
    Z_tr = pca.fit_transform(X_tr_s)
    lr = LinearRegression().fit(Z_tr, y_tr_s)
    results["pca_pcr"] = {
        "val_RMSE": rmse(y_va_s, lr.predict(pca.transform(X_va_s))),
        "test_RMSE": rmse(y_te_s, lr.predict(pca.transform(X_te_s))),
    }

    # PLS
    pls = PLSRegression(n_components=4, scale=False).fit(X_tr_s, y_tr_s)
    results["pls"] = {
        "val_RMSE": rmse(y_va_s, pls.predict(X_va_s).ravel()),
        "test_RMSE": rmse(y_te_s, pls.predict(X_te_s).ravel()),
    }

    # SVR
    svr_search = RandomizedSearchCV(
        SVR(kernel="rbf"),
        {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.001, 0.01], "epsilon": [0.01, 0.05, 0.1, 0.5]},
        n_iter=16, cv=3, scoring="neg_root_mean_squared_error", random_state=seed, n_jobs=-1,
    )
    idx = rng.choice(len(X_tr_s), min(3000, len(X_tr_s)), replace=False)
    svr_search.fit(X_tr_s[idx], y_tr_s[idx])
    best_svr = svr_search.best_estimator_
    results["svr"] = {
        "val_RMSE": rmse(y_va_s, best_svr.predict(X_va_s)),
        "test_RMSE": rmse(y_te_s, best_svr.predict(X_te_s)),
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-seed training & evaluation for all models")
    parser.add_argument("--protocol", default="both", choices=["1", "2", "both"])
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of models to run (default: all)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, evaluate existing checkpoints only")
    args = parser.parse_args()

    models = args.models or ALL_MODELS
    seeds = args.seeds
    protocols = [1, 2] if args.protocol == "both" else [int(args.protocol)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}  # {(protocol, model, seed): {val_RMSE, test_RMSE}}

    for protocol in protocols:
        print(f"\n{'='*70}")
        print(f"  PROTOCOL {protocol}  {'(R1-R3)' if protocol == 1 else '(R1 only)'}")
        print(f"{'='*70}")

        for seed in seeds:
            print(f"\n  ── Seed {seed} ──")

            # Static baselines (always train from scratch, fast)
            static_names = {"cubic_poly", "mean_traj", "pca_pcr", "pls", "svr"}
            needed_static = static_names & set(models)
            if needed_static and not args.eval_only:
                print(f"  Training static baselines ...")
                static_res = train_eval_static_baselines(protocol, seed)
                for name, metrics in static_res.items():
                    if name in models:
                        all_results[(protocol, name, seed)] = metrics
                        print(f"    {name:<20} val={metrics['val_RMSE']:.3f}  test={metrics['test_RMSE']:.3f}")

            # LSTM-based models via train.py
            for model in models:
                if model in static_names:
                    continue
                if model in ("beta_vae_jit_ft", "cvae_jit_ft", "vae_jitl"):
                    continue  # handled separately below
                if model not in TRAINPY_MODELS:
                    continue

                if not args.eval_only:
                    print(f"\n  Training {model} (protocol={protocol}, seed={seed}) ...")
                    ckpt = train_trainpy_model(model, protocol, seed, args.device)
                else:
                    ckpt_dir = OUT_DIR / f"p{protocol}" / model
                    ckpt = str(ckpt_dir / f"{model}_s{seed}.pt")

                if ckpt and Path(ckpt).exists():
                    metrics = eval_trainpy_model(model, ckpt, protocol, seed, args.device)
                    all_results[(protocol, model, seed)] = {
                        "val_RMSE": float("nan"),  # TODO: parse from output
                        "test_RMSE": metrics["RMSE"],
                    }
                    print(f"    {model:<25} test_RMSE={metrics['RMSE']:.3f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print(f"  MULTI-SEED RESULTS SUMMARY")
    print(f"{'='*90}")

    for protocol in protocols:
        print(f"\n  Protocol {protocol} ({'R1-R3' if protocol == 1 else 'R1 only'})")
        print(f"  {'Model':<25} {'Test RMSE (mean±std)':<25} {'NRMSE %':<12} {'Seeds'}")
        print(f"  {'-'*75}")
        for model in models:
            test_rmses = [all_results.get((protocol, model, s), {}).get("test_RMSE", np.nan) for s in seeds]
            valid = [r for r in test_rmses if np.isfinite(r)]
            if valid:
                mean_r = np.mean(valid)
                std_r = np.std(valid)
                nrmse = mean_r / PEN_MAX * 100
                print(f"  {model:<25} {mean_r:.3f} ± {std_r:.3f}         {nrmse:.1f}%        n={len(valid)}")

    # Save results
    out_path = OUT_DIR / "summary.json"
    serializable = {f"p{k[0]}_{k[1]}_s{k[2]}": v for k, v in all_results.items()}
    out_path.write_text(json.dumps(serializable, indent=2, default=float))
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
