#!/usr/bin/env python3
"""
compute_pr_k95_sweep.py
========================
Gabriel GUIET-DUPRÉ

Compute the participation ratio (PR) and k95 of the beta-VAE latent code,
per beta value, averaged over the 5 encoder seeds, and merge with the R4
test RMSE already reported in each beta's results.csv (Protocol 2).

Reuses load_raman_splits() and _apply_sg() from scripts/train_autoencoder.py
so the training split reconstructed here is IDENTICAL to the one seen by
that checkpoint during training (same --seed -> same np.random.default_rng
permutation -> same 80/20 batch split).

Run this from the ReaKt project root (same level as scripts/, src/ and the
IndPenSim CSV), e.g.:

    PYTHONPATH=. python scripts/compute_pr_k95_sweep.py

Output: outputs/beta_sweep_pr_k95.csv, one row per beta, with PR/k95 mean+/-std
over seeds and the matching R4 RMSE mean+/-std pulled from results.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

import train_autoencoder as ta          # scripts/train_autoencoder.py
from src.autoencoder.model import BetaVAE_Raman

# ── Config ───────────────────────────────────────────────────────────────────
CSV_PATH = str(_ROOT / "100_Batches_IndPenSim_V3.csv")
LATENT_DIM = 64
SEEDS = [42, 123, 456, 789, 1024]
DEVICE = "cpu"          # forward-pass only, CPU is enough for this
ENC_BATCH = 2048

beta_files = {
    0.0005: "outputs/seed_sweep_beta/beta_0.0005/beta_vae_pi_lstm/p2/results.csv",
    0.001:  "outputs/seed_sweep_beta/beta_0.001/beta_vae_pi_lstm/p2/results.csv",
    0.002:  "outputs/seed_sweep_beta/beta_0.002/beta_vae_pi_lstm/p2/results.csv",
    0.005:  "/home/gabriel/ReaKt/outputs/seed_sweep/beta_vae_pi_lstm/p2/results.csv",
    0.01:   "outputs/seed_sweep_beta/beta_0.01/beta_vae_pi_lstm/p2/results.csv",
    0.02:   "outputs/seed_sweep_beta/beta_0.02/beta_vae_pi_lstm/p2/results.csv",
    0.05:   "outputs/seed_sweep_beta/beta_0.05/beta_vae_pi_lstm/p2/results.csv",
}


# ── PR / k95 ─────────────────────────────────────────────────────────────────
def _pr_k95(mu: np.ndarray) -> tuple[float, int]:
    """
    Participation ratio PR = (sum lambda)^2 / sum(lambda^2) and k95 (number
    of PCs needed for 95% of variance), from a (N, latent_dim) matrix of
    posterior means mu.
    """
    mu_c = mu - mu.mean(axis=0, keepdims=True)
    cov = np.cov(mu_c, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)
    eigvals = np.sort(eigvals)[::-1]
    eigvals = eigvals[eigvals > 1e-12]
    if eigvals.size == 0:
        return float("nan"), 0
    pr = float((eigvals.sum() ** 2) / (eigvals ** 2).sum())
    cum = np.cumsum(eigvals) / eigvals.sum()
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    return pr, k95


# ── Encode one seed's training split with its own checkpoint + scaler ────────
def _encode_seed(enc_dir: Path, seed: int) -> np.ndarray | None:
    ckpt_path = enc_dir / f"beta_vae_s{seed}.pt"
    scaler_path = enc_dir / f"beta_vae_scaler_s{seed}.joblib"
    if not ckpt_path.exists() or not scaler_path.exists():
        print(f"    [seed={seed}] missing checkpoint or scaler in {enc_dir}, skipping")
        return None

    # Same seed -> same permutation -> same train/val split as at training time.
    X_train_raw, _ = ta.load_raman_splits(CSV_PATH, seed=seed, regime_ablation=True)
    X_train = ta._apply_sg(np.clip(np.nan_to_num(X_train_raw, nan=0.0), 0.0, None))

    scaler = joblib.load(scaler_path)
    X_train = scaler.transform(X_train).astype(np.float32)

    state = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    latent_dim = state.get("latent_dim", LATENT_DIM)
    model = BetaVAE_Raman(input_length=ta.N_RAMAN_COLS, latent_dim=latent_dim)
    model.load_state_dict(state["model_state_dict"])
    model.to(DEVICE).eval()

    mus = []
    with torch.no_grad():
        for i in range(0, len(X_train), ENC_BATCH):
            xb = torch.from_numpy(X_train[i:i + ENC_BATCH]).to(DEVICE)
            _, mu, _ = model(xb)
            mus.append(mu.cpu().numpy())
    return np.concatenate(mus, axis=0)


# ── RMSE from the already-computed results.csv ────────────────────────────────
def _rmse_from_csv(results_csv: Path) -> tuple[float, float]:
    if not results_csv.exists():
        print(f"  WARNING: {results_csv} not found, RMSE left as NaN")
        return float("nan"), float("nan")
    df = pd.read_csv(results_csv)
    mean_row = df[df["seed"].astype(str) == "mean"]
    std_row = df[df["seed"].astype(str) == "std"]
    rmse_mean = float(mean_row["test_pen_rmse"].iloc[0]) if not mean_row.empty else float("nan")
    rmse_std = float(std_row["test_pen_rmse"].iloc[0]) if not std_row.empty else float("nan")
    return rmse_mean, rmse_std


def main() -> None:
    rows = []
    for beta, results_csv in beta_files.items():
        results_csv = Path(results_csv)
        enc_dir = results_csv.parent / "encoders"
        print(f"\n=== beta = {beta} ===")
        print(f"  encoders dir: {enc_dir}")

        pr_list, k95_list = [], []
        for seed in SEEDS:
            mu = _encode_seed(enc_dir, seed)
            if mu is None:
                continue
            pr, k95 = _pr_k95(mu)
            print(f"    seed={seed:<5d} n={mu.shape[0]:<7d} PR={pr:.3f}  k95={k95}")
            pr_list.append(pr)
            k95_list.append(k95)

        rmse_mean, rmse_std = _rmse_from_csv(results_csv)

        rows.append({
            "beta": beta,
            "pr_mean": float(np.mean(pr_list)) if pr_list else float("nan"),
            "pr_std": float(np.std(pr_list)) if pr_list else float("nan"),
            "k95_mean": float(np.mean(k95_list)) if k95_list else float("nan"),
            "k95_std": float(np.std(k95_list)) if k95_list else float("nan"),
            "n_seeds": len(pr_list),
            "rmse_r4_mean": rmse_mean,
            "rmse_r4_std": rmse_std,
        })

    out = pd.DataFrame(rows).sort_values("beta")
    out_path = _ROOT / "outputs" / "beta_sweep_pr_k95.csv"
    out.to_csv(out_path, index=False)

    print("\n" + "=" * 78)
    print(f"{'beta':>8}  {'PR (mean±std)':>16}  {'k95 (mean±std)':>16}  {'RMSE R4':>16}")
    print("-" * 78)
    for _, r in out.iterrows():
        print(
            f"{r['beta']:>8}  "
            f"{r['pr_mean']:>7.2f}±{r['pr_std']:<7.2f}  "
            f"{r['k95_mean']:>6.1f}±{r['k95_std']:<7.1f}  "
            f"{r['rmse_r4_mean']:>7.3f}±{r['rmse_r4_std']:<7.3f}"
        )
    print("=" * 78)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
