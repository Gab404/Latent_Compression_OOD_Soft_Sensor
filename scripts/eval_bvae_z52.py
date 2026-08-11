"""
scripts/eval_bvae_z52.py
========================
Use a single latent dimension of the frozen β-VAE (β=0.05) as a direct
Penicillin predictor via a simple 1-D linear fit:

    y_hat = a · z_dim + b

    where  z_dim = μ[dim]  (posterior mean from the frozen encoder, no grad)
           a, b  = least-squares fit on R1 sparse points

This tests the hypothesis that z52 (the active dim with r(z52, Pen)=+0.993)
is sufficient for penicillin prediction with zero learnable capacity beyond
a single scale + bias.

Usage
-----
    PYTHONPATH=. python scripts/eval_bvae_z52.py
    PYTHONPATH=. python scripts/eval_bvae_z52.py --dim 52
    PYTHONPATH=. python scripts/eval_bvae_z52.py --beta 0.5 --dim 49
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ── Project root ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.autoencoder.model import BetaVAE_Raman
from src.data.dataset import RAMAN_START_COL_IDX, TARGET_COLS

# ── Config ─────────────────────────────────────────────────────────────────────
CSV_PATH   = "./100_Batches_IndPenSim_V3.csv"
SWEEP_DIR  = Path("./outputs/beta_sweep_e2e/checkpoints")
OUTPUT_DIR = Path("./outputs/bvae_z_linear")

N_RAMAN        = 2001
LATENT_DIM     = 64
SG_WINDOW, SG_POLY, SG_DERIV = 15, 2, 1
MIN_BATCH_LEN  = 50

REGIMES = {
    "R1":    list(range(0,   30)),
    "R2":    list(range(30,  60)),
    "R3":    list(range(60,  90)),
    "Fault": list(range(90, 100)),
}


# ── Data helpers ───────────────────────────────────────────────────────────────

def _sg(X: np.ndarray) -> np.ndarray:
    return savgol_filter(X, window_length=SG_WINDOW,
                         polyorder=SG_POLY, deriv=SG_DERIV, axis=1)


def load_batches(csv_path: str) -> Dict[str, List[Dict]]:
    print(f"\n[DATA]  Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"[DATA]  Shape: {df.shape[0]:,} × {df.shape[1]}")

    t_vals    = df["Time (h)"].values
    batch_ids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t_vals[i] < t_vals[i - 1]:
            bid += 1
        batch_ids[i] = bid
    df["_batch_id"] = batch_ids

    all_cols    = df.columns.tolist()
    raman_names = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + N_RAMAN]
    pen_col     = TARGET_COLS[1]

    regime_id_sets = {r: set(ids) for r, ids in REGIMES.items()}
    buckets: Dict[str, List[Dict]] = {r: [] for r in REGIMES}

    for b_id in range(bid + 1):
        b = df[df["_batch_id"] == b_id].copy().reset_index(drop=True)
        if len(b) < MIN_BATCH_LEN:
            continue
        y_sparse = b[pen_col].values.copy()
        X_raw    = b[raman_names].values.astype(np.float64)
        X_raw    = np.clip(np.nan_to_num(X_raw, 0.0), 0.0, None)
        mask     = np.abs(X_raw).sum(axis=1) > 0
        if mask.sum() < MIN_BATCH_LEN:
            continue
        entry = {"X_sg": _sg(X_raw[mask]), "y_sparse": y_sparse[mask], "batch_id": b_id}
        for regime, ids in regime_id_sets.items():
            if b_id in ids:
                buckets[regime].append(entry)
                break

    for r, bl in buckets.items():
        print(f"[DATA]  {r}: {len(bl)} batches")
    return buckets


# ── Encoder ────────────────────────────────────────────────────────────────────

def load_encoder(beta: float, device: torch.device) -> Tuple[BetaVAE_Raman, object]:
    beta_str    = str(beta)
    vae_path    = SWEEP_DIR / f"b{beta_str}" / "vae.pt"
    scaler_path = SWEEP_DIR / f"b{beta_str}" / "vae_scaler.joblib"
    if not vae_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {vae_path}")
    state  = torch.load(vae_path, map_location="cpu", weights_only=False)
    model  = BetaVAE_Raman(input_length=N_RAMAN, latent_dim=LATENT_DIM)
    model.load_state_dict(state["model_state_dict"])
    model  = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    scaler = joblib.load(scaler_path)
    print(f"[ENC]   Loaded β-VAE (β={beta}) — FROZEN")
    return model, scaler


@torch.no_grad()
def get_mu(X_sg: np.ndarray, encoder: BetaVAE_Raman, scaler, device: torch.device) -> np.ndarray:
    X_scaled = scaler.transform(X_sg).astype(np.float32)
    mu, _    = encoder.encode(torch.from_numpy(X_scaled).to(device))
    return mu.cpu().numpy()   # (N, LATENT_DIM)


def collect_sparse(
    batches: List[Dict],
    encoder: BetaVAE_Raman,
    scaler,
    device:  torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (mu_all, y_all, batch_ids_all) at sparse measurement points."""
    mus, ys, bids = [], [], []
    for b in batches:
        mask = ~np.isnan(b["y_sparse"])
        if mask.sum() == 0:
            continue
        mu = get_mu(b["X_sg"][mask], encoder, scaler, device)
        mus.append(mu)
        ys.append(b["y_sparse"][mask])
        bids.append(np.full(mask.sum(), b["batch_id"], dtype=np.int32))
    return np.vstack(mus), np.concatenate(ys), np.concatenate(bids)


# ── Pearson r across all dims ──────────────────────────────────────────────────

def pearson_all_dims(mu: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute Pearson r between each latent dim and y. Returns (LATENT_DIM,)."""
    r = np.array([
        float(np.corrcoef(mu[:, d], y)[0, 1])
        for d in range(mu.shape[1])
    ])
    return r


# ── Evaluation ─────────────────────────────────────────────────────────────────

def eval_linear_dim(
    dim:      int,
    lr_model: LinearRegression,
    mu_all:   np.ndarray,
    y_all:    np.ndarray,
    bid_all:  np.ndarray,
    label:    str,
) -> Dict:
    z    = mu_all[:, dim].reshape(-1, 1)
    pred = lr_model.predict(z).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_all, pred)))
    mae  = float(np.mean(np.abs(y_all - pred)))
    r2   = float(r2_score(y_all, pred))
    n    = len(y_all)
    print(f"  [{label:<20}]  RMSE={rmse:.4f} g/L  MAE={mae:.4f}  R²={r2:.4f}  n={n}")

    # Per-batch breakdown
    per_batch = []
    for b_id in np.unique(bid_all):
        m = bid_all == b_id
        if m.sum() == 0:
            continue
        pb_rmse = float(np.sqrt(mean_squared_error(y_all[m], pred[m])))
        per_batch.append({"batch_id": int(b_id), "RMSE": pb_rmse, "n": int(m.sum())})

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "n": n, "per_batch": per_batch}


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_scatter(
    dim:      int,
    lr_model: LinearRegression,
    splits:   Dict[str, Tuple[np.ndarray, np.ndarray]],  # label → (mu, y)
    save_path: str,
    beta:      float,
) -> None:
    """Scatter z_dim vs Penicillin for each regime + fitted line."""
    colors = {"R1 (train)": "#2980b9", "R2 (OOD)": "#27ae60",
              "R3 (OOD)": "#8e44ad", "Fault (OOD)": "#e74c3c"}
    markers = {"R1 (train)": "o", "R2 (OOD)": "s", "R3 (OOD)": "^", "Fault (OOD)": "D"}

    fig, ax = plt.subplots(figsize=(7, 5))

    all_z = np.concatenate([mu[:, dim] for mu, _ in splits.values()])
    z_line = np.linspace(all_z.min(), all_z.max(), 200).reshape(-1, 1)
    y_line = lr_model.predict(z_line).ravel()
    ax.plot(z_line, y_line, "k--", linewidth=2, label="Linear fit (R1 train)", zorder=5)

    for label, (mu, y) in splits.items():
        z = mu[:, dim]
        r = float(np.corrcoef(z, y)[0, 1])
        ax.scatter(z, y, s=12, alpha=0.5, color=colors[label],
                   marker=markers[label], label=f"{label}  r={r:+.3f}", zorder=4)

    ax.set_xlabel(f"z{dim}  (latent dim {dim})", fontsize=11)
    ax.set_ylabel("Penicillin (g/L)", fontsize=11)
    ax.set_title(f"β-VAE (β={beta})  z{dim} → Penicillin  (1-D linear fit)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_corr_bar(
    r_vals:    np.ndarray,
    dim:       int,
    save_path: str,
    beta:      float,
    top_n:     int = 20,
) -> None:
    """Bar chart of top-N |r| dimensions, highlighting the chosen dim."""
    order   = np.argsort(np.abs(r_vals))[::-1][:top_n]
    labels  = [f"z{d}" for d in order]
    heights = r_vals[order]
    colors  = ["#e74c3c" if d == dim else "#95a5a6" for d in order]

    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.6), 4))
    ax.bar(range(top_n), heights, color=colors, alpha=0.85)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_ylabel("Pearson r  (z_dim vs Penicillin)", fontsize=10)
    ax.set_title(f"β-VAE (β={beta}) — Top {top_n} latent dims by |r| with Penicillin  "
                 f"[red = z{dim}]", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_summary_bar(results: Dict, dim: int, beta: float, save_path: str) -> None:
    """RMSE per split + PLS R1 reference diamonds."""
    pls_ref = {"R1 (train)": 0.2463, "R2 (OOD)": 0.2683,
               "R3 (OOD)": 0.2604, "Fault (OOD)": 0.2555}
    mlp_ref = {"R1 (train)": 0.4454, "R2 (OOD)": 0.5238,
               "R3 (OOD)": 0.4926, "Fault (OOD)": 0.5486}

    splits = list(results.keys())
    rmses  = [results[s]["RMSE"] for s in splits]
    x      = np.arange(len(splits))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, rmses, color="#e67e22", alpha=0.85, label=f"β-VAE z{dim} linear")
    for bar, v in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    for i, s in enumerate(splits):
        if s in pls_ref:
            ax.plot(i, pls_ref[s], "D", color="#2980b9", markersize=9, zorder=5,
                    label="PLS K=64 (R1)" if i == 0 else "_nolegend_")
        if s in mlp_ref:
            ax.plot(i, mlp_ref[s], "^", color="#8e44ad", markersize=9, zorder=5,
                    label="β-VAE+MLP (R1)" if i == 0 else "_nolegend_")

    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=10)
    ax.set_ylabel("Penicillin RMSE (g/L)", fontsize=10)
    ax.set_title(f"β-VAE (β={beta}) z{dim} linear  vs  MLP head  vs  PLS",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_time_series(
    dim:      int,
    lr_model: LinearRegression,
    batch:    Dict,
    encoder:  BetaVAE_Raman,
    scaler,
    device:   torch.device,
    save_path: str,
    beta:      float,
) -> None:
    """Overlay z_dim linear prediction vs ground truth for one batch."""
    mu_all = get_mu(batch["X_sg"], encoder, scaler, device)   # (T, D)
    z      = mu_all[:, dim].reshape(-1, 1)
    y_pred = lr_model.predict(z).ravel()

    # Sparse observations
    sparse_mask = ~np.isnan(batch["y_sparse"])
    t_all   = np.arange(len(y_pred))
    t_sp    = t_all[sparse_mask]
    y_sp    = batch["y_sparse"][sparse_mask]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Top: prediction vs observations
    ax = axes[0]
    ax.plot(t_all, y_pred, color="#e67e22", linewidth=1.8,
            label=f"z{dim} linear prediction")
    ax.scatter(t_sp, y_sp, color="#2c3e50", s=30, zorder=5,
               label="Offline measurement")
    ax.set_ylabel("Penicillin (g/L)", fontsize=10)
    ax.set_title(f"β-VAE (β={beta})  z{dim} linear  —  batch {batch['batch_id']+1}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # Bottom: raw z_dim trajectory
    ax = axes[1]
    ax.plot(t_all, mu_all[:, dim], color="#e74c3c", linewidth=1.5)
    ax.set_ylabel(f"z{dim} (μ)", fontsize=10)
    ax.set_xlabel("Time step", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(csv_path: str = CSV_PATH, beta: float = 0.05, dim: int = 52) -> None:
    out_dir = OUTPUT_DIR / f"beta{beta}_z{dim}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INIT]  device={device}  β={beta}  target dim=z{dim}")

    # ── 1. Load frozen encoder ─────────────────────────────────────────────────
    encoder, scaler = load_encoder(beta, device)

    # ── 2. Load data ───────────────────────────────────────────────────────────
    buckets = load_batches(csv_path)

    # ── 3. Encode all sparse points ────────────────────────────────────────────
    print("\n[ENC]   Encoding sparse points for each regime ...")
    encoded = {}
    for regime, batches in buckets.items():
        mu, y, bid = collect_sparse(batches, encoder, scaler, device)
        encoded[regime] = (mu, y, bid)
        r = float(np.corrcoef(mu[:, dim], y)[0, 1])
        print(f"  {regime:6s}  n={len(y)}  r(z{dim}, Pen)={r:+.4f}")

    # ── 4. Pearson r across ALL dims (on R1 data) ──────────────────────────────
    mu_r1, y_r1, bid_r1 = encoded["R1"]
    r_all = pearson_all_dims(mu_r1, y_r1)
    top5  = np.argsort(np.abs(r_all))[::-1][:5]
    print(f"\n[CORR]  Top-5 dims by |r| on R1 sparse points:")
    for d in top5:
        print(f"  z{d:>3d}  r={r_all[d]:+.4f}" + (" ← selected" if d == dim else ""))

    # ── 5. Fit 1-D linear regression on R1 ────────────────────────────────────
    z_train = mu_r1[:, dim].reshape(-1, 1)
    lr      = LinearRegression()
    lr.fit(z_train, y_r1)
    a, b_coef = float(lr.coef_[0]), float(lr.intercept_)
    print(f"\n[FIT]   y = {a:.4f} · z{dim} + {b_coef:.4f}  (fit on R1 sparse, n={len(y_r1)})")

    # ── 6. Evaluate on all regimes ─────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  EVALUATION — 1-D linear: y = a·z{dim} + b")
    print(f"{'─'*55}")

    split_labels = ["R1 (train)", "R2 (OOD)", "R3 (OOD)", "Fault (OOD)"]
    regime_keys  = ["R1", "R2", "R3", "Fault"]
    results = {}
    for label, rkey in zip(split_labels, regime_keys):
        mu_r, y_r, bid_r = encoded[rkey]
        m = eval_linear_dim(dim, lr, mu_r, y_r, bid_r, label)
        results[label] = m

    # ── 7. Summary table ───────────────────────────────────────────────────────
    r1_rmse = results["R1 (train)"]["RMSE"]
    sep = "─" * 70
    print(f"\n{'='*70}")
    print(f"  β-VAE (β={beta}) z{dim} → Penicillin  [1-D linear,  a={a:.4f}  b={b_coef:.4f}]")
    print(f"{'='*70}")
    print(f"  {'Split':<22}  {'RMSE':>8}  {'MAE':>8}  {'R²':>8}  {'OOD Δ':>8}")
    print(sep)
    for label in split_labels:
        m   = results[label]
        ood = (m["RMSE"] - r1_rmse) / (r1_rmse + 1e-8) * 100
        ood_s = f"{ood:+.1f}%" if label != "R1 (train)" else "—"
        print(f"  {label:<22}  {m['RMSE']:>8.4f}  {m['MAE']:>8.4f}  {m['R2']:>8.4f}  {ood_s:>8}")
    print(f"{'='*70}")

    # ── 8. Plots ───────────────────────────────────────────────────────────────
    scatter_splits = {
        label: (encoded[rk][0], encoded[rk][1])
        for label, rk in zip(split_labels, regime_keys)
    }
    plot_scatter(dim, lr, scatter_splits,
                 save_path=str(out_dir / f"scatter_z{dim}.png"), beta=beta)
    plot_corr_bar(r_all, dim,
                  save_path=str(out_dir / "corr_bar.png"), beta=beta)
    plot_summary_bar(results, dim, beta,
                     save_path=str(out_dir / "summary_comparison.png"))

    # Time-series plot on a representative batch from each regime
    for rkey, bl in buckets.items():
        if bl:
            rep_batch = bl[len(bl) // 2]   # middle batch of each regime
            plot_time_series(dim, lr, rep_batch, encoder, scaler, device,
                             save_path=str(out_dir / f"timeseries_{rkey.lower()}_batch{rep_batch['batch_id']+1}.png"),
                             beta=beta)

    print(f"\n  Outputs → {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate β-VAE frozen encoder: single latent dim → Penicillin (linear).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv",  default=CSV_PATH)
    parser.add_argument("--beta", type=float, default=0.05,
                        help="β value (must have a trained sweep checkpoint)")
    parser.add_argument("--dim",  type=int, default=52,
                        help="Latent dimension index to use as predictor")
    args = parser.parse_args()
    main(csv_path=args.csv, beta=args.beta, dim=args.dim)
