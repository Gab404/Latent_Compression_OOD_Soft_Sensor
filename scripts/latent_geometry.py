#!/usr/bin/env python
"""
latent_geometry.py
==================
Propriétés du code latent INVARIANTES PAR ROTATION, pour chaque encodeur,
et lien avec la robustesse aux pannes (RMSE R4).

Deux mesures, calculées sur les latents μ (sans réentraînement) :

  1. Dimensionnalité effective (participation ratio, PR) des latents du
     régime d'entraînement R1. PR = (Σλ)² / Σλ², λ = valeurs propres de la
     covariance. PR faible = code compressé. Calculée deux fois :
       - pr_raw    : sur les μ bruts produits par l'encodeur (compression du
                     code lui-même) ;
       - pr_scaled : après MinMax par dimension ajusté sur R1, c'est-à-dire
                     la géométrie que voit réellement le LSTM. C'est dans cet
                     espace que l'expérience de rotation a été faite.
     Le PR est invariant par rotation, mais PAS par un changement d'échelle
     dimension par dimension : d'où les deux versions.

  2. Déplacement sous panne : distance de Mahalanobis moyenne des latents
     R4 (pannes) par rapport à la distribution des latents R1, divisée par
     la même distance calculée sur R1 lui-même. Ratio ≈ 1 : les pannes
     restent dans la distribution d'entraînement. Ratio >> 1 : les pannes
     poussent le code hors distribution.
     Calculée aussi pour R2 (changement de stratégie de contrôle, sans panne)
     comme point de comparaison.

Le déplacement (shift_r2, shift_r4) est invariant par toute transformation
affine inversible du code (rotation, MinMax par dimension, changement
d'échelle). La version shift_r4_full (ridge) ne l'est qu'approximativement.

Mesure secondaire, NON invariante par rotation (dépend des axes) :
  - oor_r4 : fraction des valeurs latentes R4 hors de la plage [min, max]
    observée sur R1, c'est-à-dire hors [0, 1] après le MinMax vu par le LSTM.

Usage
-----
    PYTHONPATH=. python scripts/latent_geometry.py

Sorties : outputs/latent_geometry.csv, outputs/latent_geometry.png
"""
from __future__ import annotations

import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG : À VÉRIFIER AVANT DE LANCER
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = "./100_Batches_IndPenSim_V3.csv"
ROOT     = "/home/gabriel/ReaKt/outputs/seed_sweep"
V3_DIR   = "./outputs/full_rotate_experiments_v3"   # logs de l'étape rotation (β-VAE)
SEEDS    = [42, 123, 456, 789, 1024]

# Seuil relatif pour les composantes gardées dans la distance de Mahalanobis
# "principale" (λ >= EIG_REL_TOL * λ_max). Les dimensions quasi mortes sont
# exclues de cette version, et incluses dans la version "full" (ridge).
EIG_REL_TOL = 1e-3

R1_IDS = set(range(0, 30))    # régime d'entraînement en Protocole 2
R2_IDS = set(range(30, 60))   # changement de stratégie de contrôle, sans panne

# Un encodeur = un modèle + un seed. rmse_csv : fichier summary.csv du sweep
# (colonnes seed, ..., test_pen_rmse). Chemins VAE / CDAE à vérifier.
ENCODERS = []
for s in SEEDS:
    ENCODERS += [
        dict(model="beta_vae", seed=s, enc_type="beta_vae",
             ckpt=f"{ROOT}/beta_vae_pi_lstm/p2/encoders/beta_vae_s{s}.pt",
             scaler=f"{ROOT}/beta_vae_pi_lstm/p2/encoders/beta_vae_scaler_s{s}.joblib",
             rmse_csv=f"{ROOT}/beta_vae_pi_lstm/p2/results.csv"),
        dict(model="vae", seed=s, enc_type="cvae_v2",
             ckpt=f"{ROOT}/cvae_pi_lstm/p2/encoders/cvae_s{s}.pt",
             scaler=f"{ROOT}/cvae_pi_lstm/p2/encoders/cvae_scaler_s{s}.joblib",
             rmse_csv=f"{ROOT}/cvae_pi_lstm/p2/results.csv"),
        dict(model="cdae", seed=s, enc_type="cdae_v2",
             ckpt=f"{ROOT}/cdae_pi_lstm/p2/encoders/cdae_s{s}.pt",
             scaler=f"{ROOT}/cdae_pi_lstm/p2/encoders/cdae_scaler_s{s}.joblib",
             rmse_csv=f"{ROOT}/cdae_pi_lstm/p2/results.csv"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Encodeurs (mêmes classes que train.py)
# ─────────────────────────────────────────────────────────────────────────────
def build_encoder(enc_type: str, ckpt: str, scaler: str, device):
    if enc_type == "beta_vae":
        from src.data.beta_vae_encoder import BetaVAERamanEncoderV2
        return BetaVAERamanEncoderV2(ckpt, scaler, device)
    if enc_type == "cvae_v2":
        from src.data.cvae_encoder import CVAERamanEncoderV2
        return CVAERamanEncoderV2(ckpt, scaler, device)
    if enc_type == "cdae_v2":
        from src.data.cdae_encoder import CDAERamanEncoderV2
        return CDAERamanEncoderV2(ckpt, scaler, device)
    raise ValueError(enc_type)


# ─────────────────────────────────────────────────────────────────────────────
# Géométrie latente
# ─────────────────────────────────────────────────────────────────────────────
def participation_ratio(Z: np.ndarray) -> float:
    lam = np.clip(np.linalg.eigvalsh(np.cov(Z, rowvar=False)), 0.0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def latent_geometry(Z1: np.ndarray, Z2: np.ndarray, Z4: np.ndarray) -> dict:
    """Z1 : latents R1 (référence), Z2 : R2, Z4 : pannes. Arrays (N, D)."""
    C = np.cov(Z1, rowvar=False)
    lam, V = np.linalg.eigh(C)
    lam = np.clip(lam, 0.0, None)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]

    pr_raw = lam.sum() ** 2 / (lam ** 2).sum()
    k95 = int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.95) + 1)

    mu = Z1.mean(axis=0)

    # Mahalanobis sur les composantes non dégénérées
    keep = lam >= EIG_REL_TOL * lam[0]
    W = V[:, keep] / np.sqrt(lam[keep])
    # Mahalanobis "full" : toutes les dimensions, ridge relatif
    Wf = V / np.sqrt(lam + 1e-3 * lam.mean())

    def mean_dist(Zx, Wx):
        return float(np.linalg.norm((Zx - mu) @ Wx, axis=1).mean())

    d1, d2, d4 = mean_dist(Z1, W), mean_dist(Z2, W), mean_dist(Z4, W)
    f1, f4 = mean_dist(Z1, Wf), mean_dist(Z4, Wf)

    lo, hi = Z1.min(axis=0), Z1.max(axis=0)
    oor = float(((Z4 < lo) | (Z4 > hi)).mean())
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    pr_scaled = participation_ratio((Z1 - lo) / span)

    return dict(
        pr_raw=float(pr_raw), pr_scaled=pr_scaled, k95=k95, k_kept=int(keep.sum()),
        shift_r2=d2 / d1, shift_r4=d4 / d1, shift_r4_full=f4 / f1,
        oor_r4=oor,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RMSE R4 par encodeur
# ─────────────────────────────────────────────────────────────────────────────
FAULT_RE = re.compile(r"Fault\s+\[Penicillin\].*?RMSE=([0-9.]+)")


def rmse_from_v3_logs(seed: int) -> float:
    """β-VAE : moyenne des runs identity (3 seeds LSTM) de l'étape rotation."""
    vals = []
    for log in Path(V3_DIR).glob(f"identity_s{seed}_l*.log"):
        found = FAULT_RE.findall(log.read_text(errors="ignore"))
        if found:
            vals.append(float(found[-1]))
    return float(np.mean(vals)) if vals else np.nan


def rmse_from_summary(csv_path: str, seed: int) -> float:
    p = Path(csv_path)
    if not p.exists():
        return np.nan
    df = pd.read_csv(p)
    df = df[pd.to_numeric(df["seed"], errors="coerce") == seed]
    return float(df["test_pen_rmse"].iloc[0]) if len(df) else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    from src.data.dataset import FAULT_COL

    missing = [e for e in ENCODERS
               if not (Path(e["ckpt"]).exists() and Path(e["scaler"]).exists())]
    if missing:
        print("Encodeurs introuvables (ignorés) :")
        for e in missing:
            print(f"  - {e['model']} s{e['seed']} : {e['ckpt']}")
    todo = [e for e in ENCODERS if e not in missing]
    if not todo:
        raise SystemExit("Aucun encodeur trouvé : vérifie ROOT et les chemins.")

    print(f"\nLecture de {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    t = df["Time (h)"].values
    bid = np.concatenate([[0], np.cumsum(np.diff(t) < 0)])
    fault_batch = (pd.Series(df[FAULT_COL].values)
                   .groupby(bid).transform("max").values > 0)
    m_r1 = np.isin(bid, list(R1_IDS)) & ~fault_batch
    m_r2 = np.isin(bid, list(R2_IDS)) & ~fault_batch
    m_r4 = fault_batch
    print(f"  lignes R1={m_r1.sum():,}  R2={m_r2.sum():,}  R4={m_r4.sum():,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for e in todo:
        print(f"\n>>> {e['model']}  seed={e['seed']}")
        enc = build_encoder(e["enc_type"], e["ckpt"], e["scaler"], device)
        Z = np.asarray(enc.encode_dataframe(df), dtype=np.float64)
        del enc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # lignes sans spectre (vecteur latent tout à zéro) ou non finies : exclues
        valid = np.isfinite(Z).all(axis=1) & ~(Z == 0).all(axis=1)
        g = latent_geometry(Z[m_r1 & valid], Z[m_r2 & valid], Z[m_r4 & valid])

        if e["model"] == "beta_vae":
            r4 = rmse_from_v3_logs(e["seed"])
            if np.isnan(r4):
                r4 = rmse_from_summary(e["rmse_csv"], e["seed"])
        else:
            r4 = rmse_from_summary(e["rmse_csv"], e["seed"])

        row = dict(model=e["model"], seed=e["seed"], rmse_r4=r4, **g)
        rows.append(row)
        print(f"    PR raw={g['pr_raw']:.2f}  PR scaled={g['pr_scaled']:.2f}  k95={g['k95']}  "
              f"shift R2={g['shift_r2']:.2f}  shift R4={g['shift_r4']:.2f}  "
              f"(full {g['shift_r4_full']:.2f})  oor R4={g['oor_r4']:.3f}  "
              f"RMSE R4={r4:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv("outputs/latent_geometry.csv", index=False)

    # ── Résumé par modèle ────────────────────────────────────────────────────
    cols = ["pr_raw", "pr_scaled", "k95", "shift_r2", "shift_r4", "shift_r4_full", "oor_r4", "rmse_r4"]
    print("\n=== Moyenne ± écart-type par modèle ===")
    summ = res.groupby("model")[cols].agg(["mean", "std"])
    with pd.option_context("display.float_format", "{:.3f}".format,
                           "display.width", 200):
        print(summ)

    # ── Corrélations avec la RMSE R4 ─────────────────────────────────────────
    def corr_block(sub: pd.DataFrame, label: str):
        sub = sub.dropna(subset=["rmse_r4"])
        print(f"\n=== Corrélation de Spearman avec RMSE R4 : {label} (n={len(sub)}) ===")
        if len(sub) < 4:
            print("  pas assez de points")
            return
        for c in ["pr_raw", "pr_scaled", "shift_r4", "shift_r4_full", "oor_r4", "shift_r2"]:
            rho, p = stats.spearmanr(sub[c], sub["rmse_r4"])
            print(f"  {c:>14} : rho = {rho:+.2f}   p = {p:.3f}")

    corr_block(res[res["model"] == "beta_vae"], "encodeurs β-VAE seulement")
    corr_block(res, "tous les encodeurs")

    # ── Figure ───────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"beta_vae": "tab:green", "vae": "tab:blue", "cdae": "tab:red"}
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, x, xl in [(axes[0], "pr_raw", "Dimensionnalité effective du code (PR, μ bruts)"),
                          (axes[1], "shift_r4", "Déplacement sous panne (Mahalanobis R4 / R1)")]:
            for m, sub in res.groupby("model"):
                ax.scatter(sub[x], sub["rmse_r4"], label=m, color=colors.get(m), s=40)
                for _, r in sub.iterrows():
                    ax.annotate(str(r["seed"]), (r[x], r["rmse_r4"]),
                                fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel(xl)
            ax.set_ylabel("RMSE R4 (g/L)")
            ax.grid(alpha=0.3)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig("outputs/latent_geometry.png", dpi=200)
        print("\nFigure : outputs/latent_geometry.png")
    except Exception as exc:
        print(f"\n(figure non générée : {exc})")

    print("Résultats : outputs/latent_geometry.csv")


if __name__ == "__main__":
    main()
