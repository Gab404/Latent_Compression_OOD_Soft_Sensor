#!/usr/bin/env python
"""
compute_sap.py
==============
Calcule le SAP Score (Separability and Predictability) pour l'espace latent.
Permet de comparer dynamiquement le CVAE, le Beta-VAE Light, le Beta-VAE Heavy et le CDAE.

Usage:
    python scripts/compute_sap.py --model cvae
    python scripts/compute_sap.py --model bvae_light
    python scripts/compute_sap.py --model bvae_heavy
    python scripts/compute_sap.py --model cdae
"""

import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from pathlib import Path

# --- Imports du projet ---
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset import PenicillinDataModule, FAULT_COL
from src.data.cvae_encoder import CVAERamanEncoderV2
from src.data.cdae_encoder import CDAERamanEncoderV2
from src.data.beta_vae_encoder import BetaVAERamanEncoderV2

# --- 1. Configuration et Chemins ---
CSV_PATH = "./100_Batches_IndPenSim_V3.csv"

# Chemins des modèles
PATHS = {
    "cdae": {
        "ckpt": "/home/gabriel/ReaKt/outputs/seed_sweep/cdae_pi_lstm/p2/encoders/cdae_s42.pt",
        "scaler": "/home/gabriel/ReaKt/outputs/seed_sweep/cdae_pi_lstm/p2/encoders/cdae_scaler_s42.joblib",
        "encoder_class": CDAERamanEncoderV2,
        "title": "CDAE (Convolutional Denoising AE)",
        "output_img": "sap_heatmap_cdae.png"
    },
    "cvae": {
        "ckpt": "/home/gabriel/ReaKt/outputs/seed_sweep/cvae_pi_lstm/p2/encoders/cvae_s42.pt",
        "scaler": "/home/gabriel/ReaKt/outputs/seed_sweep/cvae_pi_lstm/p2/encoders/cvae_scaler_s42.joblib",
        "encoder_class": CVAERamanEncoderV2,
        "title": "CVAE (Standard VAE)",
        "output_img": "sap_heatmap_cvae.png"
    },
    "bvae_light": {
        "ckpt": "/home/gabriel/ReaKt/outputs/seed_sweep/beta_vae_light_fixed_pi_lstm/p2/encoders/beta_vae_light_s42.pt",
        "scaler": "/home/gabriel/ReaKt/outputs/seed_sweep/beta_vae_light_fixed_pi_lstm/p2/encoders/beta_vae_light_scaler_s42.joblib",
        "encoder_class": CVAERamanEncoderV2,
        "title": "Beta-VAE Light (Fixed Beta=0.005)",
        "output_img": "sap_heatmap_bvae_light.png"
    },
    "bvae_heavy": {
        "ckpt": "/home/gabriel/ReaKt/outputs/seed_sweep/beta_vae_pi_lstm/p1/encoders/beta_vae_s42.pt",
        "scaler": "/home/gabriel/ReaKt/outputs/seed_sweep/beta_vae_pi_lstm/p1/encoders/beta_vae_scaler_s42.joblib",
        "encoder_class": BetaVAERamanEncoderV2,
        "title": "Beta-VAE Heavy (Fixed Beta=0.005)",
        "output_img": "sap_heatmap_bvae_heavy.png"
    }
}

# EXACTEMENT comme dans latent_correlation.py
PHYS_COLS = {
    "Penicillin":   "Offline Penicillin concentration(P_offline:P(g L^{-1}))",
    "Glucose":      "Substrate concentration(S:g/L)",
    "Biomass":      "Offline Biomass concentratio(X_offline:X(g L^{-1}))",
    "Temperature":  "Temperature(T:K)",
    "PAA":          "PAA concentration offline(PAA_offline:PAA (g L^{-1}))",
    "NH3":          "NH_3 concentration off-line(NH3_offline:NH3 (g L^{-1}))",
    "Viscosity":    "Viscosity(Viscosity_offline:centPoise)",
}
SPARSE_VARS = {"Biomass", "PAA", "NH3", "Viscosity", "Penicillin"}


def extract_physical(df: pd.DataFrame) -> dict:
    """Return dict of physical variable arrays (float64), NaN where not available."""
    out = {}
    for name, col in PHYS_COLS.items():
        if col not in df.columns:
            print(f"  [WARN] column not found: {col!r} — skipping {name}")
            continue
        vals = df[col].values.astype(np.float64)
        if name in SPARSE_VARS:
            vals[vals <= 0] = np.nan
        n_valid = int(np.isfinite(vals).sum())
        print(f"  {name:12s}: {n_valid:,} valid rows")
        out[name] = vals
    return out


def main():
    # Parsing des arguments
    parser = argparse.ArgumentParser(description="Calcul du SAP Score")
    parser.add_argument("--model", type=str, choices=["cvae", "bvae_light", "bvae_heavy", "cdae"], default="bvae_light",
                        help="Choix de l'architecture à évaluer (cvae, bvae_light, bvae_heavy, ou cdae)")
    args = parser.parse_args()
    
    m_cfg = PATHS[args.model]
    
    print(f"▶ Chargement de l'encodeur {args.model.upper()}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instanciation dynamique de la bonne classe d'encodeur
    encoder = m_cfg["encoder_class"](m_cfg["ckpt"], m_cfg["scaler"], device)
    
    cfg_data = {
        "csv_path": CSV_PATH,
        "use_raman": True,
        "seq_len": 1
    }
    
    print("▶ Chargement du CSV brut pour extraire les variables physiques...")
    df_raw = pd.read_csv(CSV_PATH)
    
    print(f"\n▶ Chargement du dataset (Application de l'encodeur {args.model.upper()} sur les spectres)...")
    dm = PenicillinDataModule(cfg_data, raman_encoder=encoder)
    dm.load()
    
    df_clean = pd.concat(dm.batches_clean, ignore_index=True)
    
    print("\n▶ Récupération des données synchronisées...")
    phys_dict = extract_physical(df_clean)
    
    # Latents : Sélection explicite et garantie des 64 neurones
    latent_cols = [f"_raman_{i}" for i in range(64)]
    Z = df_clean[latent_cols].values
    
    var_names = list(phys_dict.keys())
    n_latents = Z.shape[1]
    n_targets = len(var_names)
    
    print(f"\n▶ Calcul de la matrice de prédictibilité (Latents: {n_latents}, Targets: {n_targets})...")
    score_matrix = np.zeros((n_latents, n_targets))
    
    # --- 3. Calcul des R² (Prédictibilité) avec protection NaN ---
    for j, var_name in enumerate(var_names):
        y_j = phys_dict[var_name]
        
        for i in range(n_latents):
            z_i = Z[:, i]
            
            # Création d'un masque combiné : Vrai seulement si y n'est PAS NaN ET z n'est PAS NaN
            valid_mask = np.logical_and(np.isfinite(y_j), np.isfinite(z_i))
            
            y_j_clean = y_j[valid_mask]
            z_i_clean = z_i[valid_mask].reshape(-1, 1)
            
            if len(y_j_clean) < 10:
                score_matrix[i, j] = 0.0
                continue
                
            reg = LinearRegression().fit(z_i_clean, y_j_clean)
            r2 = reg.score(z_i_clean, y_j_clean)
            score_matrix[i, j] = max(0.0, r2)
            
    # --- 4. Calcul du SAP Score (Séparabilité) ---
    print("\n▶ Calcul des Gaps (Top 1 - Top 2)...")
    sap_scores = []
    
    for j, target_name in enumerate(var_names):
        scores_j = score_matrix[:, j]
        sorted_scores = np.sort(scores_j)[::-1]
        top1 = sorted_scores[0]
        top2 = sorted_scores[1]
        
        gap = top1 - top2
        sap_scores.append(gap)
        
        best_neuron = np.argmax(scores_j)
        print(f"  - {target_name:<15} : Gap = {gap:.3f} (Top1: {top1:.3f} par z_{best_neuron:02d}, Top2: {top2:.3f})")

    final_sap_score = np.mean(sap_scores)
    print(f"\n=========================================")
    print(f"🏆 SAP SCORE {args.model.upper()} FINAL = {final_sap_score:.4f}")
    print(f"=========================================\n")

    # --- 5. Visualisation (Heatmap) ---
    out_img = m_cfg["output_img"]
    print(f"▶ Génération de la Heatmap ({out_img})...")
    plt.figure(figsize=(8, 12))
    
    sns.heatmap(
        score_matrix, 
        cmap="viridis", 
        xticklabels=var_names,
        yticklabels=[f"z_{i}" for i in range(n_latents)],
        cbar_kws={'label': 'R² Score (Predictability)'}
    )
    plt.title(f"Predictability Matrix - SAP Score = {final_sap_score:.3f}\n{m_cfg['title']}")
    plt.xlabel("Variables Physiques (Ground Truth)")
    plt.ylabel("Dimensions Latentes (Neurones)")
    plt.tight_layout()
    plt.savefig(out_img, dpi=300)
    print(f"✅ Heatmap sauvegardée sous '{out_img}'.")

if __name__ == "__main__":
    main()