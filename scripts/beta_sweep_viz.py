import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Définition des valeurs de beta et des chemins vers les CSV
# Adaptez les chemins selon l'arborescence exacte de vos dossiers de log
beta_files = {
    0.0005: "outputs/seed_sweep_beta/beta_0.0005/beta_vae_pi_lstm/p2/results.csv",
    0.001:  "outputs/seed_sweep_beta/beta_0.001/beta_vae_pi_lstm/p2/results.csv",
    0.002:  "outputs/seed_sweep_beta/beta_0.002/beta_vae_pi_lstm/p2/results.csv",
    0.005:  "/home/gabriel/ReaKt/outputs/seed_sweep/beta_vae_pi_lstm/p2/results.csv",
    0.01:   "outputs/seed_sweep_beta/beta_0.01/beta_vae_pi_lstm/p2/results.csv",
    0.02:   "outputs/seed_sweep_beta/beta_0.02/beta_vae_pi_lstm/p2/results.csv",
    0.05:   "outputs/seed_sweep_beta/beta_0.05/beta_vae_pi_lstm/p2/results.csv",
    1.00:   "/home/gabriel/ReaKt/outputs/seed_sweep/cvae_pi_lstm/p2/results.csv"
}

betas = []
means = []
stds = []

# 2. Extraction des données
for beta, file_path in beta_files.items():
    if os.path.exists(file_path):
        # Lecture du CSV en utilisant la colonne 'seed' comme index
        df = pd.read_csv(file_path, index_col='seed')
        
        # Extraction de la moyenne et de l'écart-type pour le RMSE R4
        mean_rmse = df.loc['mean', 'test_pen_rmse']
        std_rmse = df.loc['std', 'test_pen_rmse']
        
        betas.append(beta)
        means.append(mean_rmse)
        stds.append(std_rmse)
    else:
        print(f"⚠️ Fichier introuvable pour beta = {beta} : {file_path}")

# Conversion en arrays numpy pour faciliter les calculs
betas = np.array(betas)
means = np.array(means)
stds = np.array(stds)

# 3. Création du graphique (Style inspiré de temporal_profile.png)
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

color = '#1f77b4' # Bleu scientifique classique (ou mettez '#2ca02c' pour du vert)

# Tracé de la ligne de moyenne
ax.plot(betas, means, color=color, linewidth=2.5, label='Mean RMSE (R4)')

# Ajout de la zone de variance semi-transparente (+/- 1 sigma)
ax.fill_between(betas, 
                means - stds, 
                means + stds, 
                color=color, 
                alpha=0.2, 
                label=r'$\pm 1\sigma$')

# 4. Formatage des axes et du style
# Échelle logarithmique indispensable pour bien visualiser l'écart entre 0.0005 et 0.05
ax.set_xscale('log')

# Étiquettes
ax.set_xlabel(r'Regularization parameter $\beta$ (log scale)', fontsize=12)
ax.set_ylabel('Test RMSE on Faulty Batches (R4) [g/L]', fontsize=12)
ax.set_title('Impact of Latent Regularization on Fault Robustness (U-Curve)', fontsize=14, pad=15)

# Forcer l'affichage exact de vos valeurs beta sur l'axe X
ax.set_xticks(betas)
ax.set_xticklabels([str(b) for b in betas], rotation=45)

# Nettoyage des bordures (supérieure et droite) pour un rendu académique
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Légende
ax.legend(frameon=False, loc='upper left', fontsize=11)

# Sauvegarde et affichage
plt.tight_layout()
plt.savefig('outputs/beta_sweep_u_curve.png', dpi=300, bbox_inches='tight')
print("Graphique sauvegardé sous le nom 'outputs/beta_sweep_u_curve.png'")
plt.show()