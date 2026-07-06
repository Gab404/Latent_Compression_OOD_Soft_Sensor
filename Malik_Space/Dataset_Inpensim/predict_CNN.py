import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import random

print("📊 Lancement du Benchmark Temporel (CNN 1D V4)...")

# =====================================================================
# 1. REDÉFINITION DE L'ARCHITECTURE CNN 1D V4
# =====================================================================
class FusionModel(nn.Module):
    def __init__(self, phys_dim):
        super(FusionModel, self).__init__()
        self.raman_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=2, padding=5), nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten()
        )
        self.phys_branch = nn.Sequential(
            nn.Linear(phys_dim, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU()
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(528, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x_raman, x_phys):
        feat_raman = self.raman_branch(x_raman)
        feat_phys = self.phys_branch(x_phys)
        combined = torch.cat((feat_raman, feat_phys), dim=1)
        return self.fusion_head(combined)

# =====================================================================
# 2. PRÉPARATION DU DATASET ET ISOLEMENT D'UNE CUVE DE TEST
# =====================================================================
df = pd.read_csv('IndPenSim_100_Batches.csv')
df = df.rename(columns={'Batch ID': 'Volume (L)'})
df['VRAI_Batch_ID'] = (df['Time (h)'] < df['Time (h)'].shift(1)).cumsum() + 1
df = df.sort_values(by=['VRAI_Batch_ID', 'Time (h)']).reset_index(drop=True)

colonnes_raman = df.columns[39:]
colonnes_physiques = ['pH(pH:pH)', 'Temperature(T:K)', 'Agitator RPM(RPM:RPM)', 'Sugar feed rate(Fs:L/h)']
colonnes_cibles = ['Offline Biomass concentratio(X_offline:X(g L^{-1}))', 'Viscosity(Viscosity_offline:centPoise)']

df[colonnes_cibles] = df.groupby('VRAI_Batch_ID')[colonnes_cibles].transform(
    lambda cuve: cuve.interpolate(method='linear', limit_direction='both')
)
df_propre = df.dropna(subset=colonnes_cibles, how='any').dropna(subset=colonnes_raman, how='all').fillna(0)

# RECRÉATION EXACTE DU SPLIT POUR ISOLER LE TEST SET
batches_uniques = df_propre['VRAI_Batch_ID'].unique()
random.seed(42) # La garantie absolue que c'est le même coffre-fort
random.shuffle(batches_uniques)
split_idx = int(len(batches_uniques) * 0.8)
test_batches = batches_uniques[split_idx:]

# Sélection aléatoire d'une cuve dans le Test Set
batch_cible = random.choice(test_batches)
df_cuve = df_propre[df_propre['VRAI_Batch_ID'] == batch_cible].copy()
temps = df_cuve['Time (h)'].values

# =====================================================================
# 3. CHARGEMENT DES OUTILS (SCALERS ET POIDS)
# =====================================================================
scaler_raman = joblib.load('scaler_raman.pkl')
scaler_phys = joblib.load('scaler_phys.pkl')
scaler_y = joblib.load('scaler_y.pkl')

model = FusionModel(phys_dim=4)
model.load_state_dict(torch.load('reakt_fusion_v4_best.pth'))
model.eval()

# =====================================================================
# 4. PRÉDICTION SUR TOUTE LA DURÉE DE LA CUVE
# =====================================================================
# Mise à l'échelle des données de la cuve sélectionnée
x_raman_scaled = scaler_raman.transform(df_cuve[colonnes_raman].values)
x_phys_scaled = scaler_phys.transform(df_cuve[colonnes_physiques].values)

# Conversion Tenseurs (Format compatible CNN)
T_raman = torch.FloatTensor(x_raman_scaled).unsqueeze(1)
T_phys = torch.FloatTensor(x_phys_scaled)

with torch.no_grad():
    predictions_scaled = model(T_raman, T_phys).numpy()

# Remise à l'échelle humaine (g/L et centiPoise)
predictions_reelles = scaler_y.inverse_transform(predictions_scaled)
valeurs_reelles = df_cuve[colonnes_cibles].values

# Calcul de l'erreur absolue moyenne (MAE) spécifique à cette cuve
mae_bio = np.mean(np.abs(predictions_reelles[:, 0] - valeurs_reelles[:, 0]))
mae_visco = np.mean(np.abs(predictions_reelles[:, 1] - valeurs_reelles[:, 1]))

print(f"\n--- RÉSULTATS POUR LA CUVE TEST N°{batch_cible} ---")
print(f"Erreur Moyenne Biomasse : {mae_bio:.2f} g/L")
print(f"Erreur Moyenne Viscosité: {mae_visco:.2f} cP")

# =====================================================================
# 5. AFFICHAGE GRAPHIQUE
# =====================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Tracé de la Biomasse
ax1.plot(temps, valeurs_reelles[:, 0], 'k-', lw=3, label="Vraie Biomasse (Interpolée)")
ax1.plot(temps, predictions_reelles[:, 0], 'b--', lw=2, label="Prédiction IA (CNN 1D)")
ax1.set_title(f"Évaluation Temporelle de la Biomasse - Cuve {batch_cible} (Test Set)", fontsize=14, fontweight='bold')
ax1.set_ylabel("Biomasse (g/L)", fontsize=12)
ax1.legend(loc="upper left")
ax1.grid(True, linestyle=':', alpha=0.6)

# Tracé de la Viscosité
ax2.plot(temps, valeurs_reelles[:, 1], 'k-', lw=3, label="Vraie Viscosité")
ax2.plot(temps, predictions_reelles[:, 1], 'r--', lw=2, label="Prédiction IA (CNN 1D)")
ax2.set_title(f"Évaluation Temporelle de la Viscosité - Cuve {batch_cible} (Test Set)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Temps (Heures)", fontsize=12)
ax2.set_ylabel("Viscosité (cP)", fontsize=12)
ax2.legend(loc="upper left")
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()