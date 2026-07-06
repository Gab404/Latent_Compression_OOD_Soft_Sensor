import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import random

print("📊 Lancement de l'Audit Visuel des Performances...")

# =====================================================================
# 1. REDÉFINITION DE L'ARCHITECTURE (Nécessaire pour charger les poids)
# =====================================================================
class PositionalEncoding1D(nn.Module):
    def __init__(self, embed_dim, max_len=500):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, embed_dim))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SelfAttention1D(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class CoAtNetFusion(nn.Module):
    def __init__(self, phys_dim):
        super(CoAtNetFusion, self).__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=2, padding=5), nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU()
        )
        self.pos_encoder = PositionalEncoding1D(embed_dim=32)
        self.attention_block = SelfAttention1D(embed_dim=32, num_heads=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.phys_branch = nn.Sequential(
            nn.Linear(phys_dim, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU()
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(48, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x_raman, x_phys):
        feat_conv = self.conv_stem(x_raman).transpose(1, 2)
        feat_conv = self.pos_encoder(feat_conv) 
        attended = self.attention_block(feat_conv).transpose(1, 2)
        combined = torch.cat((self.pool(attended).squeeze(-1), self.phys_branch(x_phys)), dim=1)
        return self.fusion_head(combined)

# =====================================================================
# 2. PRÉPARATION DU DATASET ET ISOLEMENT DU SET DE TEST
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
df_propre = df.dropna(subset=colonnes_cibles, how='any')
df_propre = df_propre.dropna(subset=colonnes_raman, how='all').fillna(0)

# RECRÉATION EXACTE DU SPLIT POUR ISOLER LE TEST SET
batches_uniques = df_propre['VRAI_Batch_ID'].unique()
random.seed(42) # La garantie que le test set est bien le même !
random.shuffle(batches_uniques)
split_idx = int(len(batches_uniques) * 0.8)
test_batches = batches_uniques[split_idx:]

# Choisissons au hasard une des cuves du Test Set
batch_cible = random.choice(test_batches)
df_cuve = df_propre[df_propre['VRAI_Batch_ID'] == batch_cible].copy()
temps = df_cuve['Time (h)'].values

# =====================================================================
# 3. CHARGEMENT DES OUTILS (SCALERS ET POIDS)
# =====================================================================
scaler_raman = joblib.load('scaler_raman.pkl')
scaler_phys = joblib.load('scaler_phys.pkl')
scaler_y = joblib.load('scaler_y.pkl')

model = CoAtNetFusion(phys_dim=4)
# Assure-toi que le nom correspond à ton fichier d'entraînement (ex: 'reakt_coatnet_v5.pth')
model.load_state_dict(torch.load('reakt_coatnet_v5_best.pth'))
model.eval()

# =====================================================================
# 4. PRÉDICTION ET AFFICHAGE
# =====================================================================
# Mise à l'échelle des données de la cuve de test
x_raman_scaled = scaler_raman.transform(df_cuve[colonnes_raman].values)
x_phys_scaled = scaler_phys.transform(df_cuve[colonnes_physiques].values)

# Conversion pour le modèle
T_raman = torch.FloatTensor(x_raman_scaled).unsqueeze(1)
T_phys = torch.FloatTensor(x_phys_scaled)

with torch.no_grad():
    predictions_scaled = model(T_raman, T_phys).numpy()

# Remise à l'échelle humaine (g/L et centiPoise)
predictions_reelles = scaler_y.inverse_transform(predictions_scaled)
valeurs_reelles = df_cuve[colonnes_cibles].values

# Calcul de l'erreur absolue moyenne (MAE) pour le bilan
mae_bio = np.mean(np.abs(predictions_reelles[:, 0] - valeurs_reelles[:, 0]))
mae_visco = np.mean(np.abs(predictions_reelles[:, 1] - valeurs_reelles[:, 1]))

print(f"\n--- RÉSULTATS POUR LA CUVE TEST N°{batch_cible} ---")
print(f"Erreur Moyenne Biomasse : {mae_bio:.2f} g/L")
print(f"Erreur Moyenne Viscosité: {mae_visco:.2f} cP")

# Création du graphique à double fenêtre
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 1. Tracé de la Biomasse
ax1.plot(temps, valeurs_reelles[:, 0], 'k-', lw=3, label="Vraie Biomasse (Interpolée)")
ax1.plot(temps, predictions_reelles[:, 0], 'b--', lw=2, label="Prédiction IA (CoAtNet)")
ax1.set_title(f"Évaluation de la Biomasse - Cuve {batch_cible} (Test Set)", fontsize=14, fontweight='bold')
ax1.set_ylabel("Biomasse (g/L)", fontsize=12)
ax1.legend(loc="upper left")
ax1.grid(True, linestyle=':', alpha=0.6)

# 2. Tracé de la Viscosité
ax2.plot(temps, valeurs_reelles[:, 1], 'k-', lw=3, label="Vraie Viscosité")
ax2.plot(temps, predictions_reelles[:, 1], 'r--', lw=2, label="Prédiction IA (CoAtNet)")
ax2.set_title(f"Évaluation de la Viscosité - Cuve {batch_cible} (Test Set)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Temps (Heures)", fontsize=12)
ax2.set_ylabel("Viscosité (cP)", fontsize=12)
ax2.legend(loc="upper left")
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()