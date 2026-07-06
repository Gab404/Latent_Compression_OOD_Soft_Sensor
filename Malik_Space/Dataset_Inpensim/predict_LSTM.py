import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import random

print("📊 Lancement de l'Audit Visuel des Performances (LSTM V6)...")

# =====================================================================
# 1. REDÉFINITION DE L'ARCHITECTURE (Pour charger les poids)
# =====================================================================
class CNNFusionV4(nn.Module):
    def __init__(self, phys_dim=4):
        super(CNNFusionV4, self).__init__()
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
    def forward(self, x_raman, x_phys):
        feat_raman = self.raman_branch(x_raman)
        feat_phys = self.phys_branch(x_phys)
        return torch.cat((feat_raman, feat_phys), dim=1)

class ReaktTemporelV6(nn.Module):
    def __init__(self, extracteur_v4, lstm_hidden_dim=64):
        super(ReaktTemporelV6, self).__init__()
        self.extracteur_spatial = extracteur_v4
        self.lstm = nn.LSTM(
            input_size=528, 
            hidden_size=lstm_hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.2
        )
        self.tete_finale = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 16), nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x_raman_seq, x_phys_seq):
        batch_size, seq_len, c, w = x_raman_seq.size()
        x_raman_plat = x_raman_seq.view(batch_size * seq_len, c, w)
        x_phys_plat = x_phys_seq.view(batch_size * seq_len, -1)
        features_spatiales = self.extracteur_spatial(x_raman_plat, x_phys_plat)
        features_temporelles = features_spatiales.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features_temporelles)
        dernier_instant = lstm_out[:, -1, :] 
        return self.tete_finale(dernier_instant)

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

batches_uniques = df_propre['VRAI_Batch_ID'].unique()
random.seed(42) # La garantie que le test set est bien le même !
random.shuffle(batches_uniques)
split_idx = int(len(batches_uniques) * 0.8)
test_batches = batches_uniques[split_idx:]

batch_cible = random.choice(test_batches)
df_cuve = df_propre[df_propre['VRAI_Batch_ID'] == batch_cible].copy()

# =====================================================================
# 3. CHARGEMENT DES OUTILS ET CRÉATION DES FENÊTRES TEMPORELLES
# =====================================================================
scaler_raman = joblib.load('scaler_raman.pkl')
scaler_phys = joblib.load('scaler_phys.pkl')
scaler_y = joblib.load('scaler_y.pkl')

extracteur_spatial = CNNFusionV4(phys_dim=4)
model = ReaktTemporelV6(extracteur_v4=extracteur_spatial, lstm_hidden_dim=64)
# Assure-toi que le nom correspond bien à ton fichier d'entraînement V6
model.load_state_dict(torch.load('reakt_lstm_v6_best.pth'))
model.eval()

# Mise à l'échelle
x_raman_scaled = scaler_raman.transform(df_cuve[colonnes_raman].values)
x_phys_scaled = scaler_phys.transform(df_cuve[colonnes_physiques].values)

# --- La Mécanique du LSTM : Création des Séquences ---
SEQ_LEN = 10
X_ram_seq, X_phy_seq = [], []

for i in range(len(df_cuve) - SEQ_LEN + 1):
    X_ram_seq.append(x_raman_scaled[i : i + SEQ_LEN])
    X_phy_seq.append(x_phys_scaled[i : i + SEQ_LEN])

T_raman = torch.FloatTensor(np.array(X_ram_seq)).unsqueeze(2) # Ajout du canal [batch, seq, canal, features]
T_phys = torch.FloatTensor(np.array(X_phy_seq))

# =====================================================================
# 4. PRÉDICTION ET AFFICHAGE
# =====================================================================
with torch.no_grad():
    predictions_scaled = model(T_raman, T_phys).numpy()

predictions_reelles = scaler_y.inverse_transform(predictions_scaled)

# Pour s'aligner graphiquement, on ignore les 9 premiers instants de la vraie cuve
# (Puisque le LSTM a besoin de 10 points pour pondre sa première vraie prédiction)
valeurs_reelles = df_cuve[colonnes_cibles].values[SEQ_LEN - 1:]
temps = df_cuve['Time (h)'].values[SEQ_LEN - 1:]

mae_bio = np.mean(np.abs(predictions_reelles[:, 0] - valeurs_reelles[:, 0]))
mae_visco = np.mean(np.abs(predictions_reelles[:, 1] - valeurs_reelles[:, 1]))

print(f"\n--- RÉSULTATS POUR LA CUVE TEST N°{batch_cible} ---")
print(f"Erreur Moyenne Biomasse : {mae_bio:.2f} g/L")
print(f"Erreur Moyenne Viscosité: {mae_visco:.2f} cP")

# Création du graphique à double fenêtre
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 1. Tracé de la Biomasse
ax1.plot(temps, valeurs_reelles[:, 0], 'k-', lw=3, label="Vraie Biomasse (Interpolée)")
ax1.plot(temps, predictions_reelles[:, 0], 'b--', lw=2, label="Prédiction IA (LSTM)")
ax1.set_title(f"Évaluation Temporelle de la Biomasse - Cuve {batch_cible} (Test Set)", fontsize=14, fontweight='bold')
ax1.set_ylabel("Biomasse (g/L)", fontsize=12)
ax1.legend(loc="upper left")
ax1.grid(True, linestyle=':', alpha=0.6)

# 2. Tracé de la Viscosité
ax2.plot(temps, valeurs_reelles[:, 1], 'k-', lw=3, label="Vraie Viscosité")
ax2.plot(temps, predictions_reelles[:, 1], 'r--', lw=2, label="Prédiction IA (LSTM)")
ax2.set_title(f"Évaluation Temporelle de la Viscosité - Cuve {batch_cible} (Test Set)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Temps (Heures)", fontsize=12)
ax2.set_ylabel("Viscosité (cP)", fontsize=12)
ax2.legend(loc="upper left")
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()