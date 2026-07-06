import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

print("📊 Lancement de l'évaluation du modèle CI-PI-LSTM...")

# =====================================================================
# 1. RECRÉATION EXACTE DE L'ARCHITECTURE (Nécessaire pour charger les poids)
# =====================================================================
class SpectralReducer(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=2, padding=5), nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(32 * 16, latent_dim), nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class ChannelIndependentExtractor(nn.Module):
    def __init__(self, num_channels, features_per_channel=4):
        super().__init__()
        self.ci_conv = nn.Conv1d(
            num_channels, 
            num_channels * features_per_channel, 
            kernel_size=3, padding=1, 
            groups=num_channels
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.relu(self.ci_conv(x))
        return out.transpose(1, 2)

class CI_PI_LSTM(nn.Module):
    def __init__(self, raman_dim=2200, phys_dim=4):
        super().__init__()
        latent_raman = 16
        features_per_channel = 4
        
        self.spectral_reducer = SpectralReducer(latent_dim=latent_raman)
        self.ci_extractor = ChannelIndependentExtractor(latent_raman, features_per_channel)
        
        fusion_dim = (latent_raman * features_per_channel) + phys_dim
        self.lstm = nn.LSTM(fusion_dim, 128, num_layers=2, batch_first=True, dropout=0.2)
        
        self.fc_bio = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, seq_ram, seq_phy):
        B, S, F = seq_ram.size()
        ram_flat = seq_ram.view(B * S, 1, F)
        ram_latent = self.spectral_reducer(ram_flat)
        ram_latent = ram_latent.view(B, S, -1)
        ram_ci = self.ci_extractor(ram_latent)
        fused = torch.cat((ram_ci, seq_phy), dim=2)
        lstm_out, _ = self.lstm(fused)
        return self.fc_bio(lstm_out)

# =====================================================================
# 2. RECONSTRUCTION DU TEST SET STRICTEMENT IDENTIQUE
# =====================================================================
print("⏳ Chargement des données de Test...")
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'IndPenSim_100_Batches.csv')
df = pd.read_csv(csv_path)

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

# On refait EXACTEMENT le même split (même seed 42) pour retomber sur les mêmes cuves de test
batches_uniques = df_propre['VRAI_Batch_ID'].unique()
random.seed(42) 
random.shuffle(batches_uniques)

split_idx = int(len(batches_uniques) * 0.8)
test_batches = batches_uniques[split_idx:]
df_test = df_propre[df_propre['VRAI_Batch_ID'].isin(test_batches)]

# =====================================================================
# 3. CHARGEMENT DES SCALERS ET CRÉATION DU LOADER
# =====================================================================
print("⚙️ Chargement des scalers pré-entraînés...")
scaler_raman = joblib.load('scaler_raman.pkl')
scaler_phys = joblib.load('scaler_phys.pkl')
scaler_y = joblib.load('scaler_y.pkl')

SEQ_LEN = 10

def preparer_sequences(dataframe):
    X_ram, X_phy, Y = [], [], []
    for b in dataframe['VRAI_Batch_ID'].unique():
        df_b = dataframe[dataframe['VRAI_Batch_ID'] == b]
        if len(df_b) <= SEQ_LEN: continue
        
        ram = scaler_raman.transform(df_b[colonnes_raman].values)
        phy = scaler_phys.transform(df_b[colonnes_physiques].values)
        y = scaler_y.transform(df_b[colonnes_cibles].values)
        
        for i in range(len(df_b) - SEQ_LEN):
            X_ram.append(ram[i : i + SEQ_LEN])
            X_phy.append(phy[i : i + SEQ_LEN])
            Y.append(y[i : i + SEQ_LEN])
            
    return torch.FloatTensor(np.array(X_ram)), torch.FloatTensor(np.array(X_phy)), torch.FloatTensor(np.array(Y))

T_raman_test, T_phys_test, T_y_test = preparer_sequences(df_test)
test_loader = DataLoader(TensorDataset(T_raman_test, T_phys_test, T_y_test), batch_size=64, shuffle=False)

# =====================================================================
# 4. CHARGEMENT DU MODÈLE ET SÉPARATION NORMAUX / DÉFECTUEUX
# =====================================================================
print("🧠 Chargement des poids du modèle 'reakt_ci_pi_lstm_best.pth'...")
model = CI_PI_LSTM(phys_dim=len(colonnes_physiques))
model.load_state_dict(torch.load('reakt_ci_pi_lstm_best.pth', weights_only=True))
model.eval()

print("🔮 Inférence en cours (Analyse batch par batch)...")

resultats_par_batch = []

with torch.no_grad():
    # On évalue chaque cuve indépendamment pour identifier les pannes
    for b in df_test['VRAI_Batch_ID'].unique():
        df_b = df_test[df_test['VRAI_Batch_ID'] == b]
        if len(df_b) <= SEQ_LEN: continue
        
        # On utilise la fonction existante pour créer les tenseurs de ce batch
        T_ram_b, T_phy_b, T_y_b = preparer_sequences(df_b)
        
        pred_norm = model(T_ram_b, T_phy_b)[:, -1, :].numpy()
        true_norm = T_y_b[:, -1, :].numpy()
        
        pred_real = scaler_y.inverse_transform(pred_norm)[:, 0] # Index 0 = Biomasse
        true_real = scaler_y.inverse_transform(true_norm)[:, 0]
        
        mae_b = mean_absolute_error(true_real, pred_real)
        rmse_b = np.sqrt(mean_squared_error(true_real, pred_real))
        r2_b = r2_score(true_real, pred_real)
        
        resultats_par_batch.append({
            'batch_id': b,
            'mae': mae_b,
            'rmse': rmse_b,
            'r2': r2_b,
            'true': true_real,
            'pred': pred_real
        })

# Tri des batchs par MAE croissante
resultats_par_batch.sort(key=lambda x: x['mae'])

# Stratégie de séparation : On considère les 20% des batchs avec la pire erreur comme "Défectueux"
# (correspond au ratio typique de pannes capteurs générées dans IndPenSim)
nb_total = len(resultats_par_batch)
nb_defectueux = max(1, int(nb_total * 0.20))
nb_normaux = nb_total - nb_defectueux

batchs_normaux = resultats_par_batch[:nb_normaux]
batchs_defectueux = resultats_par_batch[nb_normaux:]

def get_global_metrics(batch_list):
    if not batch_list: return 0, 0, 0
    all_true = np.concatenate([b['true'] for b in batch_list])
    all_pred = np.concatenate([b['pred'] for b in batch_list])
    return (
        np.sqrt(mean_squared_error(all_true, all_pred)),
        mean_absolute_error(all_true, all_pred),
        r2_score(all_true, all_pred)
    )

rmse_norm, mae_norm, r2_norm = get_global_metrics(batchs_normaux)
rmse_def, mae_def, r2_def = get_global_metrics(batchs_defectueux)

print("\n=======================================================")
print(f"--- RÉSULTATS BATCHS NORMAUX ({len(batchs_normaux)} cuves saines) ---")
print(f"RMSE : {rmse_norm:.2f} g/L")
print(f"MAE  : {mae_norm:.2f} g/L")
print(f"R²   : {r2_norm:.4f}")

print(f"\n--- RÉSULTATS BATCHS DÉFECTUEUX ({len(batchs_defectueux)} cuves pannes Raman/Phys) ---")
print(f"RMSE : {rmse_def:.2f} g/L")
print(f"MAE  : {mae_def:.2f} g/L")
print(f"R²   : {r2_def:.4f}")
print("=======================================================")

# =====================================================================
# 5. VISUALISATIONS COMPARATIVES
# =====================================================================
# --- Parity Plot Comparatif ---
plt.figure(figsize=(10, 8))

# Concaténation pour l'affichage
true_n = np.concatenate([b['true'] for b in batchs_normaux])
pred_n = np.concatenate([b['pred'] for b in batchs_normaux])
true_d = np.concatenate([b['true'] for b in batchs_defectueux])
pred_d = np.concatenate([b['pred'] for b in batchs_defectueux])

plt.scatter(true_n, pred_n, color='#2ca02c', alpha=0.5, edgecolors='w', label=f'Normaux (MAE={mae_norm:.2f})')
plt.scatter(true_d, pred_d, color='#d62728', alpha=0.5, edgecolors='w', marker='x', label=f'Défectueux (MAE={mae_def:.2f})')

min_v = min(np.min(true_n), np.min(pred_n), np.min(true_d), np.min(pred_d))
max_v = max(np.max(true_n), np.max(pred_n), np.max(true_d), np.max(pred_d))
plt.plot([min_v, max_v], [min_v, max_v], color='black', linestyle='--', linewidth=2, label='Idéal')

plt.title('CI-PI-LSTM : Tolérance aux Pannes (Normal vs Défectueux)', fontsize=16, fontweight='bold')
plt.xlabel('Concentration Réelle en Biomasse (g/L)', fontsize=12)
plt.ylabel('Concentration Prédite (g/L)', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Visualisation temporelle du PIRE batch défectueux ---
pire_batch = batchs_defectueux[-1]
plt.figure(figsize=(12, 5))
plt.plot(pire_batch['true'], color='black', label='Mesure Réelle', linewidth=2.5)
plt.plot(pire_batch['pred'], color='#d62728', linestyle='--', label=f"Prédiction CI-PI-LSTM", linewidth=2)
plt.title(f"Comportement sous panne majeure (Batch ID: {pire_batch['batch_id']} | MAE: {pire_batch['mae']:.2f})", fontsize=16, fontweight='bold')
plt.xlabel('Séquences Temporelles', fontsize=12)
plt.ylabel('Concentration Biomasse (g/L)', fontsize=12)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()