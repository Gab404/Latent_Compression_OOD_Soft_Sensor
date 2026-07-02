import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("--- ÉVALUATION : FAULT-TOLERANT DUAL LSTM (PÉNICILLINE) ---")
print("⏳ Chargement du dataset de test...")
df = pd.read_csv('IndPenSim_100_Batches.csv')
colonnes_raman = df.columns[39:] 
df = df.rename(columns={'Batch ID': 'Volume (L)'})

df = df.copy()
df['VRAI_Batch_ID'] = (df['Time (h)'] < df['Time (h)'].shift(1)).cumsum() + 1
df['VRAI_Batch_ID'] = df['VRAI_Batch_ID'].astype(int) 
df = df.sort_values(by=['VRAI_Batch_ID', 'Time (h)']).reset_index(drop=True)

colonnes_physiques = ['pH(pH:pH)', 'Temperature(T:K)', 'Agitator RPM(RPM:RPM)', 'Sugar feed rate(Fs:L/h)']

# Recherche dynamique de la colonne Pénicilline
pen_cols = [c for c in df.columns if 'Penicillin' in c or 'penicillin' in c]
col_cible = pen_cols[0]
for c in pen_cols:
    if df[c].notna().sum() > 0:
        col_cible = c
        if 'Offline' in c: break
colonnes_cibles = [col_cible]

# Interpolation et nettoyage
df[col_cible] = df.groupby('VRAI_Batch_ID')[col_cible].transform(
    lambda cuve: cuve.interpolate(method='linear', limit_direction='both')
)
df_propre = df.dropna(subset=colonnes_cibles, how='any')
df_propre = df_propre.dropna(subset=colonnes_raman, how='all').fillna(0)

batches_uniques = sorted(df_propre['VRAI_Batch_ID'].unique())
split_idx = int(len(batches_uniques) * 0.8)
train_batches = batches_uniques[:split_idx]
test_batches = batches_uniques[split_idx:]

df_test = df_propre[df_propre['VRAI_Batch_ID'].isin(test_batches)].copy()

print("⚙️ Chargement des Scalers...")
scaler_raman = joblib.load('scaler_raman_ft.pkl')
scaler_phys = joblib.load('scaler_phys_ft.pkl')
scaler_y = joblib.load('scaler_y_ft.pkl')

# Reconstitution de l'historique (LSTM II) depuis les vraies données
dictionnaire_historique_y = {}
for b in batches_uniques:
    df_b = df_propre[df_propre['VRAI_Batch_ID'] == b]
    valeur_finale_brute = df_b[col_cible].iloc[-1]
    dictionnaire_historique_y[int(b)] = scaler_y.transform([[valeur_finale_brute]])[0]

class FaultTolerant_Dual_LSTM(nn.Module):
    def __init__(self, raman_raw_dim=2200, phys_dim=4, raman_latent_dim=32, hist_dim=1, lstm_hidden=64):
        super(FaultTolerant_Dual_LSTM, self).__init__()
        self.raman_latent_dim = raman_latent_dim
        self.raman_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=2, padding=5), nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(32 * 16, raman_latent_dim), nn.ReLU()
        )
        self.lstm_process = nn.LSTM(raman_latent_dim + phys_dim, lstm_hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.lstm_history = nn.LSTM(hist_dim, 32, num_layers=2, batch_first=True, dropout=0.2)
        self.fusion_head = nn.Sequential(
            nn.Linear(lstm_hidden + 32, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, seq_raman, seq_phys, seq_hist):
        batch_size, seq_len, _ = seq_raman.size()
        ram_flat = seq_raman.contiguous().view(batch_size * seq_len, 1, -1)
        ram_latent = self.raman_encoder(ram_flat).view(batch_size, seq_len, self.raman_latent_dim)
        lstm_out_proc, _ = self.lstm_process(torch.cat((ram_latent, seq_phys), dim=2))
        lstm_out_hist, _ = self.lstm_history(seq_hist)
        return self.fusion_head(torch.cat((lstm_out_proc[:, -1, :], lstm_out_hist[:, -1, :]), dim=1))

print("🧠 Chargement du modèle entraîné...")
model = FaultTolerant_Dual_LSTM(phys_dim=len(colonnes_physiques))
model.load_state_dict(torch.load('reakt_fault_tolerant_penicillin.pth', weights_only=True))
model.eval()

SEQ_LEN = 10
W_HIST = 5

all_true_rates = []
all_pred_rates = []

all_true_concentrations = []
all_pred_concentrations = []

print("📈 Évaluation en cours (Reconstitution des Courbes de Pénicilline)...")
with torch.no_grad():
    for b in test_batches:
        df_b = df_test[df_test['VRAI_Batch_ID'] == b]
        if len(df_b) <= SEQ_LEN: continue
        
        # Scaling local
        ram_scaled = scaler_raman.transform(df_b[colonnes_raman].values)
        phy_scaled = scaler_phys.transform(df_b[colonnes_physiques].values)
        y_scaled = scaler_y.transform(df_b[colonnes_cibles].values)
        
        # Historique
        hist_y = [dictionnaire_historique_y.get(past_b, [0.0]) for past_b in range(b - W_HIST, b)]
        
        X_ram, X_phy, X_hist, Y_rate_true_scaled = [], [], [], []
        
        # Calcul du Vrai Taux Scaled (Différentiel)
        y_scaled_series = pd.Series(y_scaled.flatten())
        true_rates = y_scaled_series.diff().fillna(0).values
        
        for i in range(len(df_b) - SEQ_LEN):
            X_ram.append(ram_scaled[i : i + SEQ_LEN])
            X_phy.append(phy_scaled[i : i + SEQ_LEN])
            X_hist.append(hist_y)
            Y_rate_true_scaled.append(true_rates[i + SEQ_LEN])
            
        T_ram = torch.FloatTensor(np.array(X_ram))
        T_phy = torch.FloatTensor(np.array(X_phy))
        T_hist = torch.FloatTensor(np.array(X_hist))
        
        # Prédiction des vitesses (Le différentiel)
        pred_rates_scaled = model(T_ram, T_phy, T_hist).numpy().flatten()
        
        all_true_rates.extend(Y_rate_true_scaled)
        all_pred_rates.extend(pred_rates_scaled)
        
        # ============================================================
        # LA MAGIE DE LA RECONSTITUTION (L'Intégrale de la section 3.1)
        # ============================================================
        valeurs_reconstruites_scaled = []
        # On part de la dernière valeur connue avant la première prédiction
        valeur_actuelle = y_scaled[SEQ_LEN - 1][0] 
        
        for rate in pred_rates_scaled:
            valeur_actuelle = valeur_actuelle + rate
            valeurs_reconstruites_scaled.append(valeur_actuelle)
            
        # Inverse Transform pour revenir aux vrais g/L
        pred_real = scaler_y.inverse_transform(np.array(valeurs_reconstruites_scaled).reshape(-1, 1)).flatten()
        true_real = scaler_y.inverse_transform(y_scaled[SEQ_LEN:]).flatten()
        
        all_pred_concentrations.extend(pred_real)
        all_true_concentrations.extend(true_real)

# 1. Métriques sur le Taux (L'Entraînement)
rmse_rate = np.sqrt(mean_squared_error(all_true_rates, all_pred_rates))
mae_rate = mean_absolute_error(all_true_rates, all_pred_rates)

# 2. Métriques sur la Concentration (La Réalité Industrielle)
rmse_conc = np.sqrt(mean_squared_error(all_true_concentrations, all_pred_concentrations))
mae_conc = mean_absolute_error(all_true_concentrations, all_pred_concentrations)
r2_conc = r2_score(all_true_concentrations, all_pred_concentrations)

print("\n" + "="*50)
print("🎯 RÉSULTATS 1 : APPRENTISSAGE (SUR LA VITESSE / RATE)")
print(f"-> Ces erreurs sont celles que le réseau voyait pendant l'entraînement.")
print(f"RMSE (Vitesse) : {rmse_rate:.6f} (Écart typique sur le différentiel)")
print(f"MAE  (Vitesse) : {mae_rate:.6f}")
print("="*50)

print("\n" + "="*50)
print("🚀 RÉSULTATS 2 : PRODUCTION (CONCENTRATION PÉNICILLINE g/L)")
print(f"-> Erreurs réelles après avoir fait l'intégrale (Reconstitution)")
print(f"RMSE : {rmse_conc:.3f} g/L")
print(f"MAE  : {mae_conc:.3f} g/L")
print(f"R²   : {r2_conc:.4f}")
print("="*50)

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Graphique 1 : Parity Plot (Concentration finale)
ax1.scatter(all_true_concentrations, all_pred_concentrations, color='#e67e22', alpha=0.5, edgecolor='w')
min_val = min(min(all_true_concentrations), min(all_pred_concentrations))
max_val = max(max(all_true_concentrations), max(all_pred_concentrations))
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
ax1.set_title("Parity Plot : Concentration de Pénicilline (Test Set)", fontsize=14, fontweight='bold')
ax1.set_xlabel("Vraie Concentration (g/L)", fontsize=12)
ax1.set_ylabel("Concentration Prédite (g/L)", fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.3)

# Boîte de texte
textstr = f'RMSE = {rmse_conc:.2f} g/L\nMAE = {mae_conc:.2f} g/L\n$R^2$ = {r2_conc:.3f}'
props = dict(boxstyle='round', facecolor='#2c3e50', alpha=0.8, edgecolor='#7f8c8d')
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=props)

# Graphique 2 : Zoom sur une cuve spécifique (Endpoint Detection)
# On prend arbitrairement le 3ème batch du Test Set
batch_exemple = test_batches[2]
idx_start = 0
for b in test_batches:
    df_b = df_test[df_test['VRAI_Batch_ID'] == b]
    length = len(df_b) - SEQ_LEN
    if b == batch_exemple:
        break
    idx_start += length

true_batch = all_true_concentrations[idx_start:idx_start+length]
pred_batch = all_pred_concentrations[idx_start:idx_start+length]

ax2.plot(true_batch, label=f'Mesure Labo (Batch #{batch_exemple})', color='#3498db', lw=3)
ax2.plot(pred_batch, label='Prédiction Dual-LSTM', color='#e74c3c', linestyle='--', lw=2)

# Ligne cible pour l'Endpoint Detection (ex: 24 g/L comme dans le papier)
ax2.axhline(y=24.0, color='#2ecc71', linestyle=':', lw=2, label='Cible Endpoint (24 g/L)')

ax2.set_title(f"Cinétique de Reconstitution (Batch {batch_exemple})", fontsize=14, fontweight='bold')
ax2.set_xlabel("Heures d'opération", fontsize=12)
ax2.set_ylabel("Pénicilline (g/L)", fontsize=12)
ax2.legend(loc='upper left', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.3)

plt.tight_layout()
plt.show()
print("\n✅ Évaluation terminée ! Graphiques générés.")