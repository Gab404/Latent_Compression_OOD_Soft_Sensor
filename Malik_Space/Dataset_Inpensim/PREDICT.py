import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import random

# --- 1. DÉFINITION DES ARCHITECTURES ---
class CNNFusionV4(nn.Module):
    def __init__(self, phys_dim=4):
        super(CNNFusionV4, self).__init__()
        self.raman_branch = nn.Sequential(nn.Conv1d(1, 16, kernel_size=11, stride=2, padding=5), nn.ReLU(), nn.MaxPool1d(3), nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(16), nn.Flatten())
        self.phys_branch = nn.Sequential(nn.Linear(phys_dim, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
    def forward(self, x_raman, x_phys):
        return torch.cat((self.raman_branch(x_raman), self.phys_branch(x_phys)), dim=1)

class PositionalEncoding1D(nn.Module):
    def __init__(self, embed_dim, max_len=500):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, embed_dim))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

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
        self.conv_stem = nn.Sequential(nn.Conv1d(1, 16, kernel_size=11, stride=2, padding=5), nn.ReLU(), nn.MaxPool1d(3), nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU())
        self.pos_encoder = PositionalEncoding1D(embed_dim=32)
        self.attention_block = SelfAttention1D(embed_dim=32, num_heads=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.phys_branch = nn.Sequential(nn.Linear(phys_dim, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        self.fusion_head = nn.Sequential(nn.Linear(48, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x_raman, x_phys):
        feat_conv = self.conv_stem(x_raman).transpose(1, 2)
        feat_conv = self.pos_encoder(feat_conv) 
        attended = self.attention_block(feat_conv).transpose(1, 2)
        combined = torch.cat((self.pool(attended).squeeze(-1), self.phys_branch(x_phys)), dim=1)
        return self.fusion_head(combined)

class FusionModel(nn.Module):
    def __init__(self, phys_dim=4):
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
    
class ReaktTemporelV6(nn.Module):
    def __init__(self, extracteur_v4, lstm_hidden_dim=64):
        super(ReaktTemporelV6, self).__init__()
        self.extracteur_spatial = extracteur_v4
        self.lstm = nn.LSTM(input_size=528, hidden_size=lstm_hidden_dim, num_layers=2, batch_first=True)
        self.tete_finale = nn.Sequential(nn.Linear(lstm_hidden_dim, 16), nn.ReLU(), nn.Linear(16, 2))
    def forward(self, x_raman_seq, x_phys_seq):
        batch_size, seq_len, c, w = x_raman_seq.size()
        features = self.extracteur_spatial(x_raman_seq.view(batch_size * seq_len, c, w), x_phys_seq.view(batch_size * seq_len, -1))
        lstm_out, _ = self.lstm(features.view(batch_size, seq_len, -1))
        return self.tete_finale(lstm_out[:, -1, :])

# --- 2. CHARGEMENT DES MODÈLES ---
scaler_raman = joblib.load('scaler_raman.pkl')
scaler_phys = joblib.load('scaler_phys.pkl')
scaler_y = joblib.load('scaler_y.pkl')

mod_v4 = FusionModel(phys_dim=4); mod_v4.load_state_dict(torch.load('reakt_fusion_v4.pth')); mod_v4.eval()
mod_v5 = CoAtNetFusion(phys_dim=4); mod_v5.load_state_dict(torch.load('reakt_coatnet_v5.pth')); mod_v5.eval()
mod_v6 = ReaktTemporelV6(CNNFusionV4()); mod_v6.load_state_dict(torch.load('reakt_lstm_v6_best.pth')); mod_v6.eval()

# --- 3. SÉLECTION ET PRÉPARATION D'UN LOT ---
df = pd.read_csv('IndPenSim_100_Batches.csv')
colonnes_raman = df.columns[39:]
colonnes_physiques = ['pH(pH:pH)', 'Temperature(T:K)', 'Agitator RPM(RPM:RPM)', 'Sugar feed rate(Fs:L/h)']
colonnes_cibles = ['Offline Biomass concentratio(X_offline:X(g L^{-1}))', 'Viscosity(Viscosity_offline:centPoise)']

# On refait le nettoyage rapide pour ne garder que les cuves saines
df[colonnes_cibles] = df.groupby('Batch ID')[colonnes_cibles].transform(
    lambda cuve: cuve.interpolate(method='linear', limit_direction='both')
)
df_propre = df.dropna(subset=colonnes_cibles, how='any')
df_propre = df_propre.dropna(subset=colonnes_raman, how='all').fillna(0)

# 💡 LA CORRECTION EST ICI : On pioche automatiquement une cuve valide !
cuves_valides = df_propre['Batch ID'].unique()
batch_id = cuves_valides[5]  # Tu peux changer [0] par [1], [5], etc. pour tester d'autres cuves
print(f"🔬 Cuve sélectionnée automatiquement pour le duel : {batch_id}")

df_cuve = df_propre[df_propre['Batch ID'] == batch_id]

# Préparation des données
r_scaled = scaler_raman.transform(df_cuve[colonnes_raman].values)
p_scaled = scaler_phys.transform(df_cuve[colonnes_physiques].values)
# --- 4. PRÉDICTION ---
with torch.no_grad():
    # V4 et V5
    p_v4 = scaler_y.inverse_transform(mod_v4(torch.FloatTensor(r_scaled).unsqueeze(1), torch.FloatTensor(p_scaled)).numpy())
    p_v5 = scaler_y.inverse_transform(mod_v5(torch.FloatTensor(r_scaled).unsqueeze(1), torch.FloatTensor(p_scaled)).numpy())
    # V6 (besoin de fenêtres de 10)
    seq = 10
    r_seq = torch.FloatTensor(np.array([r_scaled[i:i+seq] for i in range(len(r_scaled)-seq+1)])).unsqueeze(2)
    p_seq = torch.FloatTensor(np.array([p_scaled[i:i+seq] for i in range(len(p_scaled)-seq+1)]))
    p_v6 = scaler_y.inverse_transform(mod_v6(r_seq, p_seq).numpy())

# --- 5. VISUALISATION ---
plt.figure(figsize=(12, 6))
# 'ko' signifie "Black" (k) et "Gros points" (o), sans les relier par une ligne
plt.plot(df_cuve['Time (h)'], df_cuve['Offline Biomass concentratio(X_offline:X(g L^{-1}))'], 'ko', markersize=6, label="Vrais Prélèvements Labo")
plt.plot(df_cuve['Time (h)'].values, p_v4[:,0], '--', label="V4 CNN")
plt.plot(df_cuve['Time (h)'].values, p_v5[:,0], '--', label="V5.1 CoAtNet")
plt.plot(df_cuve['Time (h)'].values[seq-1:], p_v6[:,0], 'r-', label="V6 LSTM (Pilote)")
plt.title(f"Duel des modèles sur le Batch {batch_id}")
plt.xlabel("Heures"); plt.ylabel("Biomasse (g/L)"); plt.legend(); plt.grid(True)
plt.show()