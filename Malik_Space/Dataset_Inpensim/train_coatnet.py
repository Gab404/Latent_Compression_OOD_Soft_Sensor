import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib  # <-- LA CLÉ POUR SAUVEGARDER L'ÉCHELLE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import random

print("--- ENTRAÎNEMENT DÉFINITIF : COATNET V5.1 (DONNÉES PURGÉES) ---")
# =====================================================================
# --- LE NOUVEAU CORRECTIF DE DATA ENGINEERING (VRAI_Batch_ID) ---
# =====================================================================
df = pd.read_csv('IndPenSim_100_Batches.csv')

# Correction de la colonne et création des vrais numéros de lots
df = df.rename(columns={'Batch ID': 'Volume (L)'})
df['VRAI_Batch_ID'] = (df['Time (h)'] < df['Time (h)'].shift(1)).cumsum() + 1
df = df.sort_values(by=['VRAI_Batch_ID', 'Time (h)']).reset_index(drop=True)

colonnes_raman = df.columns[39:] # Vos 2200 longueurs d'onde
colonnes_physiques = ['pH(pH:pH)', 'Temperature(T:K)', 'Agitator RPM(RPM:RPM)', 'Sugar feed rate(Fs:L/h)']
colonnes_cibles = ['Offline Biomass concentratio(X_offline:X(g L^{-1}))', 'Viscosity(Viscosity_offline:centPoise)']

# 1. Interpolation étanche (groupée par VRAI_Batch_ID)
df[colonnes_cibles] = df.groupby('VRAI_Batch_ID')[colonnes_cibles].transform(
    lambda cuve: cuve.interpolate(method='linear', limit_direction='both')
)

# 2. Le bouclier anti-fantômes
df_propre = df.dropna(subset=colonnes_cibles, how='any')

# 3. Nettoyage classique du Raman
df_propre = df_propre.dropna(subset=colonnes_raman, how='all').fillna(0)

# =====================================================================
# --- SÉPARATION TRAIN/TEST SUR LES VRAIS LOTS ---
# =====================================================================
batches_uniques = df_propre['VRAI_Batch_ID'].unique()
random.seed(42) # Strictement la même graine pour comparer avec CNN et LSTM
random.shuffle(batches_uniques)

split_idx = int(len(batches_uniques) * 0.8)
train_batches = batches_uniques[:split_idx]
test_batches = batches_uniques[split_idx:]

# Filtre basé sur le vrai lot
df_train = df_propre[df_propre['VRAI_Batch_ID'].isin(train_batches)]
df_test = df_propre[df_propre['VRAI_Batch_ID'].isin(test_batches)]

# 1. Ajustement ET Sauvegarde Physique des Scalers
scaler_raman = StandardScaler().fit(df_train[colonnes_raman].values)
scaler_phys = StandardScaler().fit(df_train[colonnes_physiques].values)
scaler_y = StandardScaler().fit(df_train[colonnes_cibles].values)

joblib.dump(scaler_raman, 'scaler_raman.pkl')
joblib.dump(scaler_phys, 'scaler_phys.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("✅ Scalers mis à jour et sauvegardés en fichiers .pkl")

def preparer_tenseurs(dataframe):
    return (torch.FloatTensor(scaler_raman.transform(dataframe[colonnes_raman].values)).unsqueeze(1), 
            torch.FloatTensor(scaler_phys.transform(dataframe[colonnes_physiques].values)), 
            torch.FloatTensor(scaler_y.transform(dataframe[colonnes_cibles].values)))

T_raman_train, T_phys_train, T_y_train = preparer_tenseurs(df_train)
train_loader = DataLoader(TensorDataset(T_raman_train, T_phys_train, T_y_train), batch_size=128, shuffle=True)

# 2. L'Architecture avec Encodage Positionnel (Le GPS des longueurs d'onde)
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

model = CoAtNetFusion(phys_dim=len(colonnes_physiques))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
# --- AJOUT : Préparation des tenseurs de TEST ---
T_raman_test, T_phys_test, T_y_test = preparer_tenseurs(df_test)
test_loader = DataLoader(TensorDataset(T_raman_test, T_phys_test, T_y_test), batch_size=128, shuffle=False)

epochs = 50
meilleure_loss_test = float('inf') # Pour sauvegarder le meilleur modèle

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for b_r, b_p, b_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(b_r, b_p), b_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # --- AJOUT : Évaluation sur le Test Set ---
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for b_r, b_p, b_y in test_loader:
            loss_t = criterion(model(b_r, b_p), b_y)
            test_loss += loss_t.item()
    test_loss /= len(test_loader)
    
    # Sauvegarde uniquement si le modèle s'améliore sur les données qu'il ne connaît pas
    if test_loss < meilleure_loss_test:
        meilleure_loss_test = test_loss
        torch.save(model.state_dict(), 'reakt_coatnet_v5_best.pth')
        record = "⭐ Nouveau Record"
    else:
        record = ""

    # Affichage à chaque époque pour bien suivre
    print(f"Époque {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} {record}")

print("✅ Modèle V5.1 (Purgé) entraîné et le meilleur a été sauvegardé.")