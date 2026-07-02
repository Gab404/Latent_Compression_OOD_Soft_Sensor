import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import random

print("--- ENTRAÎNEMENT RECHERCHE : CI-PI-LSTM SUR RAMAN (INDPENSIM) ---")

# =====================================================================
# 1. DATA ENGINEERING & SÉQUENÇAGE ÉTANCHE
# =====================================================================
print("⏳ Chargement et nettoyage des données...")
df = pd.read_csv('IndPenSim_100_Batches.csv')

# Correction de la colonne et création des vrais numéros de lots
df = df.rename(columns={'Batch ID': 'Volume (L)'})
df['VRAI_Batch_ID'] = (df['Time (h)'] < df['Time (h)'].shift(1)).cumsum() + 1
df = df.sort_values(by=['VRAI_Batch_ID', 'Time (h)']).reset_index(drop=True)

colonnes_raman = df.columns[39:] # Les ~2200 colonnes Raman
colonnes_physiques = ['pH(pH:pH)', 'Temperature(T:K)', 'Agitator RPM(RPM:RPM)', 'Sugar feed rate(Fs:L/h)']
colonnes_cibles = ['Offline Biomass concentratio(X_offline:X(g L^{-1}))', 'Viscosity(Viscosity_offline:centPoise)']

# Interpolation étanche (groupée par VRAI_Batch_ID)
df[colonnes_cibles] = df.groupby('VRAI_Batch_ID')[colonnes_cibles].transform(
    lambda cuve: cuve.interpolate(method='linear', limit_direction='both')
)

# La Grande Purge
df_propre = df.dropna(subset=colonnes_cibles, how='any')
df_propre = df_propre.dropna(subset=colonnes_raman, how='all').fillna(0)

# =====================================================================
# 2. SÉPARATION TRAIN/TEST (ZERO DATA LEAKAGE)
# =====================================================================
batches_uniques = df_propre['VRAI_Batch_ID'].unique()
random.seed(42) 
random.shuffle(batches_uniques)

split_idx = int(len(batches_uniques) * 0.8)
train_batches = batches_uniques[:split_idx]
test_batches = batches_uniques[split_idx:]

df_train = df_propre[df_propre['VRAI_Batch_ID'].isin(train_batches)]
df_test = df_propre[df_propre['VRAI_Batch_ID'].isin(test_batches)]

# =====================================================================
# 3. SCALERS ET CRÉATION DE SÉQUENCES TEMPORELLES
# =====================================================================
print("⚙️ Préparation des séquences temporelles pour le LSTM...")
scaler_raman = StandardScaler().fit(df_train[colonnes_raman].values)
scaler_phys = StandardScaler().fit(df_train[colonnes_physiques].values)
scaler_y = StandardScaler().fit(df_train[colonnes_cibles].values)

joblib.dump(scaler_raman, 'scaler_raman.pkl')
joblib.dump(scaler_phys, 'scaler_phys.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

SEQ_LEN = 10 # Fenêtre temporelle pour le LSTM

def preparer_sequences(dataframe):
    X_ram, X_phy, Y = [], [], []
    
    # Isolation par cuve
    for b in dataframe['VRAI_Batch_ID'].unique():
        df_b = dataframe[dataframe['VRAI_Batch_ID'] == b]
        if len(df_b) <= SEQ_LEN:
            continue
            
        ram = scaler_raman.transform(df_b[colonnes_raman].values)
        phy = scaler_phys.transform(df_b[colonnes_physiques].values)
        y = scaler_y.transform(df_b[colonnes_cibles].values)
        
        # Création des séquences glissantes
        for i in range(len(df_b) - SEQ_LEN):
            X_ram.append(ram[i : i + SEQ_LEN])
            X_phy.append(phy[i : i + SEQ_LEN])
            Y.append(y[i : i + SEQ_LEN])
            
    return torch.FloatTensor(np.array(X_ram)), torch.FloatTensor(np.array(X_phy)), torch.FloatTensor(np.array(Y))

# Création des Loaders avec des séquences 3D
T_raman_train, T_phys_train, T_y_train = preparer_sequences(df_train)
train_loader = DataLoader(TensorDataset(T_raman_train, T_phys_train, T_y_train), batch_size=64, shuffle=True)

T_raman_test, T_phys_test, T_y_test = preparer_sequences(df_test)
test_loader = DataLoader(TensorDataset(T_raman_test, T_phys_test, T_y_test), batch_size=64, shuffle=False)

# =====================================================================
# 4. L'ARCHITECTURE DE RECHERCHE : CI-PI-LSTM RAMAN
# =====================================================================

class SpectralReducer(nn.Module):
    """ Réduit les 2200 longueurs d'onde Raman en 16 features latentes """
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
    """ Isole l'évolution temporelle de chaque feature Raman (L'innovation CI) """
    def __init__(self, num_channels, features_per_channel=4):
        super().__init__()
        self.ci_conv = nn.Conv1d(
            num_channels, 
            num_channels * features_per_channel, 
            kernel_size=3, padding=1, 
            groups=num_channels # <- Le coeur de la méthode Channel-Independent
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x : [Batch, Seq, Channels] -> Conv1D attend [Batch, Channels, Seq]
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
        
        # Fusion = (16 canaux * 4) + 4 variables physiques = 68 features
        fusion_dim = (latent_raman * features_per_channel) + phys_dim
        
        self.lstm = nn.LSTM(fusion_dim, 128, num_layers=2, batch_first=True, dropout=0.2)
        
        self.fc_bio = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 2) # Sorties : [Biomasse, Viscosité]
        )
        
    def forward(self, seq_ram, seq_phy):
        B, S, F = seq_ram.size()
        
        # 1. Réduction spectrale (application sur chaque pas de temps)
        ram_flat = seq_ram.view(B * S, 1, F)
        ram_latent = self.spectral_reducer(ram_flat)
        ram_latent = ram_latent.view(B, S, -1)
        
        # 2. Isolation Temporelle (CI)
        ram_ci = self.ci_extractor(ram_latent)
        
        # 3. Fusion et LSTM
        fused = torch.cat((ram_ci, seq_phy), dim=2)
        lstm_out, _ = self.lstm(fused)
        
        # 4. Prédiction sur toute la séquence
        return self.fc_bio(lstm_out)

# =====================================================================
# 5. BOUCLE D'ENTRAÎNEMENT (AVEC GREY-BOX PHYSICS LOSS)
# =====================================================================
model = CI_PI_LSTM(phys_dim=len(colonnes_physiques))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

epochs = 150
physics_lambda = 0.5 # Poids de la contrainte biologique
meilleure_loss_test = float('inf')

print(f"\n🚀 Lancement de l'entraînement CI-PI-LSTM sur {len(train_batches)} cuves...")

for epoch in range(epochs):
    model.train()
    loss_total = 0.0
    
    for b_raman, b_phys, b_y in train_loader:
        optimizer.zero_grad()
        pred = model(b_raman, b_phys)
        
        # 1. Loss MSE classique
        loss_mse = criterion(pred, b_y)
        
        # 2. Physics Loss sur la Biomasse (Index 0)
        # Règle d'or : La biomasse ne doit pas chuter (croissance monotone)
        # pred[:, :, 0] correspond à la biomasse sur toute la séquence
        biomasse_pred = pred[:, :, 0]
        diff_temporelle = biomasse_pred[:, :-1] - biomasse_pred[:, 1:] # t - (t+1)
        loss_physics = torch.mean(torch.relu(diff_temporelle))
        
        loss = loss_mse + physics_lambda * loss_physics
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
        
    scheduler.step()
    train_loss = loss_total / len(train_loader)
    
    # --- Phase de Validation ---
    model.eval()
    test_loss_total = 0.0
    with torch.no_grad():
        for b_r, b_p, b_y in test_loader:
            pred_test = model(b_r, b_p)
            test_loss_total += criterion(pred_test, b_y).item()
            
    test_loss = test_loss_total / len(test_loader)
    
    if test_loss < meilleure_loss_test:
        meilleure_loss_test = test_loss
        torch.save(model.state_dict(), 'reakt_ci_pi_lstm_best.pth')
        record = "⭐ Nouveau Record Sauvegardé !"
    else:
        record = ""
        
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} {record}")

print("\n✅ Modèle CI-PI-LSTM sauvegardé sous 'reakt_ci_pi_lstm_best.pth'.")