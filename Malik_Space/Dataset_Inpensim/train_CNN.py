import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import random

print("--- ENTRAÎNEMENT DÉFINITIF : CNN 1D V4 (DONNÉES PURGÉES & VALIDATION) ---")

# =====================================================================
# 1. DATA ENGINEERING & SÉQUENÇAGE ÉTANCHE
# =====================================================================
print(" Chargement et nettoyage des données...")
df = pd.read_csv('IndPenSim_100_Batches.csv')

# Correction de la colonne et création des vrais numéros de lots
df = df.rename(columns={'Batch ID': 'Volume (L)'})
df['VRAI_Batch_ID'] = (df['Time (h)'] < df['Time (h)'].shift(1)).cumsum() + 1
df = df.sort_values(by=['VRAI_Batch_ID', 'Time (h)']).reset_index(drop=True)

colonnes_raman = df.columns[39:] # Pointe désormais vers les 2200 colonnes
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
random.seed(42) # Strictement la même graine
random.shuffle(batches_uniques)

split_idx = int(len(batches_uniques) * 0.8)
train_batches = batches_uniques[:split_idx]
test_batches = batches_uniques[split_idx:]

# On filtre en utilisant VRAI_Batch_ID
df_train = df_propre[df_propre['VRAI_Batch_ID'].isin(train_batches)]
df_test = df_propre[df_propre['VRAI_Batch_ID'].isin(test_batches)]

# =====================================================================
# 3. SCALERS ET TENSEURS
# =====================================================================
print("⚙️ Préparation des matrices et des scalers...")
# Ajustement ET Sauvegarde sur le TRAIN uniquement
scaler_raman = StandardScaler().fit(df_train[colonnes_raman].values)
scaler_phys = StandardScaler().fit(df_train[colonnes_physiques].values)
scaler_y = StandardScaler().fit(df_train[colonnes_cibles].values)

joblib.dump(scaler_raman, 'scaler_raman.pkl')
joblib.dump(scaler_phys, 'scaler_phys.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print(" Scalers mis à jour et sauvegardés en fichiers .pkl")

def preparer_tenseurs(dataframe):
    T_ram = torch.FloatTensor(scaler_raman.transform(dataframe[colonnes_raman].values)).unsqueeze(1)
    T_phy = torch.FloatTensor(scaler_phys.transform(dataframe[colonnes_physiques].values))
    T_y = torch.FloatTensor(scaler_y.transform(dataframe[colonnes_cibles].values))
    return T_ram, T_phy, T_y

# Création des Loaders
T_raman_train, T_phys_train, T_y_train = preparer_tenseurs(df_train)
train_loader = DataLoader(TensorDataset(T_raman_train, T_phys_train, T_y_train), batch_size=128, shuffle=True)

T_raman_test, T_phys_test, T_y_test = preparer_tenseurs(df_test)
test_loader = DataLoader(TensorDataset(T_raman_test, T_phys_test, T_y_test), batch_size=128, shuffle=False)

# =====================================================================
# 4. ARCHITECTURE CNN 1D V4
# =====================================================================
class FusionModel(nn.Module):
    def __init__(self, phys_dim):
        super(FusionModel, self).__init__()
        self.raman_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=2, padding=5), nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten() # Sécurise l'aplatissement du tenseur (32 * 16 = 512)
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
# 5. BOUCLE D'ENTRAÎNEMENT & VALIDATION
# =====================================================================
model = FusionModel(phys_dim=len(colonnes_physiques))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

epochs = 150
meilleure_loss_test = float('inf')

print(f"\n🚀 Lancement de l'entraînement sur {len(df_train)} échantillons ({len(train_batches)} cuves)...")

for epoch in range(epochs):
    # --- Phase d'Entraînement ---
    model.train()
    loss_total = 0.0
    for b_raman, b_phys, b_y in train_loader:
        optimizer.zero_grad()
        pred = model(b_raman, b_phys)
        loss = criterion(pred, b_y)
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
    
    # --- Sauvegarde du meilleur modèle ---
    if test_loss < meilleure_loss_test:
        meilleure_loss_test = test_loss
        torch.save(model.state_dict(), 'reakt_fusion_v4_best.pth')
        record = "⭐ Nouveau Record Sauvegardé !"
    else:
        record = ""
    
    # Affichage régulier
    if (epoch + 1) % 5 == 0 or epoch == 0:
        lr_actuel = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | LR: {lr_actuel:.6f} {record}")

print("\n✅ Entraînement terminé ! Le meilleur modèle a été enregistré sous 'reakt_fusion_v4_best.pth'.")