import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
import random
import time

print("--- ENTRAÎNEMENT DÉFINITIF : REAKT V6 LSTM (DONNÉES CORRIGÉES ET TRIÉES) ---")

# =====================================================================
# 1. DÉFINITION DES ARCHITECTURES
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
# 2. PRÉPARATION STRICTE DES DONNÉES (DATA ENGINEERING)
# =====================================================================
print("Chargement, tri et séquençage des données...")

df = pd.read_csv('IndPenSim_100_Batches.csv')

# Correction des noms de colonnes et création des vrais identifiants
df = df.rename(columns={'Batch ID': 'Volume (L)'})
df['VRAI_Batch_ID'] = (df['Time (h)'] < df['Time (h)'].shift(1)).cumsum() + 1

colonnes_raman = df.columns[39:]
colonnes_physiques = ['pH(pH:pH)', 'Temperature(T:K)', 'Agitator RPM(RPM:RPM)', 'Sugar feed rate(Fs:L/h)']
colonnes_cibles = ['Offline Biomass concentratio(X_offline:X(g L^{-1}))', 'Viscosity(Viscosity_offline:centPoise)']

# Interpolation étanche par lot réel
df[colonnes_cibles] = df.groupby('VRAI_Batch_ID')[colonnes_cibles].transform(
    lambda cuve: cuve.interpolate(method='linear', limit_direction='both')
)

# Nettoyage et suppression des lignes restées NaN (ex: phases SIP/CIP initiales)
df_propre = df.dropna(subset=colonnes_cibles, how='any')
df_propre = df_propre.dropna(subset=colonnes_raman, how='all').fillna(0)

# Séparation Train/Test basée sur les vrais lots uniques
batches_uniques = df_propre['VRAI_Batch_ID'].unique()
random.seed(42)
random.shuffle(batches_uniques)

split_idx = int(len(batches_uniques) * 0.8)
train_batches = batches_uniques[:split_idx]
test_batches = batches_uniques[split_idx:]

df_train = df_propre[df_propre['VRAI_Batch_ID'].isin(train_batches)]
df_test = df_propre[df_propre['VRAI_Batch_ID'].isin(test_batches)]

# Recalcul et sauvegarde des scalers sur les ensembles sains
scaler_raman = StandardScaler().fit(df_train[colonnes_raman].values)
scaler_phys = StandardScaler().fit(df_train[colonnes_physiques].values)
scaler_y = StandardScaler().fit(df_train[colonnes_cibles].values)

joblib.dump(scaler_raman, 'scaler_raman.pkl')
joblib.dump(scaler_phys, 'scaler_phys.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("✅ Nouveaux scalers synchronisés générés et sauvegardés.")

def generer_fenetres_temporelles(dataframe, sequence_length=10):
    X_raman_list, X_phys_list, Y_list = [], [], []
    batches = dataframe['VRAI_Batch_ID'].unique()
    
    for b_id in batches:
        df_batch = dataframe[dataframe['VRAI_Batch_ID'] == b_id]
        if len(df_batch) < sequence_length:
            continue
            
        r_scaled = scaler_raman.transform(df_batch[colonnes_raman].values)
        p_scaled = scaler_phys.transform(df_batch[colonnes_physiques].values)
        y_scaled = scaler_y.transform(df_batch[colonnes_cibles].values)
        
        for i in range(len(df_batch) - sequence_length + 1):
            X_raman_list.append(r_scaled[i : i + sequence_length])
            X_phys_list.append(p_scaled[i : i + sequence_length])
            Y_list.append(y_scaled[i + sequence_length - 1])
            
    T_raman = torch.FloatTensor(np.array(X_raman_list)).unsqueeze(2)
    T_phys = torch.FloatTensor(np.array(X_phys_list))
    T_y = torch.FloatTensor(np.array(Y_list))
    return T_raman, T_phys, T_y

SEQ_LEN = 10
T_raman_train, T_phys_train, T_y_train = generer_fenetres_temporelles(df_train, sequence_length=SEQ_LEN)
T_raman_test, T_phys_test, T_y_test = generer_fenetres_temporelles(df_test, sequence_length=SEQ_LEN)

train_loader = DataLoader(TensorDataset(T_raman_train, T_phys_train, T_y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(T_raman_test, T_phys_test, T_y_test), batch_size=64, shuffle=False)

# =====================================================================
# 3. L'ENTRAÎNEMENT (TRAINING LOOP)
# =====================================================================
print(f"\nLancement de l'entraînement : {len(T_raman_train)} fenêtres d'apprentissage...")

extracteur_spatial = CNNFusionV4(phys_dim=4)
model_v6 = ReaktTemporelV6(extracteur_v4=extracteur_spatial, lstm_hidden_dim=64)

criterion = nn.MSELoss()
optimizer = optim.Adam(model_v6.parameters(), lr=0.0005) 

epochs = 100
meilleure_loss_test = float('inf')

temps_debut = time.time()

for epoch in range(epochs):
    model_v6.train() 
    train_loss = 0.0
    
    for b_raman, b_phys, b_cible in train_loader:
        optimizer.zero_grad()
        predictions = model_v6(b_raman, b_phys)
        loss = criterion(predictions, b_cible)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_v6.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
        
    train_loss /= len(train_loader)
    
    model_v6.eval() 
    test_loss = 0.0
    
    with torch.no_grad():
        for b_raman, b_phys, b_cible in test_loader:
            preds = model_v6(b_raman, b_phys)
            test_loss += criterion(preds, b_cible).item()
            
    test_loss /= len(test_loader)
    
    if test_loss < meilleure_loss_test:
        meilleure_loss_test = test_loss
        torch.save(model_v6.state_dict(), 'reakt_lstm_v6_best.pth')
        nouveau_record = "🏆 RECORD SAUVEGARDÉ !"
    else:
        nouveau_record = ""
        
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Époque {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} {nouveau_record}")

temps_total = (time.time() - temps_debut) / 60
print(f"\n✅ Entraînement terminé en {temps_total:.1f} minutes. Modèle réaligné sur les vraies cuves !")