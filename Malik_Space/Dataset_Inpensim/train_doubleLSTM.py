import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

print("--- ENTRAÎNEMENT : FAULT-TOLERANT DUAL LSTM (PÉNICILLINE SEULE) ---")
print("⏳ Chargement des données...")
df = pd.read_csv('IndPenSim_100_Batches.csv')

# CORRECTION 3 : Sécurisation des colonnes Raman AVANT d'ajouter de nouvelles colonnes
# Cela empêche le numéro de batch de se retrouver mélangé aux données optiques
colonnes_raman = df.columns[39:] 

df = df.rename(columns={'Batch ID': 'Volume (L)'})

# CORRECTION 1 : Défragmentation mémoire précoce pour éviter le PerformanceWarning
df = df.copy()
df['VRAI_Batch_ID'] = (df['Time (h)'] < df['Time (h)'].shift(1)).cumsum() + 1
df['VRAI_Batch_ID'] = df['VRAI_Batch_ID'].astype(int) 

df = df.sort_values(by=['VRAI_Batch_ID', 'Time (h)']).reset_index(drop=True)

colonnes_physiques = ['pH(pH:pH)', 'Temperature(T:K)', 'Agitator RPM(RPM:RPM)', 'Sugar feed rate(Fs:L/h)']

# CORRECTION 2 : Recherche dynamique de la vraie colonne Pénicilline
pen_cols = [c for c in df.columns if 'Penicillin' in c or 'penicillin' in c]
col_cible = pen_cols[0] # Par défaut
for c in pen_cols:
    if df[c].notna().sum() > 0: # On cherche la colonne qui contient VRAIMENT des données
        col_cible = c
        if 'Offline' in c: # On donne la priorité à la mesure hors-ligne si elle existe
            break

print(f"🎯 Colonne cible détectée et sélectionnée : {col_cible}")
colonnes_cibles = [col_cible]

# Interpolation pour boucher les trous des mesures hors-ligne
df[col_cible] = df.groupby('VRAI_Batch_ID')[col_cible].transform(
    lambda cuve: cuve.interpolate(method='linear', limit_direction='both')
)

df_propre = df.dropna(subset=colonnes_cibles, how='any')
df_propre = df_propre.dropna(subset=colonnes_raman, how='all').fillna(0)
df_propre = df_propre.copy() 

# Séparation Train / Test
batches_uniques = sorted(df_propre['VRAI_Batch_ID'].unique())
if len(batches_uniques) == 0:
    raise ValueError("Erreur critique : Plus aucune donnée après le nettoyage. Le dataset est vide.")

split_idx = int(len(batches_uniques) * 0.8)
train_batches = batches_uniques[:split_idx]
test_batches = batches_uniques[split_idx:]

df_train = df_propre[df_propre['VRAI_Batch_ID'].isin(train_batches)].copy()
df_test = df_propre[df_propre['VRAI_Batch_ID'].isin(test_batches)].copy()

print("⚙️ Entraînement des Scalers...")
scaler_raman = StandardScaler().fit(df_train[colonnes_raman].values)
scaler_phys = StandardScaler().fit(df_train[colonnes_physiques].values)
scaler_y = StandardScaler().fit(df_train[colonnes_cibles].values)

joblib.dump(scaler_raman, 'scaler_raman_ft.pkl')
joblib.dump(scaler_phys, 'scaler_phys_ft.pkl')
joblib.dump(scaler_y, 'scaler_y_ft.pkl')

df_train_scaled = df_train.copy()
df_train_scaled[colonnes_raman] = scaler_raman.transform(df_train[colonnes_raman].values)
df_train_scaled[colonnes_physiques] = scaler_phys.transform(df_train[colonnes_physiques].values)
df_train_scaled[col_cible] = scaler_y.transform(df_train[colonnes_cibles].values)

df_test_scaled = df_test.copy()
df_test_scaled[colonnes_raman] = scaler_raman.transform(df_test[colonnes_raman].values)
df_test_scaled[colonnes_physiques] = scaler_phys.transform(df_test[colonnes_physiques].values)
df_test_scaled[col_cible] = scaler_y.transform(df_test[colonnes_cibles].values)

print("📈 Calcul du Process Rate (Vitesse de création de Pénicilline)...")
# CORRECTION 4 : On passe le nom de la colonne en String (et non en Liste) au diff()
# Cela force Pandas à renvoyer une Series, évitant le bug des NaNs lors de l'assignation !
df_train_scaled['Rate_Pen'] = df_train_scaled.groupby('VRAI_Batch_ID')[col_cible].diff().fillna(0)
df_test_scaled['Rate_Pen'] = df_test_scaled.groupby('VRAI_Batch_ID')[col_cible].diff().fillna(0)

# Dictionnaire de l'historique : Sauvegarde du résultat final de chaque cuve
dictionnaire_historique_y = {}
for b in batches_uniques:
    df_b = df_propre[df_propre['VRAI_Batch_ID'] == b]
    valeur_finale_brute = df_b[col_cible].iloc[-1]
    dictionnaire_historique_y[int(b)] = scaler_y.transform([[valeur_finale_brute]])[0]

SEQ_LEN = 10 # Longueur de la fenêtre glissante pour le Process LSTM (LSTM I)
W_HIST = 5   # Nombre d'anciennes cuves à regarder pour l'Historical LSTM (LSTM II)

def preparer_sequences_differentielles(dataframe):
    X_ram, X_phy, X_hist, Y_rate = [], [], [], []
    
    for b in dataframe['VRAI_Batch_ID'].unique():
        b = int(b) 
        df_b = dataframe[dataframe['VRAI_Batch_ID'] == b]
        if len(df_b) <= SEQ_LEN: 
            continue
            
        # 1. Préparation de l'historique (LSTM II) : Les 5 cuves précédentes
        hist_y = []
        for past_b in range(b - W_HIST, b):
            if past_b in dictionnaire_historique_y:
                hist_y.append(dictionnaire_historique_y[past_b])
            else:
                hist_y.append([0.0]) # Padding pour les toutes premières cuves
                
        ram_scaled = df_b[colonnes_raman].values
        phy_scaled = df_b[colonnes_physiques].values
        rate_scaled = df_b['Rate_Pen'].values
        
        # 2. Fenêtre glissante pour la cuve actuelle (LSTM I)
        for i in range(len(df_b) - SEQ_LEN):
            X_ram.append(ram_scaled[i : i + SEQ_LEN])
            X_phy.append(phy_scaled[i : i + SEQ_LEN])
            X_hist.append(hist_y)
            Y_rate.append([rate_scaled[i + SEQ_LEN]]) # Cible = La vitesse (Rate)
            
    return torch.FloatTensor(np.array(X_ram)), torch.FloatTensor(np.array(X_phy)), \
           torch.FloatTensor(np.array(X_hist)), torch.FloatTensor(np.array(Y_rate))

print("⚙️ Fabrication des tenseurs temporels...")
Tr_ram, Tr_phy, Tr_hist, Tr_rate = preparer_sequences_differentielles(df_train_scaled)

if len(Tr_ram) == 0:
    raise ValueError("Erreur : Aucune séquence n'a pu être générée. Vérifie les données d'entrée.")

train_loader = DataLoader(TensorDataset(Tr_ram, Tr_phy, Tr_hist, Tr_rate), batch_size=64, shuffle=True)

Te_ram, Te_phy, Te_hist, Te_rate = preparer_sequences_differentielles(df_test_scaled)
test_loader = DataLoader(TensorDataset(Te_ram, Te_phy, Te_hist, Te_rate), batch_size=64, shuffle=False)

# =====================================================================
# L'ARCHITECTURE FAULT-TOLERANT (DOUBLE LSTM + CNN 1D)
# =====================================================================
class FaultTolerant_Dual_LSTM(nn.Module):
    def __init__(self, raman_raw_dim=2200, phys_dim=4, raman_latent_dim=32, hist_dim=1, lstm_hidden=64):
        super(FaultTolerant_Dual_LSTM, self).__init__()
        self.raman_latent_dim = raman_latent_dim
        
        # LOUPE RAMAN (CNN 1D)
        self.raman_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=2, padding=5), nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(32 * 16, raman_latent_dim), nn.ReLU()
        )
        
        # LSTM I : Variables de Processus de la cuve en cours (Raman Compressé + Physique)
        process_input_dim = raman_latent_dim + phys_dim
        self.lstm_process = nn.LSTM(process_input_dim, lstm_hidden, num_layers=2, batch_first=True, dropout=0.2)
        
        # LSTM II : Historique des mesures hors-lignes des anciennes cuves
        self.lstm_history = nn.LSTM(hist_dim, 32, num_layers=2, batch_first=True, dropout=0.2)
        
        # TÊTE DE FUSION (Prédiction de la vitesse de la Pénicilline)
        fusion_dim = lstm_hidden + 32
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1) # Sortie Ultra-Spécialisée : 1 valeur (La Vitesse Pénicilline)
        )

    def forward(self, seq_raman, seq_phys, seq_hist):
        batch_size, seq_len, _ = seq_raman.size()
        
        # 1. Traitement CNN sur le Raman
        ram_flat = seq_raman.contiguous().view(batch_size * seq_len, 1, -1)
        ram_latent = self.raman_encoder(ram_flat).view(batch_size, seq_len, self.raman_latent_dim)
        
        # 2. LSTM I (Processus Actuel)
        process_features = torch.cat((ram_latent, seq_phys), dim=2)
        lstm_out_proc, _ = self.lstm_process(process_features)
        last_proc = lstm_out_proc[:, -1, :] # Conclusion processus
        
        # 3. LSTM II (Historique de compensation)
        lstm_out_hist, _ = self.lstm_history(seq_hist)
        last_hist = lstm_out_hist[:, -1, :] # Conclusion historique
        
        # 4. Fusion et Prédiction du Différentiel
        fused = torch.cat((last_proc, last_hist), dim=1)
        return self.fusion_head(fused)

model = FaultTolerant_Dual_LSTM(phys_dim=len(colonnes_physiques))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

epochs = 100
meilleure_loss_test = float('inf')

print(f"\n🚀 Lancement de l'entraînement (Prédiction de Vitesse) sur {len(Tr_ram)} séquences...")

for epoch in range(epochs):
    model.train()
    loss_total = 0.0
    for b_ram, b_phy, b_hist, b_rate in train_loader:
        optimizer.zero_grad()
        pred_rate = model(b_ram, b_phy, b_hist)
        loss = criterion(pred_rate, b_rate)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    
    scheduler.step()
    train_loss = loss_total / len(train_loader)
    
    model.eval()
    test_loss_total = 0.0
    with torch.no_grad():
        for b_r, b_p, b_h, b_rate in test_loader:
            pred_test = model(b_r, b_p, b_h)
            test_loss_total += criterion(pred_test, b_rate).item()
            
    test_loss = test_loss_total / len(test_loader)
    
    if test_loss < meilleure_loss_test:
        meilleure_loss_test = test_loss
        torch.save(model.state_dict(), 'reakt_fault_tolerant_penicillin.pth')
        record = "⭐ Record (Tolérance aux Pannes) !"
    else:
        record = ""
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss (Rate): {train_loss:.5f} | Test Loss (Rate): {test_loss:.5f} {record}")

print("\n✅ Modèle hybride sauvegardé sous 'reakt_fault_tolerant_penicillin.pth'.")