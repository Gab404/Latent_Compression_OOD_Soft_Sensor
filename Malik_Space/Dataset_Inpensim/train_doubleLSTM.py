import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
import random
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

print("--- ENTRAÎNEMENT : DOUBLE PI-LSTM (PHYSICS-INFORMED & FAULT-TOLERANT) ---")

print("⏳ Chargement des données (IndPenSim V3)...")
df = pd.read_csv('IndPenSim_100_Batches.csv')
df = df.rename(columns={'Batch ID': 'Volume (L)'})

# Création du VRAI_Batch_ID
df = df.copy()
df['VRAI_Batch_ID'] = (df['Time (h)'] < df['Time (h)'].shift(1)).cumsum() + 1
df['VRAI_Batch_ID'] = df['VRAI_Batch_ID'].astype(int) 
df = df.sort_values(by=['VRAI_Batch_ID', 'Time (h)']).reset_index(drop=True)

# Sélection des 3 variables physiques (On a retiré l'Agitator RPM car variance = 0)
colonnes_physiques = ['pH(pH:pH)', 'Temperature(T:K)', 'Sugar feed rate(Fs:L/h)']
colonnes_raman = df.columns[39:] 

# Recherche de la cible Pénicilline
pen_cols = [c for c in df.columns if 'Penicillin' in c or 'penicillin' in c]
col_cible = pen_cols[0]
for c in pen_cols:
    if df[c].notna().sum() > 0:
        col_cible = c
        if 'Offline' in c: break

print(f"🎯 Cible détectée : {col_cible}")
colonnes_cibles = [col_cible]

# Interpolation étanche
df[col_cible] = df.groupby('VRAI_Batch_ID')[col_cible].transform(
    lambda cuve: cuve.interpolate(method='linear', limit_direction='both')
)

df_propre = df.dropna(subset=colonnes_cibles, how='any')
df_propre = df_propre.dropna(subset=colonnes_raman, how='all').fillna(0)
df_propre = df_propre.copy() 

batches_uniques = sorted(df_propre['VRAI_Batch_ID'].unique())

# Les 10 dernières cuves sont les PANNES (Test Set OOD)
cuves_normales = batches_uniques[:-10]
test_batches = batches_uniques[-10:] 

# Split 80/20 sur les 90 cuves normales (Seed 42)
random.seed(42)
random.shuffle(cuves_normales)
train_batches = cuves_normales[:72]
val_batches = cuves_normales[72:]

print(f"✂️  Séparation : TRAIN ({len(train_batches)}) | VAL ({len(val_batches)}) | TEST PANNES ({len(test_batches)})")

df_train = df_propre[df_propre['VRAI_Batch_ID'].isin(train_batches)].copy()
df_val   = df_propre[df_propre['VRAI_Batch_ID'].isin(val_batches)].copy()
df_test  = df_propre[df_propre['VRAI_Batch_ID'].isin(test_batches)].copy()

print("⚙️ Entraînement des Scalers...")
scaler_raman = StandardScaler().fit(df_train[colonnes_raman].values)
scaler_phys = StandardScaler().fit(df_train[colonnes_physiques].values)

# Pour le PI-LSTM, on a besoin de DEUX scalers Y (Un pour la concentration, un pour la vitesse)
scaler_y_conc = StandardScaler().fit(df_train[colonnes_cibles].values)

# Calcul du Process Rate
df_train['Rate_Pen'] = df_train.groupby('VRAI_Batch_ID')[col_cible].diff().fillna(0)
df_val['Rate_Pen']   = df_val.groupby('VRAI_Batch_ID')[col_cible].diff().fillna(0)
df_test['Rate_Pen']  = df_test.groupby('VRAI_Batch_ID')[col_cible].diff().fillna(0)

scaler_y_rate = StandardScaler().fit(df_train[['Rate_Pen']].values)

joblib.dump(scaler_raman, 'scaler_raman_pi.pkl')
joblib.dump(scaler_phys, 'scaler_phys_pi.pkl')
joblib.dump(scaler_y_conc, 'scaler_y_conc_pi.pkl')
joblib.dump(scaler_y_rate, 'scaler_y_rate_pi.pkl')

def apply_scaling(dataframe):
    df_scaled = dataframe.copy()
    df_scaled[colonnes_raman] = scaler_raman.transform(dataframe[colonnes_raman].values)
    df_scaled[colonnes_physiques] = scaler_phys.transform(dataframe[colonnes_physiques].values)
    df_scaled[col_cible] = scaler_y_conc.transform(dataframe[colonnes_cibles].values)
    df_scaled['Rate_Pen'] = scaler_y_rate.transform(dataframe[['Rate_Pen']].values)
    return df_scaled

df_train_scaled = apply_scaling(df_train)
df_val_scaled   = apply_scaling(df_val)
df_test_scaled  = apply_scaling(df_test)

# Historique (LSTM II) basé sur la VRAIE concentration finale scalée
dict_hist_y = {}
for b in batches_uniques:
    valeur_finale_brute = df_propre[df_propre['VRAI_Batch_ID'] == b][col_cible].iloc[-1]
    dict_hist_y[int(b)] = scaler_y_conc.transform([[valeur_finale_brute]])[0]

SEQ_LEN = 10 
W_HIST = 5   

def preparer_sequences_pi(dataframe):
    X_ram, X_phy, X_hist = [], [], []
    Y_rate, Y_conc_prev, Y_conc_target = [], [], []
    
    for b in dataframe['VRAI_Batch_ID'].unique():
        b = int(b) 
        df_b = dataframe[dataframe['VRAI_Batch_ID'] == b]
        if len(df_b) <= SEQ_LEN: continue
            
        hist_y = [dict_hist_y.get(past_b, [0.0]) for past_b in range(b - W_HIST, b)]
                
        ram_scaled = df_b[colonnes_raman].values
        phy_scaled = df_b[colonnes_physiques].values
        rate_scaled = df_b['Rate_Pen'].values
        conc_scaled = df_b[col_cible].values
        
        for i in range(len(df_b) - SEQ_LEN):
            X_ram.append(ram_scaled[i : i + SEQ_LEN])
            X_phy.append(phy_scaled[i : i + SEQ_LEN])
            X_hist.append(hist_y)
            
            # Cibles pour la PI-Loss
            Y_rate.append([rate_scaled[i + SEQ_LEN]])             # La vitesse à deviner
            Y_conc_prev.append([conc_scaled[i + SEQ_LEN - 1]])    # La concentration juste avant (t-1)
            Y_conc_target.append([conc_scaled[i + SEQ_LEN]])      # La vraie concentration cible à t
            
    return (torch.FloatTensor(np.array(X_ram)), torch.FloatTensor(np.array(X_phy)), 
            torch.FloatTensor(np.array(X_hist)), torch.FloatTensor(np.array(Y_rate)),
            torch.FloatTensor(np.array(Y_conc_prev)), torch.FloatTensor(np.array(Y_conc_target)))

print("⚙️ Fabrication des tenseurs temporels pour PI-LSTM...")
Tr_ram, Tr_phy, Tr_hist, Tr_rate, Tr_c_prev, Tr_c_tgt = preparer_sequences_pi(df_train_scaled)
train_loader = DataLoader(TensorDataset(Tr_ram, Tr_phy, Tr_hist, Tr_rate, Tr_c_prev, Tr_c_tgt), batch_size=64, shuffle=True)

Va_ram, Va_phy, Va_hist, Va_rate, Va_c_prev, Va_c_tgt = preparer_sequences_pi(df_val_scaled)
val_loader = DataLoader(TensorDataset(Va_ram, Va_phy, Va_hist, Va_rate, Va_c_prev, Va_c_tgt), batch_size=64, shuffle=False)

Te_ram, Te_phy, Te_hist, Te_rate, Te_c_prev, Te_c_tgt = preparer_sequences_pi(df_test_scaled)
test_loader = DataLoader(TensorDataset(Te_ram, Te_phy, Te_hist, Te_rate, Te_c_prev, Te_c_tgt), batch_size=64, shuffle=False)

class Double_PI_LSTM(nn.Module):
    def __init__(self, raman_latent_dim=32, phys_dim=3, hist_dim=1, lstm_hidden=64):
        super(Double_PI_LSTM, self).__init__()
        self.raman_latent_dim = raman_latent_dim
        
        # CNN 1D
        self.raman_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=2, padding=5), nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(32 * 16, raman_latent_dim), nn.ReLU()
        )
        
        # LSTM I (Processus actuel)
        self.lstm_process = nn.LSTM(raman_latent_dim + phys_dim, lstm_hidden, num_layers=2, batch_first=True, dropout=0.2)
        
        # LSTM II (Historique/Pannes)
        self.lstm_history = nn.LSTM(hist_dim, 32, num_layers=2, batch_first=True, dropout=0.2)
        
        self.fusion_head = nn.Sequential(
            nn.Linear(lstm_hidden + 32, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1) # Prédit la vitesse
        )

    def forward(self, seq_raman, seq_phys, seq_hist):
        batch_size, seq_len, _ = seq_raman.size()
        ram_flat = seq_raman.contiguous().view(batch_size * seq_len, 1, -1)
        ram_latent = self.raman_encoder(ram_flat).view(batch_size, seq_len, self.raman_latent_dim)
        
        lstm_out_proc, _ = self.lstm_process(torch.cat((ram_latent, seq_phys), dim=2))
        lstm_out_hist, _ = self.lstm_history(seq_hist)
        
        return self.fusion_head(torch.cat((lstm_out_proc[:, -1, :], lstm_out_hist[:, -1, :]), dim=1))

model = Double_PI_LSTM(phys_dim=len(colonnes_physiques))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Paramètres de dé-normalisation (Pour calculer la physique dans le monde RÉEL)
mu_r = torch.FloatTensor(scaler_y_rate.mean_)
sig_r = torch.FloatTensor(scaler_y_rate.scale_)
mu_c = torch.FloatTensor(scaler_y_conc.mean_)
sig_c = torch.FloatTensor(scaler_y_conc.scale_)

# Poids des Loss Physiques
LAMBDA_CONC = 0.5 # Force d'Ancrage
LAMBDA_MONO = 0.1 # Force de Monotonicité

def compute_pi_loss(pred_rate_scaled, true_rate_scaled, prev_conc_scaled, target_conc_scaled):
    # 1. Loss standard sur le scaling (Facilite la convergence du réseau)
    loss_rate = criterion(pred_rate_scaled, true_rate_scaled)
    
    # Passage dans l'espace RÉEL (g/L)
    pred_rate_real = pred_rate_scaled * sig_r + mu_r
    prev_conc_real = prev_conc_scaled * sig_c + mu_c
    target_conc_real = target_conc_scaled * sig_c + mu_c
    
    # 2. Loss d'Ancrage (Reconstitution de la concentration réelle)
    pred_conc_real = prev_conc_real + pred_rate_real
    loss_anchoring = criterion(pred_conc_real, target_conc_real)
    
    # 3. Loss de Monotonicité (Pénicilline >= 0, la vitesse ne peut pas être <-0.1 g/L)
    # On autorise une micro baisse due à la dilution, mais pas plus.
    loss_monotonicity = torch.mean(torch.relu(-(pred_rate_real + 0.1)))
    
    return loss_rate + LAMBDA_CONC * loss_anchoring + LAMBDA_MONO * loss_monotonicity, loss_rate

epochs = 80
meilleure_loss_val = float('inf')

print(f"\n🚀 Lancement de l'entraînement Double PI-LSTM...")

for epoch in range(epochs):
    # --- TRAIN ---
    model.train()
    train_loss_total = 0.0
    for b_ram, b_phy, b_hist, b_rate, b_c_prev, b_c_tgt in train_loader:
        optimizer.zero_grad()
        
        pred_rate_scaled = model(b_ram, b_phy, b_hist)
        
        # Calcul de la perte hybride (PI-Loss)
        loss, _ = compute_pi_loss(pred_rate_scaled, b_rate, b_c_prev, b_c_tgt)
        
        loss.backward()
        optimizer.step()
        train_loss_total += loss.item()
    
    scheduler.step()
    train_loss = train_loss_total / len(train_loader)
    
    # --- VALIDATION (Saine) ---
    model.eval()
    val_loss_total = 0.0
    val_pure_rate_loss = 0.0
    with torch.no_grad():
        for b_r, b_p, b_h, b_rate, b_c_prev, b_c_tgt in val_loader:
            pred_rate_scaled = model(b_r, b_p, b_h)
            loss_pi, loss_r = compute_pi_loss(pred_rate_scaled, b_rate, b_c_prev, b_c_tgt)
            val_loss_total += loss_pi.item()
            val_pure_rate_loss += loss_r.item()
            
    val_loss = val_loss_total / len(val_loader)
    
    # --- TEST (Pannes OOD) ---
    test_loss_total = 0.0
    with torch.no_grad():
        for b_r, b_p, b_h, b_rate, b_c_prev, b_c_tgt in test_loader:
            pred_rate_scaled = model(b_r, b_p, b_h)
            loss_pi, _ = compute_pi_loss(pred_rate_scaled, b_rate, b_c_prev, b_c_tgt)
            test_loss_total += loss_pi.item()
            
    test_loss = test_loss_total / len(test_loader)
    
    if val_loss < meilleure_loss_val:
        meilleure_loss_val = val_loss
        torch.save(model.state_dict(), 'reakt_pi_dual_lstm.pth')
        record = "⭐ Record Validé !"
    else:
        record = ""
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:03d}/{epochs} | Train PI-Loss: {train_loss:.5f} | Val PI-Loss: {val_loss:.5f} | Test (Faults): {test_loss:.5f} {record}")

print("\n✅ Modèle Double PI-LSTM sauvegardé sous 'reakt_pi_dual_lstm.pth'.")