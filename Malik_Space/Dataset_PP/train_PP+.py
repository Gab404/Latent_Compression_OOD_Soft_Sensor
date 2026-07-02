import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

print("⏳ Étape R&D : Entraînement REAKT E2E (Train: Lots 1-6 | Test: Lots 7-8)...")

# =====================================================================
# 1. ARCHITECTURE E2E (Fusion Multimodale : Permittivité + Physique)
# =====================================================================
class E2EDispersionFusionNet(nn.Module):
    def __init__(self, input_perm_dim=13, input_phys_dim=7, latent_disp_dim=8, hidden_size=64):
        super(E2EDispersionFusionNet, self).__init__()
        self.latent_disp_dim = latent_disp_dim
        
        # --- Encodeur Dispersion Temporel-Distribué (TDDE) ---
        self.encoder = nn.Sequential(
            nn.Linear(input_perm_dim, 13), 
            nn.ReLU(), 
            nn.Dropout(0.2), 
            nn.Linear(13, latent_disp_dim)
        )
        
        # --- Cœur Temporel Grey-Box (PI-LSTM) ---
        self.fusion_dim = latent_disp_dim + input_phys_dim
        self.lstm = nn.LSTM(self.fusion_dim, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        
        # --- Tête de Prédiction Bio ---
        self.fc_bio = nn.Linear(hidden_size, 2) 

    def forward(self, seq_all_inputs):
        batch_size, seq_len, _ = seq_all_inputs.size()
        
        # Séparation des flux : 13 fréquences (Papier 1) et 7 variables physiques (Papier 2)
        x_perm_seq = seq_all_inputs[:, :, :13]
        x_phys_seq = seq_all_inputs[:, :, 13:]
        
        # Compression intelligente de la permittivité
        x_perm_flat = x_perm_seq.contiguous().view(batch_size * seq_len, 13)
        latent_perm_seq = self.encoder(x_perm_flat).view(batch_size, seq_len, self.latent_disp_dim)
        
        # Fusion Contextuelle
        fused_features = torch.cat((latent_perm_seq, x_phys_seq), dim=2)
        
        # Traitement séquentiel
        lstm_out, _ = self.lstm(fused_features)
        
        return self.fc_bio(lstm_out[:, -1, :])

# =====================================================================
# 2. PRÉPARATION DES SÉQUENCES (PAR BATCH)
# =====================================================================
def preparer_sequences_par_batch(df_norm, sequence_length=60):
    X_seq, Y_seq = [], []
    
    # On itère par Batch_ID pour ne pas créer de séquences à cheval sur deux cuves
    for batch_id in df_norm['Batch_ID'].unique():
        df_batch = df_norm[df_norm['Batch_ID'] == batch_id]
        
        X_batch = df_batch.iloc[:, :20].values
        Y_batch = df_batch.iloc[:, 20:22].values
        
        for i in range(len(X_batch) - sequence_length):
            # Respect de la contrainte structurelle exigée pour l'itération
            if True: 
                X_seq.append(X_batch[i : i + sequence_length])
                Y_seq.append(Y_batch[i + sequence_length])
            
    return torch.tensor(np.array(X_seq), dtype=torch.float32), torch.tensor(np.array(Y_seq), dtype=torch.float32)

# =====================================================================
# EXECUTION PRINCIPALE
# =====================================================================
if __name__ == "__main__":
    
    print("🚀 Chargement du dataset Multi-Batch...")
    try:
        df = pd.read_csv("dataset_pichia_multibatch.csv")
    except FileNotFoundError:
        print("🛑 ERREUR : Lance le script de création multi-batch d'abord.")
        exit()

    # --- NORMALISATION ---
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    
    # On normalise X (colonnes 0 à 19) et Y (colonnes 20 à 21)
    df_scaled = df.copy()
    df_scaled.iloc[:, :20] = scaler_X.fit_transform(df.iloc[:, :20])
    df_scaled.iloc[:, 20:22] = scaler_Y.fit_transform(df.iloc[:, 20:22])
    
    # --- SPLIT STRATÉGIQUE (TRAIN: 1 à 6 | TEST: 7 et 8) ---
    print("✂️ Séparation des données en cours...")
    print("   -> Entraînement (Train) : Lots 1, 2, 3, 4, 5, 6")
    print("   -> Validation (Test)    : Lots 7, 8")
    
    df_train = df_scaled[df_scaled['Batch_ID'] <= 6]
    df_test = df_scaled[df_scaled['Batch_ID'] >= 7]
    
    # Création des séquences 3D avec mémoire de 60 minutes
    SEQ_LENGTH = 60
    X_train, Y_train = preparer_sequences_par_batch(df_train, SEQ_LENGTH)
    X_test, Y_test = preparer_sequences_par_batch(df_test, SEQ_LENGTH)
    
    idx_P_in = 19  # Index de la variable d'état actuel pour la contrainte physique
    
    # --- ENTRAÎNEMENT ---
    print("\n✅ Initialisation REAKT E2E...")
    model = E2EDispersionFusionNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mse_criterion = nn.MSELoss()

    epochs_lstm = 50
    physics_lambda = 0.5 
    
    print("🧠 Début de l'optimisation PI-LSTM sur 6 lots...")
    for epoch in range(epochs_lstm):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        mse_loss_lstm = mse_criterion(outputs, Y_train)
        
        # --- Grey-Box Physics Loss ---
        P_pred = outputs[:, 1]
        P_measurable_t = X_train[:, -1, idx_P_in] 
        physics_loss = torch.mean(torch.relu(P_measurable_t - P_pred))
        
        total_loss = mse_loss_lstm + physics_lambda * physics_loss
        total_loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs_lstm}] - Total Loss: {total_loss.item():.6f}")

    # =====================================================================
    # 3. ÉVALUATION EN AVEUGLE (SUR LES LOTS 7 ET 8)
    # =====================================================================
    print("\n📊 Évaluation de l'extrapolation sur les Expériences 7 et 8...")
    model.eval()
    with torch.no_grad():
        test_outputs_norm = model(X_test)
        
    pred_y_real = scaler_Y.inverse_transform(test_outputs_norm.numpy())
    true_y_real = scaler_Y.inverse_transform(Y_test.numpy())
    
    P_pred_test = pred_y_real[:, 1]
    P_true_test = true_y_real[:, 1]

    rmse_P = np.sqrt(mean_squared_error(P_true_test, P_pred_test))
    mae_P = mean_absolute_error(P_true_test, P_pred_test)
    r2_P = r2_score(P_true_test, P_pred_test)
    
    print(f"\n=========================================================")
    print(f"--- RÉSULTATS REAKT E2E (SUR LOTS 7 & 8 INCONNUS) ---")
    print(f"RMSE : {rmse_P:.2f} mg/L")
    print(f"MAE  : {mae_P:.2f} mg/L")
    print(f"R²   : {r2_P:.4f}")
    print(f"=========================================================")

    # --- VISUELS ---
    plt.figure(figsize=(14, 6))
    plt.plot(P_true_test, color='black', label='Mesure Réelle (Lots 7 & 8 concaténés)', linewidth=2.5)
    plt.plot(P_pred_test, color='#2ca02c', linestyle='--', label='Prédiction REAKT E2E', linewidth=2)
    
    # Ligne de séparation verticale approximative entre le lot 7 et le lot 8
    # (Calculée en fonction de la taille du lot 7 dans le test set)
    len_lot_7 = len(df_test[df_test['Batch_ID'] == 7]) - SEQ_LENGTH
    if len_lot_7 > 0:
        plt.axvline(x=len_lot_7, color='gray', linestyle=':', label='Séparation Lot 7 / Lot 8')

    plt.title('Généralisation REAKT : Prédiction en Aveugle sur les Expériences 7 et 8', fontsize=16, fontweight='bold')
    plt.xlabel('Échantillons Temporels (Minutes)', fontsize=12)
    plt.ylabel('Concentration Protéine (mg/L)', fontsize=12)
    
    textstr = f'RMSE = {rmse_P:.2f}\nMAE = {mae_P:.2f}\n$R^2$ = {r2_P:.4f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=props)

    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()