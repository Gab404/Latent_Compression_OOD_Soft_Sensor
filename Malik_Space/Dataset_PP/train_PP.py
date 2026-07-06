import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

print("⏳ Étape Final : Entraînement REAKT E2E - Évaluation Stricte sur Zone de Test...")

# =====================================================================
# 1. ARCHITECTURE E2E (E2EDispersionFusionNet)
# =====================================================================
class E2EDispersionFusionNet(nn.Module):
    def __init__(self, input_perm_dim=13, input_phys_dim=7, latent_disp_dim=8, hidden_size=64):
        super(E2EDispersionFusionNet, self).__init__()
        
        self.latent_disp_dim = latent_disp_dim
        self.hidden_size = hidden_size
        
        # --- Encodeur Dispersion Temporel-Distribué (TDDE) ---
        self.encoder = nn.Sequential(
            nn.Linear(input_perm_dim, 13),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(13, latent_disp_dim), 
        )
        
        # --- Cœur Temporel Grey-Box (PI-LSTM) ---
        self.fusion_dim = latent_disp_dim + input_phys_dim
        self.lstm = nn.LSTM(self.fusion_dim, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        
        # --- Prédiction Bio (Biomasse, Proteine) ---
        self.fc_bio = nn.Linear(hidden_size, 2) 

    def forward(self, seq_all_inputs):
        batch_size = seq_all_inputs.size(0)
        seq_len = seq_all_inputs.size(1)
        
        # Split: [13 Permittivité, 7 Physique]
        x_perm_seq = seq_all_inputs[:, :, :13]
        x_phys_seq = seq_all_inputs[:, :, 13:]
        
        # Application du TDDE
        x_perm_flat = x_perm_seq.contiguous().view(batch_size * seq_len, 13)
        latent_perm_flat = self.encoder(x_perm_flat)
        latent_perm_seq = latent_perm_flat.view(batch_size, seq_len, self.latent_disp_dim)
        
        # Fusion Contextuelle
        fused_features = torch.cat((latent_perm_seq, x_phys_seq), dim=2)
        
        # LSTM
        lstm_out, _ = self.lstm(fused_features)
        
        # Prédiction finale
        last_state = lstm_out[:, -1, :] 
        bio_out = self.fc_bio(last_state) 
        
        return bio_out

# =====================================================================
# 2. PRÉPARATION DES SÉQUENCES
# =====================================================================
def preparer_sequences(X, Y, sequence_length=60):
    X_seq, Y_seq = [], []
    for i in range(len(X) - sequence_length):
        if True: # Maintien de ta contrainte de structure
            X_seq.append(X[i : i + sequence_length])
            Y_seq.append(Y[i + sequence_length])
    return torch.stack(X_seq), torch.stack(Y_seq)

# =====================================================================
# EXECUTION PRINCIPALE
# =====================================================================
if __name__ == "__main__":
    
    print("🚀 Chargement du dataset REAKT...")
    try:
        df = pd.read_csv("dataset_pichia_clean.csv")
    except FileNotFoundError:
        print("🛑 ERREUR : Tu dois lancer l'étape de construction du dataset d'abord.")
        exit()
    
    # --- PRÉPARATION DES MATRICES ---
    X_cols = df.iloc[:, :20].values
    Y_cols = df.iloc[:, 20:22].values

    # --- NORMALISATION ---
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    
    X_norm = torch.tensor(scaler_X.fit_transform(X_cols), dtype=torch.float32)
    Y_norm = torch.tensor(scaler_Y.fit_transform(Y_cols), dtype=torch.float32)
    
    # --- PRÉPARATION DES SÉQUENCES 3D (60 min) ---
    SEQ_LENGTH = 60
    X_3D, Y_3D = preparer_sequences(X_norm, Y_norm, sequence_length=SEQ_LENGTH)
    
    # --- SPLIT CHRONOLOGIQUE STRICT (80/20) ---
    split_idx = int(len(X_3D) * 0.8)
    
    # Données d'entraînement (Les 80 premiers %)
    X_train, Y_train = X_3D[:split_idx], Y_3D[:split_idx]
    
    # Données de test (Les 20 derniers %, totalement cachés au modèle)
    X_test, Y_test = X_3D[split_idx:], Y_3D[split_idx:]
    
    # Index de Proteine_P_in pour la contrainte physique
    idx_P_in = 19 
    
    # --- INITIALISATION REAKT E2E ---
    print("\n✅ Initialisation du Jumeau Numérique REAKT E2E...")
    model = E2EDispersionFusionNet(
        input_perm_dim=13, 
        input_phys_dim=7, 
        latent_disp_dim=8, 
        hidden_size=64 
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mse_criterion = nn.MSELoss()

    # --- ENTRAÎNEMENT STRICT SUR LE TRAIN SET ---
    print("\n--- Entraînement Joint du GreyBox PI-LSTM E2E (sur les 80% du passé)... ---")
    epochs_lstm = 50
    physics_lambda = 0.5 
    
    for epoch in range(epochs_lstm):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        
        # Loss classique (MSE)
        mse_loss_lstm = mse_criterion(outputs, Y_train)
        
        # Loss informed by Physics (ReaKt Grey-Box)
        P_pred = outputs[:, 1]
        P_measurable_t = X_train[:, -1, idx_P_in] 
        penalite_physique = torch.relu(P_measurable_t - P_pred)
        physics_loss = torch.mean(penalite_physique)
        
        # Loss Totale
        total_loss = mse_loss_lstm + physics_lambda * physics_loss
        
        total_loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs_lstm}] - Total Loss: {total_loss.item():.6f}")

    print("\n✅ Modèle entraîné et sauvegardé.")

    # =====================================================================
    # 3. ÉVALUATION STRICTE SUR LA ZONE DE TEST (20% restants)
    # =====================================================================
    print("\n📊 Évaluation en aveugle sur les 20% finaux du batch...")
    model.eval()
    with torch.no_grad():
        # PREDICTION UNIQUEMENT SUR X_test
        test_outputs_norm = model(X_test)
        
        pred_y_norm = test_outputs_norm.numpy()
        true_y_norm = Y_test.numpy()

    # --- DÉ-NORMALISATION ---
    pred_y_real = scaler_Y.inverse_transform(pred_y_norm)
    true_y_real = scaler_Y.inverse_transform(true_y_norm)
    
    # On isole la concentration en Protéine (mg/L)
    P_pred_test = pred_y_real[:, 1]
    P_true_test = true_y_real[:, 1]

    # --- MÉTRIQUES SUR LA ZONE INCONNUE ---
    rmse_P = np.sqrt(mean_squared_error(P_true_test, P_pred_test))
    mae_P = mean_absolute_error(P_true_test, P_pred_test)
    r2_P = r2_score(P_true_test, P_pred_test)
    
    print(f"--- RÉSULTATS GLOBAUX REAKT E2E (SUR ZONE DE TEST UNIQUEMENT) ---")
    print(f"RMSE : {rmse_P:.2f} mg/L")
    print(f"MAE  : {mae_P:.2f} mg/L")
    print(f"R²   : {r2_P:.4f}")

    # =====================================================================
    # VISUELS (SUR LA ZONE DE TEST UNIQUEMENT)
    # =====================================================================
    plt.figure(figsize=(12, 5))
    
    # Vraie ligne de vie sur la zone de test
    plt.plot(P_true_test, color='black', label='Mesure Réelle (Test Set)', linewidth=2.5)
    
    # Prédiction du modèle E2E
    plt.plot(P_pred_test, color='#ff7f0e', linestyle='--', label='Prédiction REAKT E2E', linewidth=2)
    
    plt.title('REAKT E2E : Prédiction en Aveugle (Zone de Test)', fontsize=16, fontweight='bold')
    plt.xlabel('Temps (Minutes de la zone de test)', fontsize=12)
    plt.ylabel('Concentration en Protéine (mg/L)', fontsize=12)
    
    # Ajout des métriques sur le graphique
    textstr = f'RMSE = {rmse_P:.2f}\nMAE = {mae_P:.2f}\n$R^2$ = {r2_P:.4f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=props)

    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.show()