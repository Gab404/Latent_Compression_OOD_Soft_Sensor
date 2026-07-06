import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

print("⏳ Étape Benchmark : Entraînement Baseline LR et PI-LSTM Standard sur 8 Lots...")

# =====================================================================
# 1. ARCHITECTURE PI-LSTM STANDARD (20 Entrées Directes)
# =====================================================================
class StandardPILSTM(nn.Module):
    def __init__(self, input_dim=20, hidden_size=64):
        super(StandardPILSTM, self).__init__()
        # Le LSTM prend directement les 20 variables en entrée sans couche intermédiaire
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc_bio = nn.Linear(hidden_size, 2) 

    def forward(self, x_seq):
        # x_seq shape: [Batch, Seq_Len=60, Dims=20]
        lstm_out, _ = self.lstm(x_seq)
        return self.fc_bio(lstm_out[:, -1, :])

# =====================================================================
# 2. PRÉPARATION DES SÉQUENCES (PAR BATCH)
# =====================================================================
def preparer_sequences_par_batch(df_norm, sequence_length=60):
    X_seq, Y_seq = [], []
    for batch_id in df_norm['Batch_ID'].unique():
        df_batch = df_norm[df_norm['Batch_ID'] == batch_id]
        
        X_batch = df_batch.iloc[:, :20].values
        Y_batch = df_batch.iloc[:, 20:22].values
        
        for i in range(len(X_batch) - sequence_length):
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
        print("🛑 ERREUR : Le fichier 'dataset_pichia_multibatch.csv' est introuvable.")
        exit()

    # --- NORMALISATION ---
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    
    df_scaled = df.copy()
    df_scaled.iloc[:, :20] = scaler_X.fit_transform(df.iloc[:, :20])
    df_scaled.iloc[:, 20:22] = scaler_Y.fit_transform(df.iloc[:, 20:22])
    
    # --- SPLIT TRAIN (Lots 1 à 7) / TEST (Lot 8) ---
    print("✂️ Séparation : Entraînement sur Exp 1 à 7, Test sur Exp 8...")
    df_train = df_scaled[df_scaled['Batch_ID'] <= 7]
    df_test = df_scaled[df_scaled['Batch_ID'] == 8]
    
    # Création des séquences 3D
    SEQ_LENGTH = 60
    X_train, Y_train = preparer_sequences_par_batch(df_train, SEQ_LENGTH)
    X_test, Y_test = preparer_sequences_par_batch(df_test, SEQ_LENGTH)
    
    idx_P_in = 19  # Index de Proteine_P_in pour la perte physique
    
    # =====================================================================
    # MODÈLE 1 : BASELINE RÉGRESSION LINÉAIRE (20 VARIABLES)
    # =====================================================================
    print("\n🖥️ Entraînement de la Régression Linéaire (Statique)...")
    # Pour évaluer la LR sur les mêmes points, on prend l'instant t de chaque séquence (le dernier pas de temps)
    X_train_lr = X_train[:, -1, :].numpy()
    X_test_lr = X_test[:, -1, :].numpy()
    Y_train_lr = Y_train.numpy()
    Y_test_lr = Y_test.numpy()
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_lr, Y_train_lr)
    
    pred_lr_norm = lr_model.predict(X_test_lr)
    pred_lr_real = scaler_Y.inverse_transform(pred_lr_norm)
    
    # =====================================================================
    # MODÈLE 2 : PI-LSTM STANDARD (20 VARIABLES)
    # =====================================================================
    print("\n🧠 Entraînement du PI-LSTM Standard...")
    lstm_model = StandardPILSTM(input_dim=20, hidden_size=64)
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    mse_criterion = nn.MSELoss()

    epochs = 50
    physics_lambda = 0.5 
    
    for epoch in range(epochs):
        lstm_model.train()
        optimizer.zero_grad()
        
        outputs = lstm_model(X_train)
        loss_mse = mse_criterion(outputs, Y_train)
        
        # Contrainte Physique (Monotonicité)
        P_pred = outputs[:, 1]
        P_measurable_t = X_train[:, -1, idx_P_in] 
        loss_physics = torch.mean(torch.relu(P_measurable_t - P_pred))
        
        total_loss = loss_mse + physics_lambda * loss_physics
        total_loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Époque [{epoch+1}/{epochs}] - Total Loss: {total_loss.item():.6f}")

    # Évaluation du PI-LSTM
    lstm_model.eval()
    with torch.no_grad():
        pred_lstm_norm = lstm_model(X_test).numpy()
    pred_lstm_real = scaler_Y.inverse_transform(pred_lstm_norm)

    # =====================================================================
    # CALCUL DES MÉTRIQUES ET COMPARAISON (PROTÉINE SUR LOT 8)
    # =====================================================================
    Y_test_real = scaler_Y.inverse_transform(Y_test.numpy())
    P_true = Y_test_real[:, 1]
    
    P_pred_lr = pred_lr_real[:, 1]
    P_pred_lstm = pred_lstm_real[:, 1]
    
    # Métriques LR
    rmse_lr = np.sqrt(mean_squared_error(P_true, P_pred_lr))
    mae_lr = mean_absolute_error(P_true, P_pred_lr)
    r2_lr = r2_score(P_true, P_pred_lr)
    
    # Métriques LSTM Standard
    rmse_lstm = np.sqrt(mean_squared_error(P_true, P_pred_lstm))
    mae_lstm = mean_absolute_error(P_true, P_pred_lstm)
    r2_lstm = r2_score(P_true, P_pred_lstm)
    
    print("\n=====================================================================")
    print("--- RÉSULTATS COMPARATIFS SUR L'EXPÉRIENCE 8 (ZONE DE TEST) ---")
    print(f"Régression Linéaire (20V)  -> RMSE: {rmse_lr:.2f} | MAE: {mae_lr:.2f} | R²: {r2_lr:.4f}")
    print(f"PI-LSTM Standard (20V)     -> RMSE: {rmse_lstm:.2f} | MAE: {mae_lstm:.2f} | R²: {r2_lstm:.4f}")
    print("=====================================================================")

    # =====================================================================
    # GRAPHIC COMPARATIF TEMPOREL
    # =====================================================================
    plt.figure(figsize=(14, 6))
    plt.plot(P_true, color='black', label='Réalité (Exp 8)', linewidth=2.5)
    plt.plot(P_pred_lr, color='red', linestyle='-.', label=f'Baseline LR 20V (MAE={mae_lr:.1f})', linewidth=1.5)
    plt.plot(P_pred_lstm, color='blue', linestyle='--', label=f'PI-LSTM Standard 20V (MAE={mae_lstm:.1f})', linewidth=1.5)
    
    plt.title('R&D REAKT : Analyse Comparative de Généralisation sur l\'Expérience 8', fontsize=16, fontweight='bold')
    plt.xlabel('Temps (Minutes écoulées sur le lot de Test)', fontsize=12)
    plt.ylabel('Concentration Protéine (mg/L)', fontsize=12)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()