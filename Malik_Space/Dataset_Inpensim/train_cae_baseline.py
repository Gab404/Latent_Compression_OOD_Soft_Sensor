import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

print("--- INITIALISATION DU FILTRE CAE+ (BASELINE REMOVAL) ---")

# =====================================================================
# 1. ARCHITECTURE ET FONCTION DE PERTE (Recopiées ici pour l'autonomie)
# =====================================================================
class CAE_Plus(nn.Module):
    def __init__(self, seq_len=2200):
        super(CAE_Plus, self).__init__()
        self.seq_len = seq_len
        
        # L'Encodeur (Force l'oubli des détails)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=15, stride=2, padding=7), nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(8, 16, kernel_size=15, stride=2, padding=7), nn.ReLU(),
            nn.AvgPool1d(2)
        )
        
        # Le Décodeur (Reconstruit la colline)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size=15, stride=2, padding=7, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(8, 1, kernel_size=15, stride=2, padding=7, output_padding=1)
        )

    def forward(self, signal_brut):
        x_latent = self.encoder(signal_brut)
        ligne_de_base = self.decoder(x_latent)
        ligne_de_base = F.interpolate(ligne_de_base, size=self.seq_len, mode='linear', align_corners=False)
        
        # La contrainte stricte : la ligne de base ne dépasse jamais le signal
        ligne_de_base_finale = torch.min(signal_brut, ligne_de_base)
        return ligne_de_base_finale

def perte_asymetrique_cae(baseline_predite, signal_brut, p=0.05):
    difference = signal_brut - baseline_predite
    # Pénalité asymétrique forte si on coupe un pic, faible si on est sous la courbe
    loss = torch.where(difference < 0, (1 - p) * (difference ** 2), p * (difference ** 2))
    return torch.mean(loss)

# =====================================================================
# 2. PRÉPARATION DES DONNÉES (Non-supervisé : on ne prend que le Raman !)
# =====================================================================
print("Chargement des spectres bruts...")

df = pd.read_csv('IndPenSim_100_Batches.csv')
colonnes_raman = df.columns[39:]
df_propre = df.dropna(subset=colonnes_raman, how='all').fillna(0)

# On extrait purement le signal optique brut (Pas besoin de scaler complexe ici)
spectres_bruts = df_propre[colonnes_raman].values

# Formatage pour PyTorch : [Nombre de spectres, Canaux (1), Longueur (2200)]
X_tensor = torch.FloatTensor(spectres_bruts).unsqueeze(1)

# Création d'un DataLoader très simple (Pas de Y cible !)
dataset = TensorDataset(X_tensor)
# Un batch size plus grand (128) car le modèle est très léger
dataloader = DataLoader(dataset, batch_size=128, shuffle=True) 

# =====================================================================
# 3. BOUCLE D'ENTRAÎNEMENT NON-SUPERVISÉ
# =====================================================================
print("\n--- DÉMARRAGE DE L'ENTRAÎNEMENT CAE+ ---")

# Instanciation (avec 2200 points comme vu dans tes tenseurs précédents)
model_cae = CAE_Plus(seq_len=2200)

# L'optimiseur
optimizer = optim.Adam(model_cae.parameters(), lr=0.001)

epochs = 50 # 50 époques suffisent généralement pour un autoencodeur
meilleure_loss = float('inf')

temps_debut = time.time()

for epoch in range(epochs):
    model_cae.train()
    train_loss = 0.0
    
    for batch in dataloader:
        x_brut = batch[0] # On récupère juste le spectre brut
        
        optimizer.zero_grad()
        
        # Le réseau propose une ligne de base
        baseline_predite = model_cae(x_brut)
        
        # On calcule l'erreur avec notre formule magique
        loss = perte_asymetrique_cae(baseline_predite, x_brut, p=0.05)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    train_loss /= len(dataloader)
    
    # Sauvegarde (Ici on sauvegarde simplement si la loss d'entraînement diminue)
    if train_loss < meilleure_loss:
        meilleure_loss = train_loss
        torch.save(model_cae.state_dict(), 'cae_baseline_best.pth')
        
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Époque {epoch+1:03d}/{epochs} | Loss Asymétrique: {train_loss:.6f}")

temps_total = (time.time() - temps_debut) / 60
print(f"\n✅ Filtre entraîné en {temps_total:.1f} minutes !")
print(f"Les poids du nettoyeur sont sauvés sous 'cae_baseline_best.pth'.")