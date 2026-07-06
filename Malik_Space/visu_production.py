import pandas as pd
import matplotlib.pyplot as plt

print("⏳ Chargement du dataset IndPenSim...")
df = pd.read_csv('100_Batches_IndPenSim_V3.csv')

# Le nom exact de la colonne dans IndPenSim V3
colonne_agitateur = 'Agitator RPM(RPM:RPM)'

# Extraction de toutes les valeurs
valeurs_agitateur = df[colonne_agitateur]

print(f"\n📊 Statistiques globales de l'Agitateur (sur toutes les cuves) :")
print("-" * 50)
print(valeurs_agitateur.describe())
print("-" * 50)

# Pour que ce soit visuel, on va tracer la courbe de l'agitateur sur la PREMIÈRE cuve
print("\n📈 Génération du graphique pour le Batch #1...")
# Recréation rapide du Batch_ID pour isoler une seule cuve
df['VRAI_Batch_ID'] = (df['Time (h)'] < df['Time (h)'].shift(1)).cumsum() + 1
df_batch_1 = df[df['VRAI_Batch_ID'] == 1]

# Configuration du style du graphique
plt.style.use('dark_background')
plt.figure(figsize=(10, 5))

# Tracé du RPM en fonction du temps
plt.plot(df_batch_1['Time (h)'], df_batch_1[colonne_agitateur], color='#3498db', linewidth=2.5)

plt.title("Profil de l'Agitation (RPM) - Batch 1", fontsize=14, fontweight='bold')
plt.xlabel("Temps de Fermentation (Heures)", fontsize=12)
plt.ylabel("Vitesse de rotation (RPM)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.3)

plt.tight_layout()
plt.show()