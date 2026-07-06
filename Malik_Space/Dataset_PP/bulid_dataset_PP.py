import pandas as pd

print("⏳ Étape 1 : Création du Dataset Multi-Batch (8 Expériences)...")

fichier_brut = "Dataset.xlsx"          
fichier_etat = "dataset_online.xlsx"   

colonnes_permittivite = 13 # On sait qu'il y a 13 fréquences (colonnes 9 à 21 en général)
colonnes_physiques = ['Stirrer, RPM', 'Conductivity, mS/cm', 'Volume_V', 'Phase_Sh', 'Fs_in']
colonnes_bio = ['Biomasse_X_in', 'Proteine_P_in', 'Biomasse_X_target', 'Proteine_P_target']

tous_les_lots = []

# Boucle sur les 8 expériences
for i in range(1, 9):
    sheet_name = f'Exp {i}'
    print(f"🔄 Traitement de l'{sheet_name}...")
    
    try:
        # Chargement Capteurs
        df_capteurs = pd.read_excel(fichier_brut, sheet_name=sheet_name)
        df_capteurs.columns = df_capteurs.columns.str.replace('\n', ' ').str.strip()
        
        # Chargement Cibles (Offline)
        onglet_etat = str(i)
        df_etats = pd.read_excel(fichier_etat, sheet_name=onglet_etat, header=1)
        df_etats.columns = [
            'Fs_in', 'Biomasse_X_in', 'Proteine_P_in', 'Volume_V', 'Phase_Sh', 
            'Biomasse_X_target', 'Proteine_P_target', 'Volumetric_flow_F'
        ]
        
        # Fusion et alignement de la longueur
        min_length = min(len(df_capteurs), len(df_etats))
        df_capteurs = df_capteurs.iloc[:min_length].reset_index(drop=True)
        df_etats = df_etats.iloc[:min_length].reset_index(drop=True)
        
        df_fusion = pd.concat([df_capteurs, df_etats], axis=1)
        
        # Nettoyage
        df_fusion = df_fusion.astype(str).replace(',', '.', regex=True)
        df_fusion = df_fusion.apply(pd.to_numeric, errors='coerce')
        df_fusion = df_fusion.ffill().bfill()
        
        # Extraction des 13 fréquences de permittivité (ajuste les index si ton CSV brut change)
        cols_permittivite = list(df_fusion.columns)[9:22]
        
        # Sélection finale
        colonnes_finales = cols_permittivite + colonnes_physiques + colonnes_bio
        df_clean = df_fusion[colonnes_finales].copy()
        
        # Ajout du Batch ID (CRUCIAL POUR LE LSTM)
        df_clean['Batch_ID'] = i
        
        tous_les_lots.append(df_clean)
        
    except Exception as e:
        print(f"⚠️ Erreur sur {sheet_name} : {e}")

# Concaténation de tous les lots
df_final = pd.concat(tous_les_lots, ignore_index=True)

# Sauvegarde
nom_fichier_sortie = "dataset_pichia_multibatch.csv"
df_final.to_csv(nom_fichier_sortie, index=False)

print(f"✅ Succès ! Fichier '{nom_fichier_sortie}' généré avec {len(df_final)} lignes (8 lots).")