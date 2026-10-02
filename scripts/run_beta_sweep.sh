#!/bin/bash

# Liste des valeurs de beta à tester
# BETAS=(0.0005 0.001 0.002 0.01 0.02)
BETAS=(0.05)

for BETA in "${BETAS[@]}"; do
    echo "=========================================================="
    echo " Lancement de beta_vae_pi_lstm avec beta = $BETA"
    echo "=========================================================="
    
    # On utilise --out-dir pour ne pas écraser les résultats du run précédent
    python run_seeds.py \
        --model beta_vae_pi_lstm \
        --max-beta "$BETA" \
        --out-dir "./outputs/seed_sweep_beta/beta_${BETA}" \
        --n-seeds 5 \
        --protocol 2
        
    echo "Terminé pour beta = $BETA"
done

echo "Toutes les expériences sont terminées."