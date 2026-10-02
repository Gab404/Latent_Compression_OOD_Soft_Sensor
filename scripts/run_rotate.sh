#!/bin/bash
# ==============================================================================
# Expérience de rotation du latent (post-scaling), design apparié
#
# Pour chaque seed : identity, control, targeted, full.
# Même seed => même init du LSTM et même ordre des batchs dans les 4 modes.
# La seule différence entre modes est la transformation appliquée au latent.
# La baseline est "identity", PAS le sweep d'origine.
# ==============================================================================

OUT_DIR="./outputs/full_rotate_experiments_v2"
mkdir -p "$OUT_DIR"

ENC_DIR=/home/gabriel/ReaKt/outputs/seed_sweep/beta_vae_pi_lstm/p2/encoders
SEEDS=(42 123 456 789 1024)
MODES=(identity control targeted full)
Q_SEED=100   # même matrice Q pour tous les seeds (mode full)

for SEED in "${SEEDS[@]}"; do
    for MODE in "${MODES[@]}"; do
        LOG="$OUT_DIR/${MODE}_s${SEED}.log"
        echo ">>> seed=$SEED  mode=$MODE"
        PYTHONPATH=. python train.py \
            --model beta_vae_pi_lstm --regime-ablation --seed "$SEED" \
            --raman-ckpt   "$ENC_DIR/beta_vae_s${SEED}.pt" \
            --raman-scaler "$ENC_DIR/beta_vae_scaler_s${SEED}.joblib" \
            --rotate-mode "$MODE" --rotate-seed "$Q_SEED" \
            --no-plots \
            --ckpt "$OUT_DIR/${MODE}_s${SEED}.pt" > "$LOG" 2>&1

        # Affiche les lignes utiles (log complet dans $LOG)
        grep -E "\[ROTATION\]|ÉCHEC" "$LOG" | sort -u
        grep -E "(R2|Fault)\s+\[Penicillin\]" "$LOG" | tail -2
    done
done

echo -e "\nLogs complets : $OUT_DIR/*.log"
