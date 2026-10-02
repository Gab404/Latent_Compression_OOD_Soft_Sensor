#!/bin/bash
# Étape 4 : même budget d'entraînement (200 époques, sans early stopping)
# pour identity et full. Isole l'effet "durée d'entraînement".
OUT_DIR="./outputs/full_rotate_experiments_fixed_epochs"
mkdir -p "$OUT_DIR"
ENC_DIR=/home/gabriel/ReaKt/outputs/seed_sweep/beta_vae_pi_lstm/p2/encoders
Q_SEED=100

for SEED in 42 123 456 789 1024; do
    for MODE in identity full; do
        LOG="$OUT_DIR/${MODE}_s${SEED}.log"
        echo ">>> seed=$SEED  mode=$MODE  (200 époques, no ES)"
        PYTHONPATH=. python train.py \
            --model beta_vae_pi_lstm --regime-ablation --seed "$SEED" \
            --raman-ckpt   "$ENC_DIR/beta_vae_s${SEED}.pt" \
            --raman-scaler "$ENC_DIR/beta_vae_scaler_s${SEED}.joblib" \
            --rotate-mode "$MODE" --rotate-seed "$Q_SEED" \
            --epochs 200 --no-early-stopping \
            --no-plots \
            --ckpt "$OUT_DIR/${MODE}_s${SEED}.pt" > "$LOG" 2>&1
        grep -E "\[ROTATION\] OK|ÉCHEC" "$LOG" | sort -u
        grep -E "(R2|Fault)\s+\[Penicillin\]" "$LOG" | tail -2
    done
done
