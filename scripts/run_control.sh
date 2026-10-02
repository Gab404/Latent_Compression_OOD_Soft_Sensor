#!/bin/bash
# Étape 1 : relancer uniquement le mode control (5 runs)
OUT_DIR="./outputs/full_rotate_experiments_v2"
ENC_DIR=/home/gabriel/ReaKt/outputs/seed_sweep/beta_vae_pi_lstm/p2/encoders
Q_SEED=100

for SEED in 42 123 456 789 1024; do
    LOG="$OUT_DIR/control_s${SEED}.log"
    echo ">>> seed=$SEED  mode=control"
    PYTHONPATH=. python train.py \
        --model beta_vae_pi_lstm --regime-ablation --seed "$SEED" \
        --raman-ckpt   "$ENC_DIR/beta_vae_s${SEED}.pt" \
        --raman-scaler "$ENC_DIR/beta_vae_scaler_s${SEED}.joblib" \
        --rotate-mode control --rotate-seed "$Q_SEED" \
        --no-plots \
        --ckpt "$OUT_DIR/control_s${SEED}.pt" > "$LOG" 2>&1
    grep -E "\[ROTATION\] OK|ÉCHEC|Error|error" "$LOG" | sort -u
    grep -E "(R2|Fault)\s+\[Penicillin\]" "$LOG" | tail -2
done
