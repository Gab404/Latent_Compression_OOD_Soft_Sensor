#!/bin/bash
# Étape 2 : 5 encodeurs x 3 seeds LSTM x 3 modes = 45 runs (15 paires par mode)
# Nécessite l'option --lstm-seed dans train.py
OUT_DIR="./outputs/full_rotate_experiments_v3"
mkdir -p "$OUT_DIR"
ENC_DIR=/home/gabriel/ReaKt/outputs/seed_sweep/beta_vae_pi_lstm/p2/encoders
Q_SEED=100
SEEDS=(42 123 456 789 1024)
LSTM_SEEDS=(11 22 33)
MODES=(identity control targeted full)

for SEED in "${SEEDS[@]}"; do
  for LSEED in "${LSTM_SEEDS[@]}"; do
    for MODE in "${MODES[@]}"; do
        TAG="${MODE}_s${SEED}_l${LSEED}"
        LOG="$OUT_DIR/${TAG}.log"
        echo ">>> seed=$SEED  lstm_seed=$LSEED  mode=$MODE"
        PYTHONPATH=. python train.py \
            --model beta_vae_pi_lstm --regime-ablation --seed "$SEED" \
            --lstm-seed "$LSEED" \
            --raman-ckpt   "$ENC_DIR/beta_vae_s${SEED}.pt" \
            --raman-scaler "$ENC_DIR/beta_vae_scaler_s${SEED}.joblib" \
            --rotate-mode "$MODE" --rotate-seed "$Q_SEED" \
            --no-plots \
            --ckpt "$OUT_DIR/${TAG}.pt" > "$LOG" 2>&1
        grep -E "\[ROTATION\] OK|\[LSTM-SEED\]|ÉCHEC" "$LOG" | sort -u
        grep -E "Fault\s+\[Penicillin\]" "$LOG" | tail -1
    done
  done
done
