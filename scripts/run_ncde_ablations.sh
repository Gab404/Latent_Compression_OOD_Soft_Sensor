#!/usr/bin/env bash
# run_ncde_ablations.sh
# =====================
# Sequential Neural CDE ablation study.
# Waits for the default run (PID in outputs/neural_cde/run.pid) to finish
# before launching ablations one by one to avoid GPU OOM.
#
# Usage:
#   bash scripts/run_ncde_ablations.sh
#
# Ablations:
#   1. hidden_dim=16  (smaller, faster)
#   2. hidden_dim=64  (richer dynamics)
#   3. hidden_dim=32 + dense-loss  (train on all time steps, not just sparse pts)
#
# Each run:
#   • saves checkpoint + summary.json to outputs/neural_cde_<tag>/
#   • evaluates on Fault/R4 batches
#   • includes --eval-r1 to also get R1 RMSE for the scatter plot
#
# After all ablations, run benchmark_summary.py to update the comparison table.
# ─────────────────────────────────────────────────────────────────────────────

set -e

PYTHON=".venv/bin/python"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$ROOT_DIR/outputs/neural_cde/run.pid"

echo "══ Neural CDE Ablation Runner ════════════════════════════════════════"
echo "  Root: $ROOT_DIR"

# ── Wait for default run to finish ───────────────────────────────────────────
if [ -f "$PID_FILE" ]; then
    MAIN_PID=$(cat "$PID_FILE")
    if ps -p "$MAIN_PID" > /dev/null 2>&1; then
        echo ""
        echo "  Waiting for main CDE run (PID=$MAIN_PID) to finish ..."
        while ps -p "$MAIN_PID" > /dev/null 2>&1; do
            sleep 30
            LAST_LINE=$(tail -1 "$ROOT_DIR/outputs/neural_cde/run.log" 2>/dev/null || echo "")
            echo "  [$(date '+%H:%M:%S')] $LAST_LINE"
        done
        echo "  Main run finished."
    else
        echo "  Main run already finished (PID=$MAIN_PID not running)."
    fi
else
    echo "  No PID file found at $PID_FILE — assuming main run is done."
fi

# ── Helper: run one ablation ──────────────────────────────────────────────────
run_ablation() {
    local TAG="$1"
    shift
    local OUT="$ROOT_DIR/outputs/neural_cde_${TAG}"
    local LOG="$ROOT_DIR/outputs/neural_cde_${TAG}.log"

    echo ""
    echo "── Ablation: $TAG ────────────────────────────────────────────────"
    echo "   Out: $OUT"
    echo "   Log: $LOG"
    echo "   Args: $*"

    mkdir -p "$OUT"
    PYTHONPATH="$ROOT_DIR" CUDA_VISIBLE_DEVICES=0 \
        "$ROOT_DIR/$PYTHON" -u scripts/train_neural_cde.py \
        --out-dir "$OUT" \
        --eval-r1 \
        "$@" \
        2>&1 | tee "$LOG"

    echo "   Done: $TAG"
}

# ── Ablation 1: hidden_dim=16 ─────────────────────────────────────────────────
run_ablation "h16_rk4" \
    --hidden-dim 16 \
    --solver rk4

# ── Ablation 2: hidden_dim=64 ─────────────────────────────────────────────────
run_ablation "h64_rk4" \
    --hidden-dim 64 \
    --solver rk4

# ── Ablation 3: h=32 + dense loss ────────────────────────────────────────────
run_ablation "h32_dense" \
    --hidden-dim 32 \
    --solver rk4 \
    --dense-loss

# ── Update benchmark summary ─────────────────────────────────────────────────
echo ""
echo "── Updating benchmark summary ..."
PYTHONPATH="$ROOT_DIR" "$ROOT_DIR/$PYTHON" scripts/benchmark_summary.py
echo ""
echo "══ All ablations complete. ══════════════════════════════════════════"
