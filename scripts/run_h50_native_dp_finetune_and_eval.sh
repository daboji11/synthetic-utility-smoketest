#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYNPUT_BACKEND=dummy
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
BATCH_SIZE="${BATCH_SIZE:-32}"
RUN_DIR="${RUN_DIR:-outputs/OpenDrawer/dp_pretrained_human_left/seed_0_batch${BATCH_SIZE}}"
LOG_DIR="${RUN_DIR}_logs"
mkdir -p "$LOG_DIR"
python -u scripts/preflight_h50_native_robocasa.py
python -u scripts/finetune_h50_native_dp.py --batch-size "$BATCH_SIZE" --output "$RUN_DIR" --preflight
python -u scripts/finetune_h50_native_dp.py --batch-size "$BATCH_SIZE" --output "$RUN_DIR" 2>&1 | tee "$LOG_DIR/train_seed_0.log"
python -u scripts/evaluate_h50_native_dp_three.py --checkpoint "$RUN_DIR/checkpoints/last.ckpt" \
  --output "$RUN_DIR/robocasa_eval/seed_3_4_5" 2>&1 | tee "$LOG_DIR/eval_seed_3_4_5.log"
