#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYNPUT_BACKEND=dummy
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
mkdir -p outputs/OpenDrawer/dp_pretrained_human_left/logs
python -u scripts/preflight_h50_native_robocasa.py
python -u scripts/finetune_h50_native_dp.py --preflight
python -u scripts/finetune_h50_native_dp.py 2>&1 | tee outputs/OpenDrawer/dp_pretrained_human_left/logs/train_seed_0.log
python -u scripts/evaluate_h50_native_dp_three.py 2>&1 | tee outputs/OpenDrawer/dp_pretrained_human_left/logs/eval_seed_3_4_5.log
