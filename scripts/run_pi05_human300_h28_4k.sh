#!/usr/bin/env bash
# User-authorized 4090 run: original human300 + 28 human demos, then paired 20-seed rollout.
set -euo pipefail

ROOT=/workspace/research/synthetic-utility-smoketest
PY=/workspace/research/openpi-robocasa/.venv/bin/python
TRAIN_DIR="$ROOT/outputs/OpenDrawer/pi05_h50_left_lora/human300_h28_4k_seed0"
EVAL_DIR="$ROOT/results/robocasa_eval/human300_h28_4k_seed0_20"
TRAIN_LOG=/workspace/pi05_human300_h28_4k_train.log
EVAL_LOG=/workspace/pi05_human300_h28_4k_eval20.log

cd "$ROOT"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYNPUT_BACKEND=dummy
export HF_HUB_ENABLE_HF_TRANSFER=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

test -x "$PY"
test -f data/raw/models/pi05_pretrain_human300/75000/source_manifest.json
test -f data/processed/OpenDrawer/h50_human_left_native_dp_20fps_256/meta/info.json
test ! -e "$TRAIN_DIR"
test ! -e "$EVAL_DIR"

echo "RUN_START human300 + 28 human LoRA, 4000 updates; then 20 matched-seed rollouts"
echo "SOURCE_CHECKPOINT=$ROOT/data/raw/models/pi05_pretrain_human300/75000"
echo "TRAIN_OUTPUT=$TRAIN_DIR"
echo "EVAL_OUTPUT=$EVAL_DIR"

"$PY" -u scripts/finetune_pi05_h50_lora.py \
  --train --batch-size 4 --steps 4000 --lr 1e-5 --seed 0 \
  --exp-name human300_h28_4k_seed0 --num-workers 4 \
  --lr-schedule-steps 10000 --save-interval 2000 --keep-period 2000 \
  2>&1 | tee "$TRAIN_LOG"

grep -q '^PI05_LORA_COMPLETE$' "$TRAIN_LOG"
test -f "$TRAIN_DIR/4000/params/_METADATA"
echo "TRAINING_VERIFIED checkpoint=$TRAIN_DIR/4000"

"$PY" -u scripts/evaluate_pi05_open_drawer.py --lora \
  --checkpoint "$TRAIN_DIR/4000" \
  --episodes 20 --horizon 750 --replan-steps 5 \
  --output "$EVAL_DIR" \
  2>&1 | tee "$EVAL_LOG"

"$PY" - "$EVAL_DIR/summary.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
assert report["episodes"] == 20
print(f"HUMAN300_H28_4K_RESULT {report['successes']}/20", flush=True)
PY

echo "RUN_COMPLETE train_log=$TRAIN_LOG eval_log=$EVAL_LOG"
