#!/usr/bin/env bash
# Paired evaluation of the saved 2,000-update human300 + 28-human LoRA checkpoint.
set -euo pipefail

ROOT=/workspace/research/synthetic-utility-smoketest
PY=/workspace/research/openpi-robocasa/.venv/bin/python
CHECKPOINT="$ROOT/outputs/OpenDrawer/pi05_h50_left_lora/human300_h28_4k_seed0/2000"
OUTPUT="$ROOT/results/robocasa_eval/human300_h28_2k_seed0_20"
LOG=/workspace/pi05_human300_h28_2k_eval20.log

cd "$ROOT"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYNPUT_BACKEND=dummy
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
test -f "$CHECKPOINT/params/_METADATA"
test ! -e "$OUTPUT"
echo "EVAL_START checkpoint=$CHECKPOINT seeds=20 horizon=750 replan_steps=5"
"$PY" -u scripts/evaluate_pi05_open_drawer.py --lora \
  --checkpoint "$CHECKPOINT" --episodes 20 --horizon 750 --replan-steps 5 \
  --output "$OUTPUT" 2>&1 | tee "$LOG"
"$PY" - "$OUTPUT/summary.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
assert report["episodes"] == 20
print(f"HUMAN300_H28_2K_RESULT {report['successes']}/20", flush=True)
PY
echo "EVAL_COMPLETE output=$OUTPUT log=$LOG"
