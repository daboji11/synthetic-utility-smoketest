#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYNPUT_BACKEND=dummy
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
PYTHON_EXE="${PYTHON_EXE:-python}"
EVAL_DIR="${EVAL_DIR:-results/robocasa_eval/h50_dp_local_5_seeds_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$EVAL_DIR" || -e "${EVAL_DIR}.log" ]]; then
  echo "Preserving existing evaluation/log: $EVAL_DIR" >&2
  exit 1
fi
mkdir -p "$(dirname "$EVAL_DIR")"
"$PYTHON_EXE" -u scripts/evaluate_h50_native_dp_seeds.py \
  --config configs/native_dp/h50_left_local_eval.yaml \
  --output "$EVAL_DIR" 2>&1 | tee "${EVAL_DIR}.log"
