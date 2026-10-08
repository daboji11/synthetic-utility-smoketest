#!/usr/bin/env bash
# Run one explicit Faucet stage on the rented RTX 5090. Nothing launches on import.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${OPENPI_ROOT:-$ROOT/../openpi-robocasa}/.venv/bin/python"
RUN="$ROOT/results/robocasa_eval/pi05_base_faucet_smoketest"
BASE="$ROOT/data/raw/models/pi05_base_faucet_interface"
TRAIN="$ROOT/outputs/TurnOnSinkFaucet/pi05_base_faucet_lora/seed_0"
LOGS="$ROOT/logs/pi05_base_faucet"
mkdir -p "$LOGS"

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYNPUT_BACKEND=dummy
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/../../.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$XDG_CACHE_HOME/uv}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$XDG_CACHE_HOME/openpi}"

stage="${1:-}"
case "$stage" in
  download-base)
    "$PY" -u "$ROOT/scripts/download_pi05_base_faucet.py" \
      2>&1 | tee "$LOGS/download_base.log"
    ;;
  preflight)
    "$PY" -u "$ROOT/scripts/preflight_pi05_base_faucet.py" --check-base \
      2>&1 | tee "$LOGS/preflight.log"
    ;;
  baseline)
    "$PY" -u "$ROOT/scripts/evaluate_pi05_base_faucet.py" \
      --checkpoint "$BASE" --output "$RUN/base" \
      2>&1 | tee "$LOGS/baseline_20.log"
    ;;
  train)
    "$PY" -u "$ROOT/scripts/finetune_pi05_base_faucet_lora.py" \
      --train --batch-size 8 --steps 3000 --lr 5e-6 \
      --seed 0 --num-workers 4 \
      2>&1 | tee "$LOGS/train_3000.log"
    ;;
  eval-500|eval-1000|eval-2000|eval-3000)
    step="${stage#eval-}"
    "$PY" -u "$ROOT/scripts/evaluate_pi05_base_faucet.py" \
      --checkpoint "$TRAIN/$step" --lora --output "$RUN/lora_$step" \
      2>&1 | tee "$LOGS/lora_${step}_20.log"
    ;;
  compare)
    "$PY" -u "$ROOT/scripts/compare_pi05_base_faucet.py" \
      2>&1 | tee "$LOGS/compare.log"
    ;;
  *)
    echo "Usage: $0 {download-base|preflight|baseline|train|eval-500|eval-1000|eval-2000|eval-3000|compare}" >&2
    exit 2
    ;;
esac
