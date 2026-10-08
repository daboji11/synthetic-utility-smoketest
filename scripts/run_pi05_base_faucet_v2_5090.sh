#!/usr/bin/env bash
# Manual launch only. Probe and full training are separate processes / fresh bases.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${OPENPI_ROOT:-$ROOT/../openpi-robocasa}/.venv/bin/python"
RUN_ID="${FAUCET_V2_RUN_ID:-human54_lr1e5_7000_seed0}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]] || { echo 'Invalid run ID'; exit 2; }
LOGS="$ROOT/logs/pi05_base_faucet_v2/$RUN_ID"
TRAIN="$ROOT/outputs/TurnOnSinkFaucet/pi05_base_faucet_state_range_v2/$RUN_ID"
RESULTS="$ROOT/results/robocasa_eval/pi05_base_faucet_v2/$RUN_ID"
cd "$ROOT"
mkdir -p "$LOGS"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYNPUT_BACKEND=dummy
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/../../.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$XDG_CACHE_HOME/openpi}"

stage() {
    local name="$1"
    printf '%s %s\n' "$(date -Is)" "$name" > "$LOGS/status.txt"
    case "$name" in
        preflight) "$PY" scripts/pi05_base_faucet_v2_config.py --run-id "$RUN_ID" ;;
        probe) "$PY" scripts/finetune_pi05_base_faucet_v2.py --train --probe --run-id "$RUN_ID" ;;
        train) "$PY" scripts/finetune_pi05_base_faucet_v2.py --train --run-id "$RUN_ID" ;;
        eval-1000|eval-3000|eval-5000|eval-7000)
            local step="${name#eval-}"
            "$PY" scripts/evaluate_pi05_base_faucet_v2.py --checkpoint "$TRAIN/$step" \
                --run-id "$RUN_ID" --output "$RESULTS/$name" ;;
        *) echo "Unknown stage: $name"; return 2 ;;
    esac 2>&1 | tee "$LOGS/$name.log"
}

mode="${1:-help}"
if [[ "$mode" == status ]]; then
    cat "$LOGS/status.txt" 2>/dev/null || true
    tail -n 8 "$LOGS/launcher.log" 2>/dev/null || true
    exit 0
fi
if [[ "$mode" == help ]]; then
    echo 'Usage: bash scripts/run_pi05_base_faucet_v2_5090.sh {all|preflight|probe|train|eval-1000|eval-3000|eval-5000|eval-7000|status}'
    exit 0
fi
# Shared lock covers all v2 invocations, including separate stages/run IDs.
exec 9> "$ROOT/logs/pi05_base_faucet_v2/active.lock"
flock -n 9 || { echo 'Another faucet v2 stage is running'; exit 1; }
trap 'rc=$?; printf "%s EXIT=%s\n" "$(date -Is)" "$rc" >> "$LOGS/status.txt"' EXIT
if [[ "$mode" == all ]]; then
    stage preflight
    stage probe
    stage train
    stage eval-7000
    printf '%s COMPLETE\n' "$(date -Is)" > "$LOGS/status.txt"
else
    stage "$mode"
fi
