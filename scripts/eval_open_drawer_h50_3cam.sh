#!/usr/bin/env bash
# Run manually inside WSL Ubuntu, after the three-camera training has finished.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
eval_python=/home/dabozi/miniconda3/envs/synutility-sim/bin/python
run_name="${1:-three_camera_seed_3_4_5}"
if [[ ! "$run_name" =~ ^[a-zA-Z0-9_]+$ ]]; then
    echo "Invalid run name: $run_name" >&2
    exit 2
fi
eval_root="$project_root/outputs/OpenDrawer/human_left_only_3cam/seed_0/robocasa_eval"
if [[ -e "$eval_root/$run_name" || -e "$eval_root/${run_name}_console.log" ]]; then
    echo "Preserving existing output. Supply a new run name as the first argument." >&2
    exit 2
fi
mkdir -p "$eval_root"
MUJOCO_GL=egl "$eval_python" -u scripts/evaluate_open_drawer_h50_act_robocasa.py \
    --multicamera --episodes 3 --start-seed 3 --seeds 3 4 5 \
    --replan-every 100 --require-training-step 10000 --save-video \
    --run-name "$run_name" 2>&1 | tee "$eval_root/${run_name}_console.log"
