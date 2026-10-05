#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
eval_python="${EVAL_PYTHON:-/home/dabozi/miniconda3/envs/synutility-sim/bin/python}"
run_name="${1:-diffusion_seed_3_4_5}"
checkpoint="${2:-$project_root/outputs/OpenDrawer/diffusion_human_left_3cam/seed_0/checkpoints/last/pretrained_model}"
if [[ ! "$run_name" =~ ^[a-zA-Z0-9_]+$ ]]; then
    echo "Invalid run name: $run_name" >&2
    exit 2
fi
eval_root="$project_root/outputs/OpenDrawer/diffusion_human_left_3cam/seed_0/robocasa_eval"
if [[ -e "$eval_root/$run_name" || -e "$eval_root/${run_name}_console.log" ]]; then
    echo "Preserving existing evaluation; provide a new run name." >&2
    exit 2
fi
mkdir -p "$eval_root"
MUJOCO_GL=egl "$eval_python" -u scripts/evaluate_open_drawer_h50_diffusion_robocasa.py \
    --checkpoint "$checkpoint" --require-training-step 10000 \
    --episodes 3 --seeds 3 4 5 --run-name "$run_name" --output-root "$eval_root" \
    2>&1 | tee "$eval_root/${run_name}_console.log"
