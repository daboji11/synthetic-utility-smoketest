#!/usr/bin/env bash
# Manually start once; independent train -> rollout groups execute sequentially.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=/workspace/research/openpi-robocasa/.venv/bin/python
DATA_ROOT="$ROOT/data/processed/OpenDrawer/synthetic_8group_20fps_9a3dd767"
RUN_ROOT=/workspace/pi05_synthetic_3groups_9a3dd767
MODEL="$ROOT/data/raw/models/pi05_pretrain_human300/75000"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYNPUT_BACKEND=dummy
export HF_HUB_ENABLE_HF_TRANSFER=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -x "$PY" ] || { echo "Missing Pi0.5 Python" >&2; exit 1; }
[ -f "$MODEL/source_manifest.json" ] || { echo "Missing model weights" >&2; exit 1; }
[ -d "$DATA_ROOT" ] || { echo "Missing selected datasets" >&2; exit 1; }
FREE_BYTES="$(df -B1 --output=avail /workspace | tail -n 1)"
[ "$FREE_BYTES" -ge 40000000000 ] || { echo "Less than 40 GB free on /workspace" >&2; exit 1; }
[ ! -e "$RUN_ROOT" ] || { echo "Preserving existing run: $RUN_ROOT" >&2; exit 1; }
mkdir "$RUN_ROOT"
printf 'PRECHECK\n' >"$RUN_ROOT/current_stage.txt"
trap 'code=$?; echo "FAILED exit=$code stage=$(cat "$RUN_ROOT/current_stage.txt")" | tee "$RUN_ROOT/FAILED.txt"; exit "$code"' ERR
"$PY" -u scripts/preflight_pi05_synthetic_3groups.py --dataset-root "$DATA_ROOT" --check-weights | tee "$RUN_ROOT/preflight.log"
cp records/opendrawer_synthetic_8group_selection_exploratory_fee2a10f.json "$RUN_ROOT/selection.json"
cp records/opendrawer_synthetic_8group_selection_exploratory_fee2a10f.csv "$RUN_ROOT/selection.csv"
for group in redundant diverse random; do
  group_dir="$RUN_ROOT/$group"
  mkdir "$group_dir"
  exp_name="synthetic_$group"_9a3dd767_seed0
  train_root="$ROOT/outputs/OpenDrawer/pi05_synthetic_8_lora/$exp_name"
  checkpoint="$train_root/4000"
  printf 'TRAIN %s\n' "$group" | tee "$RUN_ROOT/current_stage.txt"
  "$PY" -u scripts/finetune_pi05_synthetic_8_lora.py \
    --train --dataset "$DATA_ROOT/$group" --group "$group" \
    --exp-name "$exp_name" --batch-size 4 --steps 4000 \
    --lr 1e-5 --seed 0 --num-workers 4 \
    2>&1 | tee "$group_dir/train.log"
  [ -f "$train_root/training_summary.json" ]
  [ -d "$checkpoint/params" ]
  cp "$train_root/training_summary.json" "$group_dir/training_summary.json"
  cp "$train_root/actual_config.json" "$group_dir/actual_config.json"
  printf 'ROLLOUT %s\n' "$group" | tee "$RUN_ROOT/current_stage.txt"
  "$PY" -u scripts/evaluate_pi05_open_drawer.py \
    --lora --checkpoint "$train_root/last" --episodes 20 \
    --horizon 750 --replan-steps 5 --output "$group_dir/rollout" \
    2>&1 | tee "$group_dir/rollout.log"
  "$PY" - "$group_dir" <<'PY'
import csv,json,sys
from pathlib import Path
root=Path(sys.argv[1])
report=json.loads((root/"rollout/summary.json").read_text())
expected=[3,4,5,8,10,13,16,18,20,21,22,23,26,27,28,31,32,33,34,35]
assert report["episodes"]==20 and report["seeds"]==expected
assert len(report["episode_stats"])==20
assert report["successes"]==sum(bool(row["success"]) for row in report["episode_stats"])
fields=["seed","success","steps","reward","replans","clipped_count","wall_clock_s","video"]
with (root/"rollout_results.csv").open("w",newline="") as file:
    writer=csv.DictWriter(file,fieldnames=fields)
    writer.writeheader()
    writer.writerows({key:row[key] for key in fields} for row in report["episode_stats"])
(root/"result.json").write_text(json.dumps({
    "group":root.name,"episodes":20,"successes":report["successes"],
    "success_rate":report["success_rate"],"checkpoint":report["checkpoint"],
    "seed_list":expected,"output":str(root/"rollout")
},indent=2)+"\n")
print("GROUP_COMPLETE",root.name,report["successes"],"/20",flush=True)
PY
done
printf 'ALL_DONE\n' | tee "$RUN_ROOT/current_stage.txt"
"$PY" - "$RUN_ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
rows=[json.loads((root/group/"result.json").read_text()) for group in ("redundant","diverse","random")]
(root/"three_group_results.json").write_text(json.dumps(rows,indent=2)+"\n")
print("ALL_THREE_GROUPS_COMPLETE",[(x["group"],x["successes"]) for x in rows],flush=True)
PY
