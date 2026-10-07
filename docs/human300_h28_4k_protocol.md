# pi05 human300 + 28 H50 human demos: 4,000-update rerun

## Why this run

The archived human300 + 28-human LoRA run used 10,000 optimizer updates, batch 4,
peak LR 1e-5, seed 0, and the official `pi05_pretrain_human300/75000` weights.
Its paired OpenDrawer result was 8/20 versus untouched human300 at 10/20.
That difference alone does not establish overfitting. This run tests an earlier
stop on the same training protocol.

## Controlled settings

- Initialize afresh from `data/raw/models/pi05_pretrain_human300/75000/params`.
- Train only `data/processed/OpenDrawer/h50_human_left_native_dp_20fps_256`:
  28 original human episodes, 5,916 frames, 20 Hz, no synthetic data.
- LoRA only; all non-adapter weights, including the vision encoder, remain frozen.
- Batch 4, peak LR 1e-5, seed 0, 4,000 optimizer updates.
- Preserve the **original 10,000-update LR curve** (500-step warmup and cosine
  decay scheduled over 10,000 updates), then stop at update 4,000. This isolates
  stopping time better than compressing the cosine schedule to 4,000 updates.
- Save at updates 2,000 and 4,000 under a new experiment name. Keep both.
- Train and evaluate on the same pinned OpenPI/RoboCasa commits and H50 native
  normalization/camera/action/task interface as the 10,000-update run.

## GPU host: preparation (no optimizer updates)

```bash
cd /workspace/research/synthetic-utility-smoketest
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYNPUT_BACKEND=dummy
export HF_HUB_ENABLE_HF_TRANSFER=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/workspace/research/openpi-robocasa/.venv/bin/python
"$PY" -u scripts/download_pi05_robocasa.py
"$PY" -u scripts/preflight_pi05_h50.py --batch-size 4
```

## User starts training and sees every log line

Run in a persistent `tmux` shell if the browser terminal might disconnect.
The training process itself is foreground, so `TRAIN_STEP` and checkpoint logs
remain visible. `tee` saves the same output on the Pod.

```bash
cd /workspace/research/synthetic-utility-smoketest
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYNPUT_BACKEND=dummy
export HF_HUB_ENABLE_HF_TRANSFER=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
set -o pipefail
PY=/workspace/research/openpi-robocasa/.venv/bin/python
"$PY" -u scripts/finetune_pi05_h50_lora.py \
  --train --batch-size 4 --steps 4000 --lr 1e-5 --seed 0 \
  --exp-name human300_h28_4k_seed0 --num-workers 4 \
  --lr-schedule-steps 10000 --save-interval 2000 --keep-period 2000 \
  2>&1 | tee /workspace/pi05_human300_h28_4k_train.log
```

Expected training output:
`outputs/OpenDrawer/pi05_h50_left_lora/human300_h28_4k_seed0/`.
Confirm `PI05_LORA_COMPLETE` and `CHECKPOINT_SAVED step=4000` before rollout.
The checkpoint directory is not the 10,000-update checkpoint; do not pass that
older path to evaluation.

## User starts paired evaluation after training

```bash
cd /workspace/research/synthetic-utility-smoketest
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYNPUT_BACKEND=dummy
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
set -o pipefail
PY=/workspace/research/openpi-robocasa/.venv/bin/python
"$PY" -u scripts/evaluate_pi05_open_drawer.py --lora \
  --checkpoint outputs/OpenDrawer/pi05_h50_left_lora/human300_h28_4k_seed0/4000 \
  --episodes 20 --horizon 750 --replan-steps 5 \
  --output results/robocasa_eval/human300_h28_4k_seed0_20 \
  2>&1 | tee /workspace/pi05_human300_h28_4k_eval20.log
```

The evaluator's exact 20 seeds are `[3,4,5,8,10,13,16,18,20,21,22,23,26,27,28,31,32,33,34,35]`.
Compare per-seed success and trajectories with the archived untouched human300
10/20 and 10,000-update LoRA 8/20. The reused seeds are development seeds,
so a small difference is exploratory; confirm any trend on fresh seeds later.

Back up the new checkpoints and rollout records before stopping a Pod without
persistent storage.
