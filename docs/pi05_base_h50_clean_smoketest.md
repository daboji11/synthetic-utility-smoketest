# Clean pi05_base → 28 H50 human OpenDrawer smoke test

The user starts each GPU command manually. No synthetic data or previous LoRA checkpoint is used.
The scripts refuse to run on a non-4090 GPU and preserve an existing output directory.

## Exact base and data

- Official pinned OpenPI RoboCasa config: `pi05-og_robocasa_pickplacetoastertocounter` at commit `5a6beda9ff99da30b4e1b59320f6a32971d7c397`.
- That config's weight loader points to **`gs://openpi-assets/checkpoints/pi05_base/params`**. The URI is read from the config at runtime and checked. There is no standalone `pi05_base` training config in this fork.
- Downloaded parameters are linked under `data/raw/models/pi05_base_h50_interface/params`. This is a local interface package; the weights themselves remain untouched.
- Dataset: `data/processed/OpenDrawer/h50_human_left_native_dp_20fps_256`, exactly 28 original H50 human `OpenDrawer` / `open the left drawer` demos, 5916 frames at 20 Hz, 84 camera videos. The full data is in `D:\h50_pi05_handoff_data.tar`; Git does not include the videos/parquets.
- Both A and C use **the same** z-score stats derived from these 28 human demos. The original H50 raw action order is already the 12D model action order. The 16D state is reordered from `[EEF position, EEF quaternion, gripper, base position, base quaternion]` to `[EEF position, EEF quaternion, base position, base quaternion, gripper]`. This avoids the upstream generic stats converter's hardcoded order, which does not match this H50 dataset. The untouched base baseline therefore receives the human-demo statistics as an interface calibration, while its weights remain untouched. This limitation must be stated with results.
- Model input: left agent view, eye-in-hand, right agent view, each RGB 256² decoded and resized to 224² by the pinned OpenPI pipeline; 16D state; instruction `open the left drawer`. Model action horizon 50 and 12D action. Simulator control frequency and data FPS are both 20. Rollout uses horizon 750, 5 actions per replan and clipping to `[-1, 1]` before RoboCasa `convert_action`.
- Fixed seeds: `3,4,5,8,10,13,16,18,20,21,22,23,26,27,28,31,32,33,34,35`. Baseline and LoRA save the initial 16D state and hashes of all three initial camera images for each seed. Comparison checks the same ordered seeds, task and initial state within `1e-5`.

## LoRA plan

- Start from the verified base params path above, not `pi05_pretrain_human300/75000`.
- Train only LoRA matrices in PaliGemma and the action expert; freeze the vision backbone, all base weights and action/state heads. Adapter variants: `gemma_2b_lora` rank 16 alpha 16; `gemma_300m_lora` rank 32 alpha 32, attention and FFN.
- Batch 4, 2000 optimizer steps, peak learning rate `5e-6`, warmup 200 steps, cosine to `5e-7`, seed 0, no EMA, AdamW from pinned RoboCasa config. About `8000/5916 = 1.35` sampled data passes. Save at 1000 and 2000, retaining the earlier checkpoint.
- The previous H50-human300 LoRA used batch 4, 10000 steps and peak LR `1e-5`; that was a completed run, but its 8/20 did not beat human300 vanilla 10/20, so this run is deliberately shorter and lower LR.
- Stop with an explicit error if first-ten-step loss exceeds 100. This catches the zero-variance normalization failure seen in the interrupted synthetic run.

## Fresh RunPod setup and manual ignition

On a new 4090 Pod, put the repository under `/workspace/research/synthetic-utility-smoketest` and the 28-demo archive at `/workspace/h50_pi05_handoff_data.tar`. Git and the archive provide separate pieces; the archive is not in Git.

```bash
set -o pipefail
mkdir -p /workspace/research
cd /workspace/research
git clone git@github.com:daboji11/synthetic-utility-smoketest.git
cd synthetic-utility-smoketest
export PI05_SOURCE_ROOT=/workspace/research
export OPENPI_ROOT=/workspace/research/openpi-robocasa
bash scripts/setup_pi05_robocasa.sh
PY=/workspace/research/openpi-robocasa/.venv/bin/python
"$PY" -m robocasa.scripts.download_kitchen_assets --type tex tex_generative fixtures_lw objs_lw
mkdir -p data/processed/OpenDrawer
tar -xf /workspace/h50_pi05_handoff_data.tar -C data/processed/OpenDrawer
"$PY" -u scripts/preflight_pi05_base_h50.py
"$PY" -u scripts/download_pi05_base_h50.py 2>&1 | tee /workspace/pi05_base_download.log
"$PY" -u scripts/preflight_pi05_base_h50.py --check-base
```

The official base params download is large. Confirm free disk before starting. The preflight is CPU metadata validation; it does not evaluate or train.

### A. Untouched base: manually start exactly 20 rollouts

Use `tmux new -s pi05-base` if you want the visible log to survive browser disconnects. In that shell:

```bash
set -o pipefail
cd /workspace/research/synthetic-utility-smoketest
PY=/workspace/research/openpi-robocasa/.venv/bin/python
RUN="$PWD/results/robocasa_eval/pi05_base_h50_smoketest"
mkdir -p "$RUN"
"$PY" -u scripts/evaluate_pi05_base_open_drawer.py \
  --checkpoint "$PWD/data/raw/models/pi05_base_h50_interface" \
  --output "$RUN/base" 2>&1 | tee "$RUN/base.log"
```

The command prints `PROGRESS`, `EPISODE_STATS` and `FINAL_EVAL`; results are under `$RUN/base/`, including one video per seed, `episodes.json`, `initial_conditions.json`, `eval_config.json` and `summary.json`.

### B. 28 human demos LoRA: manually start after A completes

```bash
set -o pipefail
cd /workspace/research/synthetic-utility-smoketest
PY=/workspace/research/openpi-robocasa/.venv/bin/python
RUN="$PWD/results/robocasa_eval/pi05_base_h50_smoketest"
"$PY" -u scripts/finetune_pi05_base_h50_lora.py --train \
  --batch-size 4 --steps 2000 --lr 5e-6 --seed 0 --num-workers 4 \
  2>&1 | tee "$RUN/train.log"
```

The foreground log prints `TRAIN_STEP` every 10 steps and `CHECKPOINT_SAVED` at 1000 and 2000. Checkpoints and exact config go under `outputs/OpenDrawer/pi05_base_h50_left_lora/seed_0/`.

### C. Matched 20 rollouts: manually start after B completes

```bash
set -o pipefail
cd /workspace/research/synthetic-utility-smoketest
PY=/workspace/research/openpi-robocasa/.venv/bin/python
RUN="$PWD/results/robocasa_eval/pi05_base_h50_smoketest"
"$PY" -u scripts/evaluate_pi05_base_open_drawer.py \
  --lora --checkpoint "$PWD/outputs/OpenDrawer/pi05_base_h50_left_lora/seed_0/2000" \
  --output "$RUN/lora_2000" 2>&1 | tee "$RUN/lora_2000.log"
"$PY" -u scripts/compare_pi05_base_h50.py
```

The last command prints exactly `pi05_base: X/20` and `pi05_base + 28 human demos: Y/20`, and writes `paired_results.csv` plus `comparison.json`. If final performance is worse, evaluate the retained 1000-step checkpoint as a separate named exploratory run before considering synthetic experiments.
