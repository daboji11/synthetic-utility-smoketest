# RoboCasa π0.5: 28-human-left LoRA handoff

Decision date: 2026-10-05 (Australia/Sydney). ACT and DP are retired experiments; retain them.
Active target: **official RoboCasa π0.5 pretrain_human300 → strictly LoRA on 28 H50 human-left demos**.
Setup and preflight start no training or rollout.

## DP conclusion

Same 20 left-drawer seeds, native 20Hz, three cameras, 750-step horizon, native 8-action chunks:

| Model | Success |
|---|---:|
| Original RoboCasa DP pretrain_human300 | 2/20 (10%), seeds 13 and 21 |
| Full-parameter H50-28 fine-tune | 0/20 |

See records/dp_final_comparison.json, its paired CSV, and records/dp_finetune_diagnosis.json.
The completed fine-tune used **batch4**, 10,000 updates, peak LR1e-5; all three encoders trained.
Batch32 was a later proposed config, never used by this run. Failure cause is not proven.
Do not claim pretraining, 28 episodes, LoRA or a newer policy guarantees success.
DP tuning ends; retain source/checkpoints/videos.

## Git versus files to bring to school

Git contains code, pins, small statistics, this guide and docs/NEXT_SESSION_PROMPT.md.
**Git contains no dataset, weights, checkpoints or videos.**

Copy these files from home Windows D: to the Mac or external storage BEFORE leaving:

- D:\h50_pi05_handoff_data.tar (217,835,520 bytes, about208MiB)
- D:\h50_pi05_handoff_data.tar.sha256

The archive contains exactly28episodeParquets and84three-camera videos, native20Hz,5916frames.
Same fixed human base used by nativeDP; originals unchanged.
SHA256: records/pi05_data_transfer.json.
Historical DP checkpoint backup: D:\h50_dp_final_backup.tar, optional for further inspection.
Cloning Git alone will not transfer these files.

## Mac controls a Linux CUDA host

The Mac is the SSH client. Installation, training and evaluation run on a Linux GPU host.
If desired, clone this private repo on the Mac to read its context:

~~~bash
mkdir -p ~/research
cd ~/research
git clone git@github.com:daboji11/synthetic-utility-smoketest.git
cd synthetic-utility-smoketest
git pull --ff-only
~~~

Connect using the provider's actual SSH command; no credentials/host address are saved here.
On the GPU host, use persistent storage where available, for example /workspace/research:

~~~bash
mkdir -p /workspace/research
cd /workspace/research
git clone git@github.com:daboji11/synthetic-utility-smoketest.git
cd synthetic-utility-smoketest
git pull --ff-only
export PI05_SOURCE_ROOT=/workspace/research
export OPENPI_ROOT="$PI05_SOURCE_ROOT/openpi-robocasa"
bash scripts/setup_pi05_robocasa.sh
export PY="$OPENPI_ROOT/.venv/bin/python"
export MUJOCO_GL=egl PYNPUT_BACKEND=dummy HF_HUB_ENABLE_HF_TRANSFER=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
~~~

Setup uses separate Python3.11, official pinned uv.lock plus explicit RoboCasa runtime fixes.
The fork's LeRobot Git pin reports0.1.0 while both projects request0.3.3; we install0.3.3,
use its compatible Gymnasium/client versions, and omit unused gym-aloha (MuJoCo conflict).
RoboCasa explicitly requires NumPy2.2.5 and Tianshou0.4.10; compatible protobuf/Google client pins are recorded.
See requirements-pi05-runtime.txt and records/pi05_environment.json for precise versions and verification.
It verifies **JAX CUDA** and pip check; CUDA Torch alone is insufficient. The DP environment is untouched.
Final local checks:4π0.5interface tests and8native-DP/core regression tests passed.
The complete mixed historical suite cannot collect legacyACT/scratchDP tests in the nativeDP env because
those tests require a different LeRobot API; this is recorded, and no retired environment was modified.
requirements-pi05-linux-lock.txt is generated locally. No training/rollout is started.
Fresh hosts also need kitchen assets (simulator assets, not other tasks' demonstration datasets):

~~~bash
"$PY" -m robocasa.scripts.download_kitchen_assets --type tex tex_generative fixtures_lw objs_lw
~~~

Official upstream estimates LoRA >22.5GBVRAM, full fine-tuning >70GB; RoboCasa recommends80GB for training.
4090 24GB is a LoRA candidate, **not confirmed three-camera batch capacity**.
Measure memory and throughput before a long paid run.
Sources: https://github.com/Physical-Intelligence/openpi and https://github.com/robocasa-benchmark/openpi

## Transfer the fixed human base

Upload archive+checksum from the Mac with scp using the actual provider's host and SSH port.
Place both under /workspace/transfers on Linux, then:

~~~bash
cd /workspace/transfers
sha256sum -c h50_pi05_handoff_data.tar.sha256
cd /workspace/research/synthetic-utility-smoketest
mkdir -p data/processed/OpenDrawer
test ! -e data/processed/OpenDrawer/h50_human_left_native_dp_20fps_256 \
  && tar -xf /workspace/transfers/h50_pi05_handoff_data.tar -C data/processed/OpenDrawer
~~~

If the target exists, verify it without overwriting.
If the archive was not brought, locate it or explicitly prepare identical H50left28 using existing source scripts.
Do not substitute MG_30, right-drawer demos or synthetic data.

## Official checkpoint and preflight

Source: robocasa-benchmark/openpi@5a6beda9ff99da30b4e1b59320f6a32971d7c397.
HF: robocasa/robocasa365_checkpoints@c484448aba1a9b60a04c9b0ca117241518ea69f3.
Prefix: pi05_pretrain_human300/multitask_learning/75000.

~~~bash
"$PY" -u scripts/download_pi05_robocasa.py
"$PY" -u scripts/preflight_pi05_h50.py
~~~

Downloads **params+assets only:12,440,863,506bytes**, no optimizer state or other model.
Official release is **JAX Orbax**, not Torch safetensors. Keep all OCDBT files and normalization assets.
Do not reuse DP normalizer/EMA or general pi05_base weights.
Preflight checks real dataset reader/transforms and abstract trainable parameter tree.
It does not load full weights, compute loss, run backward or execute env steps.
Local verification: JAXCUDA and real-data/LoRA preflight passed; checkpoint fully downloaded.
Full official checkpoint load OOM on home8GB4060 with about2.9GB already used; no policy forward or envstep completed.
This is not a model-quality result. Run the separate smoke on the cloud GPU before rollout/training:

~~~bash
"$PY" -u scripts/smoke_pi05_checkpoint.py
~~~

It checks H50 and simulator-reset observations and performs no optimizer update/envstep.
Read records/pi05_checkpoint_inference_smoke.json; source prep does not establish training quality.

## User starts vanilla π0.5 first

~~~bash
"$PY" -u scripts/evaluate_pi05_open_drawer.py --episodes 1
"$PY" -u scripts/evaluate_pi05_open_drawer.py --episodes 3
"$PY" -u scripts/evaluate_pi05_open_drawer.py --episodes 20
~~~

Each is a separate manually started run. Check one before starting three or twenty.
New timestamped output under results/robocasa_eval contains protocol, steps, per-seed/final statistics and videos.
Explicit progress/action statistics print in terminal. Three input-camera views stream to MP4 (left/right/wrist).
Policy predicts50actions, executesfirst5 before replan, matching the official fork evaluator.
DP8-action and historical ACT100-step settings are unrelated to this policy.
Success is RoboCasa info.success.

## Strict LoRA: benchmark before long training

Official variants: gemma_2b_lora(rank16/alpha16) and gemma_300m_lora(rank32/alpha32).
Our filter freezes **all non-LoRA weights**, including SigLIP, original transformer weights, projections and heads.
Inspect preflight and actual-run trainable parameter lists.

Do not copy DPbatch32. Start one explicit50-update benchmark. Try larger batches in separate runs
only if measured VRAM allows. Review losses, compute/data times and peak memory:

~~~bash
"$PY" -u scripts/finetune_pi05_h50_lora.py \
  --train --batch-size 1 --steps 50 --exp-name benchmark_b1
~~~

Use another name for another batch, e.g. batch2/benchmark_b2. No batch capacity is certified by setup.
Actual training requires an explicit batch chosen from measurement:

~~~bash
"$PY" -u scripts/finetune_pi05_h50_lora.py \
  --train --batch-size CHOSEN_BATCH --steps 10000 --lr 1e-5 --seed 0 --exp-name seed_0
~~~

PeakLR1e-5/500stepwarmup/10kupdates are initial configurable settings, not validated hyperparameters.
Check intermediate rollouts; do not assume 10k must improve upon earlier checkpoints.
Training uses only the fixed28human episodes.

Output: outputs/OpenDrawer/pi05_h50_left_lora/seed_0/
- actual_config.json, native_config_snapshot.json, trainable_parameters.json, metrics.jsonl, training_summary.json
- native checkpoints2500/5000/7500/10000 and last→10000
- each keeps full inference params, normalization assets, train_state for recovery.
Full native params include adapters; no bare state_dict or required merge.
Metrics include loss, update, LR, compute/data wait, wall-clock time and JAX GPU counters.

~~~bash
"$PY" -u scripts/evaluate_pi05_open_drawer.py --lora \
  --checkpoint outputs/OpenDrawer/pi05_h50_left_lora/seed_0/last --episodes 3
~~~

Use the same evaluation seeds/Hz/cameras/replan protocol for vanilla/LoRA.
Use a persistent terminal such as tmux on the GPU host. Back up full params/assets/config/metrics
before stopping a nonpersistent rental. Dataset/weights remain outside Git.

## Interface decisions / upstream pitfalls

- OpenDrawer, canonical prompt open the left drawer; only validated left seeds.
- H50:28human left episodes. Stored128px images were upsampled to256; native model resizes/pads224.
  Upsampling adds no recovered detail.
- Human/nativeenv20Hz. ACT16Hz historical; DreamGen16Hz deferred pending a separate temporal decision.
- Native saved z-score normalization(use_quantile_norm=False), even though quantiles exist in assets.
- State order: EErelativexyz3, EErelativequat4, basexyz3, basequat4, gripper2.
  H50 storage order differs; official modality slices assemble it correctly.
- Action order: deltaEEpos3, rotation3, gripper1, base4, controlmode1; no extra deltas/scaling.
- Camera slots: left→base_0_rgb, wrist→left_wrist_0_rgb, right-agentview→right_wrist_0_rgb.
  Last slot's name does not mean a second physical wrist camera.
- RGB HWC uint8→native transforms→saved normalization→model. No BGR/duplicates/extra rotation.
- Upstream fallback norm stats hardcodes different PandaOmron storage indices; bypass with checkpoint stats.
- Official example uses args.task_soup while Args defines task_set. Our OpenDrawer evaluator avoids that entrypoint.
- Source upstream stays unmodified. Check saved verification status; do not claim untested training passed.
