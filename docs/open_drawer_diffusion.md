# OpenDrawer native-20Hz human-only Diffusion pipeline

> Updated requirement: this LeRobot U-Net pipeline is an engineering record only.
> The active goal is RoboCasa-pretrained DP fine-tuning on the same28 H50-left episodes.
> The scratch training command below now requires explicit --engineering-from-scratch;
> it must not be used as pretrained fine-tuning. See robocasa_pretrained_dp.md.
> One real seed3 pipeline validation passed at20Hz:750steps,24replans,zero clips,751decoded video frames.
> The500-step smoke policy had0/1 success, not a formal benchmark.

## Data and model

- Fixed human base: 28 DAVIAN-Robotics/robocasa-H50 episodes with exact instruction "open the left drawer".
- Dataset: data/processed/OpenDrawer/h50_human_left_20fps_3cam, 5,916 original timesteps, 20fps.
- No temporal resampling, action interpolation/scaling, right-drawer episodes, or synthetic trajectories.
- State/action arrays were checked exactly against all original H50 timesteps. Images retain 128x128 RGB.
- Cameras, in order: observation.images.robot0_agentview_left_image,
  observation.images.robot0_agentview_right_image, observation.images.robot0_eye_in_hand_image.
- Full provenance and source-to-local episode mapping: provenance_manifest.json in the dataset.
- Retained official LeRobot 0.6.1 Diffusion defaults: 2 observations, 64 prediction steps,
  32 executed actions, DDPM with 100 denoising steps, three separate ResNet18 encoders,
  U-Net down_dims=[512,1024,2048], MIN_MAX state/action normalization, MEAN_STD visuals.
- ImageNet initializes the visual backbones. No pretrained robot-policy checkpoint is loaded.
- BF16 autocast is enabled in both the training wrapper and CUDA evaluation.
- Model architecture, cameras, input size and horizons are identical across 4060/4090 runs.
- Official default drop_n_last_frames=7 is retained in training sampling; no source frames
  are removed from the prepared dataset. All defaults are checked against the installed version.
- Simulated control dt=0.05s; the model replans every 32 actions (1.6 simulated seconds).
  This is 0.625 model generations per simulated second, not a guarantee of real-time wall-clock speed.

## Local WSL commands

From the project root in WSL Ubuntu, with synutility312 activated:

    python scripts/prepare_open_drawer_h50_left_20fps_3cam.py

Preparation has already completed. It refuses to overwrite the existing dataset.

Tiny training (use a fresh run name; the recorded original names already exist):

    python -u scripts/train_open_drawer_h50_diffusion.py --smoke --batch-size 1 --steps 200 --run-name smoke_b1_repeat
    python -u scripts/train_open_drawer_h50_diffusion.py --smoke --batch-size 2 --steps 500 --run-name smoke_b2_repeat

Smoke outputs are separate from the formal seed_0 run. OOM is reported and never
retried with different cameras, encoder sizes or horizons.

## 4090 formal training

Config: configs/experiment/open_drawer_h50_human_left_3cam_diffusion.yaml
Provision a Linux training environment with the same LeRobot/Diffusers versions and CUDA-compatible torch.
Copy this repo's code and the complete processed native20fps three-camera dataset to the pod,
keeping the relative dataset path. The runner resolves paths from its own repo root,
not the previous WSL home directory.

    python -u scripts/train_open_drawer_h50_diffusion.py --batch-size 8 --preflight
    python -u scripts/train_open_drawer_h50_diffusion.py --batch-size 8

Batch 8 is a starting configuration for 24GB, not a completed 4090 memory measurement.
--batch-size can reduce it while preserving the model and horizons; no automatic fallback.

Formal output: outputs/OpenDrawer/diffusion_human_left_3cam/seed_0/
Checkpoint aliases: checkpoints/2500, /5000, /7500, /10000 and /last.
Each checkpoint includes model.safetensors, config.json, train_config.json,
policy_preprocessor.json, policy_postprocessor.json and their normalization safetensors,
plus the official optimizer/scheduler/training-state files.
resolved_config.json, requested_config.yaml, train_config.json, metrics.jsonl,
camera_audit.json, train.log and training_summary.json are saved at the run root.
Metrics are measured directly inside the official update loop; they are not parsed from rounded console loss.

The official policy and trainer are used unchanged. Repository-local wrappers add
provenance/shape guards, per-step metrics, three-camera input/gradient checks and checkpoint aliases.
No ACT parameters, checkpoint, custom camera embedding or ACT processor enters this pipeline.

## Local evaluation after formal training

Copy the complete final pretrained_model folder and the sibling training_state/training_step.json
back to the formal seed_0 checkpoint tree if training ran on RunPod. Optimizer files are only needed for training resume. Include all saved
processor state files, not only model.safetensors. An alternate checkpoint path may
also be supplied to the shell launcher as its second argument.

    bash scripts/eval_open_drawer_h50_diffusion.sh

This uses synutility-sim, MUJOCO_GL=egl, native20Hz, exactly three accepted-left seeds 3/4/5,
and checks that the checkpoint is step 10000. The default run name is diffusion_seed_3_4_5;
provide a new first argument to repeat without overwriting.
EVAL_PYTHON can override the interpreter on another Linux machine.
Instructions are verified on reset; explicitly requested right seeds cause an error, not silent replacement.
The evaluator is limited to at most three episodes for this stage and never launches additional training.

All three videos stream directly to disk in left/right/wrist column order.
Each video is 384x160 at 20fps, with a status strip; no image modification enters the policy.
Outputs include protocol.json, episodes.jsonl, summary.json, per-step raw/applied action logs,
per-episode action_diagnostics JSON and train-vs-rollout CSV tables.
The terminal explicitly prints success/reward/length/wall time, drawer motion, replans,
per-dimension raw action statistics and separate arm/gripper/base clipping counts.

For checkpoint/input inference checks without constructing the simulator:

    MUJOCO_GL=egl /home/dabozi/miniconda3/envs/synutility-sim/bin/python scripts/evaluate_open_drawer_h50_diffusion_robocasa.py --preflight --preflight-device cuda

Each environment observation is preprocessed with saved training processors on every step.
Official Diffusion select_action updates its two-observation history each call and generates
a new 64-step prediction only when the 32-action execution queue empties.
Action ordering and the validated RoboCasa convert_action adapter are reused; no manual scaling.
Training paths recorded on RunPod may differ locally; dataset identity and saved stats are checked.

## Recorded 4060 results (2026-10-05)

| Batch | Steps | Allocated peak MiB | Reserved peak MiB | Steady steps/s | Total worker wall seconds | First/last 50 mean loss |
|---|---|---|---|---|---|---|
| 1 | 200 | 5647.3 | 5712 | 4.53 | 59.80 | 1.0661 / 0.9965 |
| 2 | 500 | 5643.3 | 5736 | 4.50 | 126.00 | 1.0607 / 0.1340 |

CUDA tensors/allocator measurements cover the PyTorch process, not all Windows desktop GPU usage.
Timing includes worker setup and final checkpoint writing; stable throughput excludes the first ten steps.
Both runs had finite losses/gradients, no OOM, three matched encoder inputs and nonzero gradients.
Saved checkpoint CPU and CUDA/bf16 preflights verified RGB/state equivalence, saved normalization,
action roundtrip, exactly 100 U-Net calls per generation and 32 queued actions.

These are pipeline checks, not trained robot baselines. Spatial behavior and success rate remain
unmeasured until the formal training checkpoint is evaluated. Inspect videos for correct drawer
approach, handle alignment, grasp and pull. If spatial grounding is still wrong, stop and review;
do not automatically increase steps or add synthetic data.

## Dependencies and retained records

Training env: synutility312. Simulator env: synutility-sim. torch=2.7.1+cu128, LeRobot=0.6.1,
Diffusers=0.39.0, RoboCasa=1.0.1, robosuite=1.5.2. Added only missing Diffusers and its
importlib_metadata/regex/zipp dependencies; existing CUDA PyTorch was retained.
Training pip check passes. Simulator pip check reports the pre-existing
robocasa 1.0.1 -> lerobot==0.3.3 declaration versus the installed LeRobot0.6.1.
The existing working simulator version combination was not downgraded;
checkpoint/inference imports and the reused adapters work. Real Diffusion rollout remains pending.

requirements-diffusion.txt contains the incremental dependency.
requirements-diffusion-linux-lock.txt and requirements-diffusion-sim-linux-lock.txt snapshot
the actual environments; they are records, not a portable substitute for checking CUDA/driver compatibility.

The ACT branch is retired and preserved. See records/act_baseline_archived.md.
All data, videos and model/optimizer tensors remain ignored by Git.
