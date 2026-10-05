# Official RoboCasa pretrained Diffusion Policy

## Status (2026-10-05)

The native official pretrain_human300 EMA policy was strictly loaded and run in actual OpenDrawer.
One left-drawer episode at seed 3, split=all, completed 750 environment steps at 20Hz.
There were 94 model calls, each using 100 DDPM denoising steps, executing 8 sequential actions per call.
All actions were finite and no values clipped. The episode did **not** succeed; reward=0 and
max drawer opening approximately 0.00055. This confirms interface execution, not zero-shot task competence.

The complete three-camera 20fps video and per-step actions are saved at:
results/robocasa_eval/pretrained_dp_seed_3/rollout.mp4 and steps.jsonl.
The 751-frame video was decoded completely. Runtime was 235.93 seconds;
PyTorch peak allocator usage was 440.62 MiB (excludes simulator/OpenGL and driver allocations).
The saved preview shows the arm moving around the cabinet without opening the drawer.
Do not interpret this single episode as a benchmark success rate.

H50 target data is prepared and accepted by the **unmodified official LerobotCotrainingDataset**.
There has been no fine-tuning, optimizer step, synthetic mixed training or ACT change.

## Sources and pins

- Official fork: https://github.com/robocasa-benchmark/diffusion_policy
  commit 41212698b6f481ed92a55f0d5f1778ec56bea417
- Official robomimic fork: https://github.com/robocasa-benchmark/robomimic
  commit 7dc18461c4fc497095f1044f4c40cd44fff62e6a
- Model release: https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/diffusion_policy/17.40.09_train_diffusion_transformer_hybrid_pretrain_human300
  revision c484448aba1a9b60a04c9b0ca117241518ea69f3
- Only latest.ckpt (1,697,722,871 bytes), normalizer.pkl and config metadata downloaded.
- RoboCasa commit 456174f62b89b8fca99eaaf33949c29fec9cfc2a;
  robosuite commit 5ce6643f3092639d08f7b0f90ed1c6a84f50552c.
  Tracked simulator files were clean.
- Native source checkouts reside outside this project; the setup script clones pinned versions.
  Simulator assets, datasets and model weights stay ignored.

## Environment

All work ran in WSL Ubuntu-24.04. The isolated synutility-dp-pretrained environment was
cloned from the working synutility-sim environment. Existing CUDA training and simulator envs
were preserved. Runtime: Python 3.12, torch 2.7.1+cu128, RTX 4060 Laptop GPU, CUDA available.

requirements-native-dp.txt records incremental runtime versions.
requirements-native-dp-linux-lock.txt records the complete actual environment; it contains local
editable source paths and is a record, not a portable installer.

The legacy robomimic metadata declares torch 2.0.1 / torchvision 0.15.2 / numpy 1.23.2 /
diffusers 0.11.1. Those declared pins differ from this measured runtime; pip check reports them.
The isolated native env now uses RoboCasa's declared LeRobot 0.3.3 so the official workspace
and its RoboCasa lerobot_utils imports work. The native policy and GR00T reader do not use
LeRobot policy processors. pip check still reports optional LeRobot teleoperation packages
not installed and inherited version mismatches for datasets/draccus/gymnasium/wandb, along
with robomimic's legacy torch/numpy pins. These are recorded verbatim in native_dp_environment.json.
Do not reuse this env for the previous LeRobot 0.6.1 trainer.

All 10 existing repo tests passed in the preserved synutility-sim environment.
Collecting these legacy LeRobot 0.6.1 tests in the native dependency stack initially failed;
they belong to the original environment. Native verification used the actual checkpoint rollout,
official target reader, finite forward-only model loss and official workspace import.

Recorded setup issues:
1. Upstream setup.py misses namespace packages: a .pth source path was added in this env.
2. diffusers 0.39 removed typing aliases imported by native robomimic. Pinning 0.30.3 fixes
   that import without editing either upstream fork or the trained architecture.
3. wandb 0.30 introduced a protobuf conflict with the inherited tianshou package.
   wandb 0.17.9 / protobuf 3.19.6 were used. A wandb 0.15 attempt also failed because
   pathtools uses imp, which Python 3.12 removed.

On another WSL/Linux machine, first establish a working RoboCasa CUDA base env and simulator assets.
Then run bash scripts/setup_robocasa_pretrained_dp.sh (BASE_ENV defaults to synutility-sim).
The native checkpoint is downloaded separately:
    conda activate synutility-dp-pretrained
    python scripts/inspect_robocasa_pretrained_dp.py --download-weights

For an explicit local rerun, choose a fresh output directory:
    MUJOCO_GL=egl python -u scripts/evaluate_open_drawer_pretrained_dp.py --seed 3 \
      --output results/robocasa_eval/pretrained_dp_seed_3_repeat

The default split=all and seed=3 deliberately retain the already checked left-drawer scenario.
This single-scene smoke is not the official multi-task held-out benchmark.
The checkpoint architecture, DDPM schedule, crop, normalizer and action chunk length were unchanged.

## Native input and target mapping

Policy: DiffusionTransformerHybridImagePolicy, three independent language-conditioned visual
encoders, Transformer 12 layers / 8 heads / embedding 512, observation history 2,
prediction horizon 10, executed action horizon 8, DDPM 100.

| Native feature | H50 source |
| --- | --- |
| robot0_agentview_left_image | observation.images.robot0_agentview_left_image |
| robot0_agentview_right_image | observation.images.robot0_agentview_right_image |
| robot0_eye_in_hand_image | observation.images.robot0_eye_in_hand_image |
| robot0_base_to_eef_pos | observation.state[0:3] |
| robot0_base_to_eef_quat | observation.state[3:7] |
| robot0_gripper_qpos | observation.state[7:9] |
| lang_emb | official CLIP ViT-L/14 text projection, 768 dimensions |
| action | EE delta position 3, EE delta rotation 3, gripper 1, base motion 4, control mode 1 |

Native observation uses 9 proprioceptive dimensions. H50's base position/quaternion remain in
the stored 16D state, and are not model inputs. We do not concatenate a different 16D feature
into the pretrained encoder. The supplied native PandaOmron modality template has **different
storage ordering**; the target modality.json explicitly maps H50 slices instead of copying it blindly.

Raw RGB is uint8 HWC; native preprocessing gives float32 CHW /255.
The checkpoint's own normalizer transforms RGB range to [-1,1], delta actions remain identity.
No LeRobot ImageNet or min/max action processors, no extra normalization or action scaling.
Language uses the exact official LangEncoder. H50 instruction is literally lowercase
"open the left drawer"; the simulator emits "Open the left drawer.".
The strings refer to the same task, but their text embeddings are not asserted to be identical.

## Prepared H50 target dataset

data/processed/OpenDrawer/h50_human_left_native_dp_20fps_256

- Human source: DAVIAN-Robotics/robocasa-H50, revision 7e9802803a02be938683ea3f14e56e877d1c1ef7.
- Exactly 28 left episodes / 5916 original frames / 20fps. No right episodes or synthetic data.
- Source: data/processed/OpenDrawer/h50_human_left_20fps_3cam, left subset with all source timesteps.
- New export: one Parquet per episode, one video per camera per episode,
  meta/info.json, episodes.jsonl, tasks.jsonl, stats.json, modality.json.
- All 84 videos fully decoded. Numeric state, action and timestamps preserved exactly.
- Three images are bilinearly upsampled from 128x128 to the pretrained native 256x256.
  This changes input size only; it does not recover the source's missing spatial detail.
- No 16fps resampling; originals and earlier ACT datasets preserved. Native rollout control is 20Hz.
- Approximately 208 MiB.
- Actual official reader verification: 84 beginning/interior/end windows; camera RGB bounds,
  state ordering, 12D action ordering and first/last boundary padding passed.
- Output action batch [10,12], images [2,3,256,256], language [2,768].
- The pretrained policy compute_loss forward on a native batch was finite;
  no backward, optimizer or weight update was executed. This is a schema check, not a fitted loss result.

Commands:
    python scripts/prepare_h50_native_dp_dataset.py
    MUJOCO_GL=egl python -u scripts/verify_h50_native_dp_dataset.py

Preparation refuses to overwrite existing data. The verifier can be rerun.
configs/native_dp/task/h50_open_drawer_left.yaml is a load-tested **dataset/task candidate**:
dataset_paths points only to this target via H50_NATIVE_DP_DATASET, dataset_soup=null.
It does not include pretrain_human300 data in target fine-tuning.

## Fine-tuning implementation

The previous initialization/normalization/save issues are now handled by the local adapter:

- Published EMA loads into both train model and EMA; optimizer/global counters start fresh.
- PretrainedNormalizerH50Dataset uses the official dataset/window reader but returns the saved
  pretrained normalizer, so the official trainer keeps the original input convention.
- Exact optimizer-update counting drives metrics and2500/5000/7500/10000 checkpoint names.
- A final native checkpoint is guaranteed; the model architecture and official training loop are retained.
- Fine-tuned evaluator verifies update count and companion normalizer, then uses8-action chunks.
- H50 literal language text is used for fine-tuned inference, matching target training annotations.
- A real-model batch4 forward-only preflight passed. A tiny fixture tested the stock loop and reload.
- No actual H50 fine-tuning or fine-tuned policy rollouts have been launched by the agent.

[4090 setup, dataset transfer, fine-tuning and3 rollout commands](4090_h50_native_dp.md).

## Small verification records

- records/pretrained_dp_rollout_seed_3.json
- records/h50_native_dp_dataset_verification.json
- records/robocasa_pretrained_dp_candidate.json
- records/native_dp_environment.json
