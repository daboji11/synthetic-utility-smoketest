# Synthetic trajectory utility smoke test

**Project question:** Does downstream learning utility vary among synthetic robot trajectories even when they have similar validity?

**Current stage:** prepare fine-tuning of an official RoboCasa-pretrained Diffusion Policy on 28 H50 human-left episodes at native 20fps. The LeRobot U-Net DP path is retained as an engineering smoke only. The fixed base is 28 H50 left-drawer demos; synthetic data is deferred. **Data provenance correction:** DAVIAN-Robotics/robocasa-MG_30 is MimicGen synthetic simulator data, not human demonstrations. H50 is the human source and DreamGen is the future video-generated synthetic pool. Older Phase 0 sections below describe toy sequences whose action-error threshold is not robot task success.

## Active target — official pretrained RoboCasa Diffusion Policy

Official pretrain_human300 EMA checkpoint is connected through the native RoboCasa DP fork.
One actual OpenDrawer left-drawer seed=3 rollout completed: 750 steps, 94 native 8-action chunks,
finite actions, zero clipping, success=false. The complete video and per-step stats are saved locally.
This is interface verification, not a successful zero-shot benchmark.

The fixed 28 H50 human left demos have been exported at native 20Hz for the official dataset reader:
5916 frames, three cameras, 128->256 bilinear upsampling. Original datasets remain intact.
Official reader windows and pretrained forward-only compatibility checks passed.
Fine-tuning has not started. Runnable native adapters now provide fresh optimizer/EMA initialization,
preserved pretrained normalization, full checkpoints, metrics and3 evaluation episodes. See [native integration, results and target-data plan](docs/robocasa_pretrained_dp.md).

ACT runs and earlier LeRobot from-scratch DP checks remain historical engineering records.
The old scratch trainer requires explicit opt-in for non-smoke training.
For the4090 run, see [setup, dataset transfer and train/eval commands](docs/4090_h50_native_dp.md).
See [earlier engineering checks](docs/open_drawer_diffusion.md).

## Setup and run

Python 3.10+ is required. A project-local `.venv` is already configured. Run:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
python scripts/run_smoke_test.py --config configs/experiment/smoke_test.yaml
```

To recreate the environment on macOS arm64 with Python 3.10, use `uv venv .venv --python 3.10` and `uv pip install --python .venv/bin/python -r requirements-lock.txt`. `requirements.txt` lists direct dependencies without version pins for other compatible platforms.

## Phase 0 dataset inspection

The [single-task config](configs/dataset/pnp_counter_to_cab.yaml) selects `PnPCounterToCab` from `DAVIAN-Robotics/robocasa-MG_30` and `Keh0t0/dreamgen_robocasa_p3_init0_v2_droid`. The MG_30 repository stores this task under 23 object-specific instructions, so the adapter matches only instructions of the form “pick the ... from the counter and place it in the cabinet.” There are 30 matching MG_30 episodes. The synthetic repository has 60 matching episodes, all with the water-bottle instruction. The original instruction is preserved in the CSV.

```bash
python scripts/inspect_datasets.py --skip-video
python scripts/inspect_datasets.py
```

The script reports only this task, checks action/state/camera/task-label/FPS compatibility, and exports five MG_30 and ten DreamGen MP4 clips plus `episodes.csv` and `inspection.json` under `results/dataset_inspection/PnPCounterToCab/`. The adapter downloads task and episode metadata at construction, then fetches action/state and video shards lazily. Each repository stores multiple tasks in shared files. For this task the real data Parquet is approximately 19 MB and the MG_30 left-camera MP4 approximately 209 MB; the synthetic equivalents are approximately 88 MB and 148 MB. No other cameras were downloaded. MG_30 videos remain at 20 fps and synthetic videos at 16 fps. The schema matches, but joint ACT training needs an explicit temporal handling decision before use.

The current Phase 0 candidate is `OpenDrawer`. A metadata/state-action scan of all 24 synthetic tasks is saved in `results/dataset_inspection/trajectory_variation_scan.csv`. The following commands export the 60 DreamGen and 21 exact-instruction MG_30 previews, then cluster synthetic trajectories and select two ten-episode subsets. The cached Hugging Face shards can be reused with `--offline`; omit that flag on a new machine to download the required shared shards.

```bash
python scripts/inspect_datasets.py --config configs/dataset/open_drawer_mimicgen_debug.yaml --real-count 21 --synthetic-count 60 --offline
python scripts/select_open_drawer_trajectories.py
```

Selection outputs are in `results/dataset_inspection/OpenDrawer/trajectory_similarity/`: `clusters.csv`, `cluster_representatives.csv`, `redundant_10.csv`, `diverse_10.csv`, the pairwise distance matrix, method details, and three local HTML preview pages. The geometric selections are not task-success or training-utility labels. No ACT training or FPS resampling is performed.

Private or gated Hugging Face repositories require both account authorization from the repository owner and a local `.venv/bin/hf auth login`. A successful CLI login alone does not grant access to a private repository. Do not put tokens in config files or the repository.

The run writes `results/<experiment_name>/config.yaml`, `metrics.json`, `checkpoint.pt`, and `train.log`. A name already present in `results/` raises an error to preserve previous runs; change `experiment_name` to rerun. A fixed seed reproduces data, selection and training on the same platform and dependency versions.

Individual stages can also be run in order:

```bash
python scripts/prepare_dataset.py --config configs/experiment/smoke_test.yaml
python scripts/create_subset.py --config configs/experiment/smoke_test.yaml
python scripts/train_policy.py --config configs/experiment/smoke_test.yaml
python scripts/evaluate_policy.py --config configs/experiment/smoke_test.yaml
```

The stage commands write reusable data to `data/processed/<experiment_name>/` and the final artifacts to `results/<experiment_name>/`. Each stage requires outputs from the preceding stage. The one-command runner uses the same stages.

## Interfaces and extension points

`BaseRobotDataset` stores trajectories with `trajectory_id`, `observations`, `actions`, `task_id`, `success`, and optional `metadata`. Observation/action shapes are checked by the toy trainer, not imposed by the dataset interface. New dataset adapters should implement the base interface and register their loader in `src/datasets/__init__.py`. `dataset.evaluation_dataset` configures the held-out pool. Trajectory IDs must be unique within a dataset and across real and synthetic pools when merged. `metadata` is opaque to the core pipeline.

Selectors return trajectory IDs. Only `random` is implemented. The other four methods have explicit `NotImplementedError` stubs so switching a config cannot silently run a different algorithm. A future selector can use `context` for real-data coverage or a current policy.

The `SmallPolicy` is a tiny behavior cloning MLP. Replace it via the policy factory; adapt `Trainer` if the new model uses a different batch representation. The evaluator accepts an environment with `run_episode(policy)` for future closed-loop success and return; the current dataset evaluator is only a toy proxy.

For a real experiment, hold the real data, synthetic count, policy, optimization steps and evaluation protocol fixed across subsets. Split descendants of source demonstrations together, and evaluate across multiple seeds. The toy data cannot test the validity-versus-utility hypothesis.

## Not implemented yet

- Mixed real/synthetic ACT-ready dataset assembly beyond the real-only 16 fps baseline
- DreamGen/RoboCasa simulator rollout integration
- Utility grader or learned trajectory valuation
- Sophisticated diversity or policy-aware selection
- Robot simulator rollout evaluation

## Linux OpenDrawer MimicGen debug ACT run

The earlier real_16fps directory and real_only ACT outputs are historical names.
Their source is DAVIAN-Robotics/robocasa-MG_30, which contains MimicGen
synthetic simulator demonstrations. The 21 episode, 16 fps dataset and both
ACT runs are **debug_mimicgen_only** engineering checks, not human baselines.
The completed 10,000 step checkpoint remains at
outputs/OpenDrawer/real_only/seed_0/checkpoints/last/pretrained_model/.
outputs/OpenDrawer/debug_mimicgen_only/ is an alias to the preserved output
folder; each run also has EXPERIMENT_ROLE.json. These weights are not an
H50 human-only baseline.

Formal OpenDrawer human input is DAVIAN-Robotics/robocasa-H50: 54 episodes,
28 left-drawer and 26 right-drawer, 20 fps. DreamGen OpenDrawer has 60
left-drawer episodes at 16 fps. Both use 12-dimensional actions,
16-dimensional state, and 128×128 left-camera images. The H50 raw shared
Parquet and the single left-camera video shard containing OpenDrawer are
cached under data/raw/huggingface/. configs/dataset/open_drawer.yaml
selects only the H50 left-drawer instruction and DreamGen left-drawer instruction.
No H50 resampling or formal ACT training has been performed.

The debug runner writes metrics.jsonl, train.log, and the resolved
train_config.json into the run directory. LeRobot saves complete
pretrained_model/ checkpoints with model weights, policy configuration,
training configuration, and processor configurations with normalization state.
Dataset and training output directories are Git-ignored.

## Formal OpenDrawer H50 human-left ACT baseline

The fixed human base is exactly 28 H50 human demonstrations with instruction
open the left drawer. The source remains in data/raw/huggingface/. The
separate processed dataset is data/processed/OpenDrawer/h50_human_left_16fps/:
4,744 frames at 16 fps with one 128×128 left camera, 12-dimensional actions,
and 16-dimensional state. scripts/preprocess_open_drawer_h50_left_16fps.py
uses target timestamps j/16 and the nearest 20 fps source timestep, breaking
exact ties toward the earlier frame. Image, action, and state share that index.
No action interpolation or scaling is applied. The resampling manifest records
all 28 source episode IDs and selected frame indices.

The formal training configuration is
configs/experiment/open_drawer_h50_human_left_act.yaml. It uses the same ACT
policy defaults as the MG_30 debug run, batch size 4, 10,000 steps, seed 0,
CUDA, and checkpoints every 2,500 steps. The new run output will be
outputs/OpenDrawer/human_left_only/seed_0/. The run directory is intentionally
absent until training starts so the runner can protect against overwrite.

From the repository root in WSL Ubuntu:

    conda activate synutility312
    python scripts/train_open_drawer_h50_left_act.py --preflight
    python scripts/train_open_drawer_h50_left_act.py

Only the last command starts training. The runner writes metrics.jsonl,
train.log, the complete LeRobot pretrained_model checkpoint including
normalization processors, and checkpoints/last. The MG_30 debug checkpoint
remains under outputs/OpenDrawer/real_only/ and is marked
debug_mimicgen_only. No formal training was run during dataset preparation.

## H50 human-left three-camera ACT comparison

This variant uses the same 28 H50 human episodes and exactly the same state,
action, timestamp, and resampling indices as the single-camera baseline. It adds
all three available 128x128 views: robot0_agentview_left_image,
robot0_agentview_right_image, and robot0_eye_in_hand_image. All cameras use the
same selected source timestep at 16 fps.

Processed dataset: data/processed/OpenDrawer/h50_human_left_16fps_3cam.
Training config: configs/experiment/open_drawer_h50_human_left_3cam_act.yaml.
Output: outputs/OpenDrawer/human_left_only_3cam/seed_0/.

The ACT settings match the original checkpoint, including chunk_size=100,
n_action_steps=100 and no temporal ensembling. Batch size is 4, training steps
are 10000, seed is 0, and checkpoints are saved every 2500 steps. This is a
separate camera comparison; retain the single-camera run for comparison.

Run manually from the repository in WSL Ubuntu:

    conda activate synutility312
    python -u scripts/train_open_drawer_h50_left_act.py --multicamera --preflight
    python -u scripts/train_open_drawer_h50_left_act.py --multicamera

The final command starts training. It prints training metrics and saves
train.log, metrics.jsonl, requested_config.yaml, the resolved train_config.json,
and complete LeRobot checkpoints with preprocessing/postprocessing state.
Checkpoint aliases include 2500, 5000, 7500, 10000, and last.
If batch size 4 fails with CUDA OOM, the explicit batch-size-2 option uses a
separate seed_0_batch_2 directory:

    python -u scripts/train_open_drawer_h50_left_act.py --multicamera --batch-size 2

After the three-camera checkpoint has finished, evaluate it in the simulator
environment with the original 100-step execution setting:

    bash scripts/eval_open_drawer_h50_3cam.sh

This manual launcher uses the WSL synutility-sim interpreter and EGL rendering.
It requires the saved final checkpoint at training step 10000, pins the resolved
checkpoint, and tests exactly left-drawer seeds 3, 4, and 5. Control is 16 Hz,
ACT predicts 100 actions and executes all 100 before replanning, and temporal
ensembling is disabled. It loads the saved preprocessor/postprocessor and checks
all camera, state, and action normalization statistics against the training data.

Terminal output includes per-100-step progress, per-episode success, drawer
opening fraction, and per-dimension raw action/clipping statistics. Outputs are:

    outputs/OpenDrawer/human_left_only_3cam/seed_0/robocasa_eval/three_camera_seed_3_4_5/
        protocol.json
        episodes.jsonl
        summary.json
        steps/seed_3.jsonl
        steps/seed_4.jsonl
        steps/seed_5.jsonl
        videos/seed_3.mp4
        videos/seed_4.mp4
        videos/seed_5.mp4

The console log is a sibling file named three_camera_seed_3_4_5_console.log.
Videos stream to disk at 16 fps, with left, right, and wrist views in that order,
plus a status strip showing step, reward, and success. Frames are not accumulated
in RAM. The camera images supplied to ACT are preserved without video overlays.
Existing output directories and console logs are protected from overwrite.
For another run, provide a fresh name:

    bash scripts/eval_open_drawer_h50_3cam.sh three_camera_repeat_1

A CPU preflight can test the final checkpoint on a saved training observation
without creating a simulator:

    MUJOCO_GL=egl /home/dabozi/miniconda3/envs/synutility-sim/bin/python -u scripts/evaluate_open_drawer_h50_act_robocasa.py --multicamera --episodes 3 --start-seed 3 --seeds 3 4 5 --run-name three_camera_seed_3_4_5 --preflight

To rebuild the dataset on another WSL machine with the single-camera reference
and H50 metadata already cached:

    conda activate synutility312
    python -u scripts/preprocess_open_drawer_h50_left_16fps_3cam.py

Only video shards referenced by the 28 selected episodes are downloaded, using
the original pinned H50 revision. Shared shards may contain other tasks; only
the selected left-drawer episodes are processed. The original H50 files and
single-camera processed dataset are preserved.

## Three-camera ACT with explicit camera identity

The original three-camera model does consume all three images. The read-only
audit in scripts/audit_open_drawer_h50_3cam.py traced three backbone calls and
50 Transformer tokens (2 state/latent tokens plus 16 image tokens per camera).
Across the 28 episodes, 84 frames per camera showed no black or duplicate views.
Changing any view changes predicted actions. The saved audit is under
results/robocasa_eval/camera_audit_3cam/.

The installed ordinary ACT repeats the same 2D positional encoding for each
view without an explicit camera identifier. Swapping left and right images
changed the audited model's output by at most approximately 8e-7. This is a
fusion limitation, not evidence that cameras were omitted, and does not establish
the cause of rollout failure.

The separate act_camera_id policy adds a learned camera identity to the visual
positional encoding: one 512D vector per camera, 1536 extra parameters total.
The implementation lives in src/policies/act_camera_id/ and leaves the installed
LeRobot package intact. It uses the standard ACT trainer, shared ResNet18,
processors, loss, optimizer and action execution. Camera order is saved in the
model config and validated on load. Embeddings are saved with model.safetensors
and automatically restored by the evaluator.

This is an architecture comparison with the existing ordinary ACT runs.
It uses the same 28 human-left episodes, three synchronized cameras, 16 fps,
128x128 resolution, batch size 4, 10000 steps, seed 0, and 100-step replanning.
It does not establish improved success rate until tested.

Launch manually inside WSL:

    conda activate synutility312
    python -u scripts/train_open_drawer_h50_left_act.py --camera-id --preflight
    python -u scripts/train_open_drawer_h50_left_act.py --camera-id

Output: outputs/OpenDrawer/human_left_only_3cam_camera_id/seed_0/.
Config: configs/experiment/open_drawer_h50_human_left_3cam_camera_id_act.yaml.
The model is initialized for a new run; previous model weights are preserved.

After this run finishes, test exact seeds 3, 4, and 5 with live stats and videos:

    bash scripts/eval_open_drawer_h50_3cam_camera_id.sh

The launcher checks final step 10000 and writes under the camera-identity run's
robocasa_eval/three_camera_id_seed_3_4_5/ directory.

Focused verification (CPU tests, no optimizer updates or simulator rollout):

    python -m pytest -q tests/test_act_camera_identity.py

Tests check gradients reaching all views and camera identities, removal of
camera-swap invariance, and identical output after complete model/processor
save and reload. The real uint8 multi-camera training batch and 100-action
targets were also checked through the installed LeRobot dataset/policy factories.
