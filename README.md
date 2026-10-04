# Synthetic trajectory utility smoke test

**Project question:** Does downstream learning utility vary among synthetic robot trajectories even when they have similar validity?

**Current stage:** Phase 0: infrastructure and a placeholder smoke test. The generated trajectories are toy observation/action sequences. The reported `success_rate` is a **toy action-error threshold**, not robot task success. Use simulator rollouts for scientific comparisons.

## Setup and run

Python 3.10+ is required. A project-local `.venv` is already configured. Run:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
python scripts/run_smoke_test.py --config configs/experiment/smoke_test.yaml
```

To recreate the environment on macOS arm64 with Python 3.10, use `uv venv .venv --python 3.10` and `uv pip install --python .venv/bin/python -r requirements-lock.txt`. `requirements.txt` lists direct dependencies without version pins for other compatible platforms.

## Phase 0 dataset inspection

The [single-task config](configs/dataset/pnp_counter_to_cab.yaml) selects `PnPCounterToCab` from `DAVIAN-Robotics/robocasa-MG_30` and `Keh0t0/dreamgen_robocasa_p3_init0_v2_droid`. The real repository stores this task under 23 object-specific instructions, so the adapter matches only instructions of the form “pick the ... from the counter and place it in the cabinet.” There are 30 matching real episodes. The synthetic repository has 60 matching episodes, all with the water-bottle instruction. The original instruction is preserved in the CSV.

```bash
python scripts/inspect_datasets.py --skip-video
python scripts/inspect_datasets.py
```

The script reports only this task, checks action/state/camera/task-label/FPS compatibility, and exports five real and ten synthetic MP4 clips plus `episodes.csv` and `inspection.json` under `results/dataset_inspection/PnPCounterToCab/`. The adapter downloads task and episode metadata at construction, then fetches action/state and video shards lazily. Each repository stores multiple tasks in shared files. For this task the real data Parquet is approximately 19 MB and the real left-camera MP4 approximately 209 MB; the synthetic equivalents are approximately 88 MB and 148 MB. No other cameras were downloaded. Real videos remain at 20 fps and synthetic videos at 16 fps. The schema matches, but joint ACT training needs an explicit temporal handling decision before use.

The current Phase 0 candidate is `OpenDrawer`. A metadata/state-action scan of all 24 synthetic tasks is saved in `results/dataset_inspection/trajectory_variation_scan.csv`. The following commands export the 60 synthetic and 21 exact-instruction real previews, then cluster synthetic trajectories and select two ten-episode subsets. The cached Hugging Face shards can be reused with `--offline`; omit that flag on a new machine to download the required shared shards.

```bash
python scripts/inspect_datasets.py --config configs/dataset/open_drawer.yaml --real-count 21 --synthetic-count 60 --offline
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

- FPS alignment decision and ACT-ready dataset assembly
- DreamGen/RoboCasa simulator rollout integration
- Utility grader or learned trajectory valuation
- Sophisticated diversity or policy-aware selection
- Robot simulator rollout evaluation
