# 4090: H50-left native pretrained DP fine-tuning and 3 rollouts

The runnable path uses the official pretrain_human300 Transformer DP EMA checkpoint.
Target fine-tuning data is exactly28 H50 human left-drawer demos /5916 frames /20Hz.
There is no synthetic data. The three camera images are the prepared128->256 bilinear export.
Model/crop/horizon/denoiser architecture are unchanged.

Local checks: real model batch4 forward, all three camera inputs, fresh EMA/model initialization,
empty optimizer, and exact pretrained normalizer preservation passed. A tiny fixture exercised
the official trainer, metrics, final checkpoint and evaluator reload. Actual H50 fine-tuning
and the3 fine-tuned rollouts have NOT been launched here or on the rental server.

## 1. On the4090 Linux server: repository and prerequisites

Use a GPU Linux image with a working NVIDIA driver and Conda.
The repository may need GitHub authentication if private.

    mkdir -p ~/research
    cd ~/research
    git clone https://github.com/daboji11/synthetic-utility-smoketest.git
    cd synthetic-utility-smoketest
    mkdir -p data/raw/transfers

If already cloned, enter the repository and git pull --ff-only.

On a minimal Ubuntu image, run the following as root (or prefix apt-get with sudo):

    apt-get update
    apt-get install -y git rsync ffmpeg libegl1 libgl1 libglvnd0 libglfw3 libgles2

Set up pinned source forks and the isolated Conda environment:

    CONDA_EXE="$(conda info --base)/bin/conda" bash scripts/setup_robocasa_pretrained_dp.sh
    eval "$(conda shell.bash hook)"
    conda activate synutility-dp-pretrained
    export MUJOCO_GL=egl
    export PYNPUT_BACKEND=dummy

The setup clones pinned robosuite/RoboCasa/robomimic/DP and adds the upstream namespace-package path.
It clones a working base env when present, otherwise creates Python3.12.
A working CUDA PyTorch is retained; a new env defaults to torch2.7.1 /torchvision0.22.1 cu118.
PYTORCH_CUDA_INDEX can select the official cu126/cu128 wheels instead.
[Official PyTorch2.7.1 wheel commands](https://pytorch.org/get-started/previous-versions/#v271).

The legacy package metadata does not fully agree with the verified runtime pins.
Use this isolated environment and run both preflight gates below; do not replace the old LeRobot env.
Fresh-server setup has not been executed by the agent because no rental SSH endpoint is available.

## 2. Transfer the prepared28-demo dataset from local WSL

Datasets are NOT committed to Git. A208MiB transfer archive is already prepared locally:

    /home/dabozi/research/synthetic-utility-smoketest/data/raw/transfers/h50_native_dp_28.tar

In your **local WSL terminal**, replace PORT/USER/HOST with the rental SSH values:

    cd /home/dabozi/research/synthetic-utility-smoketest
    scp -P PORT data/raw/transfers/h50_native_dp_28.tar \
      data/raw/transfers/h50_native_dp_28.tar.sha256 \
      USER@HOST:~/research/synthetic-utility-smoketest/data/raw/transfers/

On the4090, from the repository root:

    sha256sum -c data/raw/transfers/h50_native_dp_28.tar.sha256
    mkdir -p data/processed/OpenDrawer
    test ! -e data/processed/OpenDrawer/h50_human_left_native_dp_20fps_256
    tar -xf data/raw/transfers/h50_native_dp_28.tar -C data/processed/OpenDrawer

Run the extraction commands only for a new destination. Reuse an existing verified copy.
Original H50 files are not required once this complete export has been transferred.
The loader uses relative episode/video paths; personal absolute paths in provenance are historical references.

## 3. Download source weights and simulator assets

In synutility-dp-pretrained on the4090:

    python -u scripts/inspect_robocasa_pretrained_dp.py --download-weights
    python -m robocasa.scripts.setup_macros
    python -m robocasa.scripts.download_kitchen_assets --type tex tex_generative fixtures_lw objs_lw

The weight command downloads only the1.70GB latest.ckpt, its normalizer and config metadata.
The official asset downloader prompts before downloading; these are simulator assets, not other tasks' demonstrations.
Existing complete assets can be reused. Selected assets match the lightwheel registry used in evaluation.

## 4. Start fine-tuning and then3 rollouts with explicit output

One command, launched by you:

    bash scripts/run_h50_native_dp_finetune_and_eval.sh

The script stops immediately if any stage fails. Order:

1. Real RoboCasa seed3 reset/camera/action/EGL check (no env steps).
2. Native fine-tuning preflight (no backward/optimizer).
3. Fine-tuning exactly10000 optimizer updates.
4. Load the final EMA checkpoint and run seed3,4,5 sequentially.
5. Print per-episode and final success statistics; save all3 videos and per-step action logs.

Fine-tuning defaults:

| Parameter | Value |
| --- | --- |
| Target dataset |28 H50 human left demos, no synthetic |
| Camera count/input size |3 /256x256, native224x224 crop |
| FPS/control frequency |20 |
| Batch size |4 |
| Optimizer updates |10000 |
| Learning rate |1e-5,500-step warmup then cosine |
| Seed |0 |
| Precision |Float32, native trainer |
| Initialization |Published EMA into both model and EMA |
| Optimizer/counters |Fresh |
| Normalizer |Published checkpoint normalizer preserved |
| Action horizon |8 executed actions /10 predicted |
| Observation history |2 |
| Denoising |100 native DDPM steps |
| During-training rollouts |Disabled |
| Logging service |W&B disabled; local metrics/logs saved |

The official trainer's global_step includes epoch logging increments.
metrics.jsonl step and checkpoint names use the exact optimizer-update counter instead.

Training prints TRAIN_STEP at the first3 updates and every100 updates.
The first actual update prints per-camera gradient checks.
Checkpoint saves print CHECKPOINT_SAVED.
Evaluation prints PROGRESS, EPISODE_STATS, EVAL_PROGRESS and FINAL_EVAL.

## Output files

    outputs/OpenDrawer/dp_pretrained_human_left/seed_0/
      config.yaml
      config.json
      normalizer.pkl
      metrics.jsonl
      logs.json.txt              # native trainer's logging format
      summary.json
      checkpoint_status.json
      checkpoints/
        002500.ckpt
        005000.ckpt
        007500.ckpt
        010000.ckpt
        last.ckpt ->010000.ckpt
        latest.ckpt ->010000.ckpt
      robocasa_eval/seed_3_4_5/
        summary.json
        episodes.json
        seed_3/{rollout.mp4,steps.jsonl,summary.json,protocol.json}
        seed_4/{rollout.mp4,steps.jsonl,summary.json,protocol.json}
        seed_5/{rollout.mp4,steps.jsonl,summary.json,protocol.json}

Console logs:

    outputs/OpenDrawer/dp_pretrained_human_left/logs/train_seed_0.log
    outputs/OpenDrawer/dp_pretrained_human_left/logs/eval_seed_3_4_5.log

The native .ckpt contains model, EMA, optimizer, full config and counters.
The checkpoint contains normalization tensors; normalizer.pkl is also saved and checked by evaluation.
This is the official DP format, not a bare state_dict or a LeRobot ACT checkpoint.

Fine-tuned inference uses the H50 literal language text "open the left drawer",
matching the training embedding. Environment instruction "Open the left drawer." is checked
for task identity and reported separately. Zero-shot pretrained inference retains the official
environment text. Actions/temporal convention are unchanged.

Existing output directories are protected. A failed job leaves failure.json and completed periodic
checkpoints. The runner does not automatically resume or overwrite; use a new --output for a new run.

## Separate commands if preferred

Preflight only:

    python -u scripts/finetune_h50_native_dp.py --preflight

Train only:

    python -u scripts/finetune_h50_native_dp.py

Evaluate final checkpoint only:

    python -u scripts/evaluate_h50_native_dp_three.py

The evaluator requires10000 actual updates by default and checks saved normalizer consistency.
For an explicitly shorter engineering run, choose a separate output and pass its --expected-updates
when evaluating. Do not treat it as the formal10000-update run.

## Keep outputs when stopping the rental

Checkpoint files, normalizer, full config, metrics and rollout videos remain local and ignored by Git.
Copy the full run directory back before removing the rental machine. From local WSL:

    mkdir -p /home/dabozi/research/synthetic-utility-smoketest/outputs/OpenDrawer/dp_pretrained_human_left
    rsync -av --progress -e 'ssh -p PORT' \
      USER@HOST:~/research/synthetic-utility-smoketest/outputs/OpenDrawer/dp_pretrained_human_left/seed_0 \
      /home/dabozi/research/synthetic-utility-smoketest/outputs/OpenDrawer/dp_pretrained_human_left/

Use a new local destination if seed_0 already exists. Do not overwrite archived ACT/debug runs.
