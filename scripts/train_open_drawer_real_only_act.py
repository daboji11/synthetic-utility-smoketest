"""Run the real-only LeRobot ACT baseline or a separate 50-step smoke run."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiment/open_drawer_real_only_act.yaml"
DATASET = ROOT / "data/processed/OpenDrawer/real_16fps"
BASE_OUTPUT = ROOT / "outputs/OpenDrawer/real_only"
CAMERA = "observation.images.robot0_agentview_left_image"
REQUIRED_MODEL_FILES = (
    "model.safetensors", "config.json", "train_config.json",
    "policy_preprocessor.json", "policy_postprocessor.json",
)
LOSS = re.compile(r"\bloss:(\S+)")
LR = re.compile(r"\blr:(\S+)")


def verify_dataset() -> None:
    dataset = LeRobotDataset(
        "local/open_drawer_real_16fps", root=DATASET, video_backend="torchcodec"
    )
    if (dataset.num_episodes != 21 or dataset.meta.fps != 16
        or dataset.meta.camera_keys != [CAMERA]
        or tuple(dataset.features["action"]["shape"]) != (12,)
        or tuple(dataset.features["observation.state"]["shape"]) != (16,)):
        raise ValueError("Real-only 16fps dataset failed preflight")


def read_gpu_memory_mib() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True,
        )
        return int(result.stdout.splitlines()[0].strip())
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def sync_checkpoint_aliases(output: Path, steps: int, save_freq: int) -> None:
    checkpoints = output / "checkpoints"
    for step in range(save_freq, steps + 1, save_freq):
        padded = checkpoints / f"{step:06d}"
        alias = checkpoints / str(step)
        if (alias != padded and not alias.exists()
            and all((padded / "pretrained_model" / name).is_file()
                    for name in REQUIRED_MODEL_FILES)):
            alias.symlink_to(padded.name, target_is_directory=True)


def verify_checkpoint(output: Path, steps: int, save_freq: int) -> None:
    checkpoints = output / "checkpoints"
    sync_checkpoint_aliases(output, steps, save_freq)
    for step in range(save_freq, steps + 1, save_freq):
        padded = checkpoints / f"{step:06d}"
        if not padded.is_dir():
            raise FileNotFoundError(f"Missing LeRobot checkpoint: {padded}")
    last = checkpoints / "last" / "pretrained_model"
    missing = [name for name in REQUIRED_MODEL_FILES if not (last / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete final checkpoint: {missing}")
    shutil.copy2(last / "train_config.json", output / "train_config.json")
    shutil.copy2(CONFIG, output / "requested_config.yaml")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run 50 steps in a separate output directory")
    parser.add_argument("--batch-size", type=int, choices=(2, 4), default=4)
    args = parser.parse_args()
    verify_dataset()
    steps = 50 if args.smoke else 10000
    save_freq = 50 if args.smoke else 2500
    log_freq = 1 if args.smoke else 10
    name = (f"smoke_seed_0_batch_{args.batch_size}" if args.smoke
            else ("seed_0" if args.batch_size == 4 else "seed_0_batch_2"))
    output = BASE_OUTPUT / name
    if output.exists():
        raise FileExistsError(f"Preserving existing run: {output}")
    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)
    temporary_log = BASE_OUTPUT / f".{name}.train.log"
    temporary_metrics = BASE_OUTPUT / f".{name}.metrics.jsonl"
    command = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        "--config_path", str(CONFIG),
        "--output_dir", str(output),
        "--batch_size", str(args.batch_size),
        "--steps", str(steps),
        "--save_freq", str(save_freq),
        "--log_freq", str(log_freq),
    ]
    started = time.monotonic()
    metric_index = 0
    with temporary_log.open("w") as log, temporary_metrics.open("w") as metrics:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            if "step:" in line:
                sync_checkpoint_aliases(output, steps, save_freq)
            if "Traceback" in line or "Error" in line or "step:" in line:
                print(line.rstrip(), flush=True)
            loss_match, lr_match = LOSS.search(line), LR.search(line)
            if loss_match and lr_match and "step:" in line:
                metric_index += 1
                loss = float(loss_match.group(1))
                lr = float(lr_match.group(1))
                if not math.isfinite(loss) or not math.isfinite(lr):
                    process.terminate()
                    raise ValueError("Non-finite training loss or learning rate")
                row = {"step": metric_index * log_freq, "train_loss": loss,
                       "learning_rate": lr,
                       "wall_clock_s": round(time.monotonic() - started, 2)}
                if metric_index == 1 or metric_index % 10 == 0:
                    row["gpu_memory_mib"] = read_gpu_memory_mib()
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
        return_code = process.wait()
        metrics.write(json.dumps({"event": "run_finished",
                                  "wall_clock_s": round(time.monotonic() - started, 2),
                                  "exit_code": return_code}) + "\n")
    if output.is_dir():
        shutil.move(str(temporary_log), output / "train.log")
        shutil.move(str(temporary_metrics), output / "metrics.jsonl")
    if return_code:
        raise SystemExit(f"LeRobot training failed with exit code {return_code}; see {output}")
    if metric_index == 0:
        raise RuntimeError("No train_loss and learning_rate entries were captured")
    verify_checkpoint(output, steps, save_freq)
    print(f"validated {metric_index} finite loss entries and checkpoint: {output}")


if __name__ == "__main__":
    main()
