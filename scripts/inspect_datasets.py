"""Inspect the paired LeRobot v3 datasets and export single-camera previews."""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.datasets import RoboCasaLeRobotDataset


def load_dataset_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config.get("task_id"), str) or not config["task_id"]:
        raise ValueError("A task_id is required")
    return config


def inspect_schema(dataset: RoboCasaLeRobotDataset, seed: int) -> dict:
    episode_id = random.Random(seed).choice(dataset.get_episode_ids())
    metadata = dataset.get_episode_metadata(episode_id)
    trajectory = dataset.get_trajectory(episode_id)
    action_dim = len(trajectory["actions"][0])
    state_dim = len(trajectory["observations"][0])
    expected_action = dataset.features["action"]["shape"][-1]
    expected_state = dataset.features["observation.state"]["shape"][-1]
    if (action_dim, state_dim) != (expected_action, expected_state):
        raise ValueError(f"Episode data contradicts declared feature shapes in {dataset.repo_id}")
    return {
        "repo_id": dataset.repo_id,
        "revision": dataset.revision,
        "episode_id": episode_id,
        "episode_length": metadata["length"],
        "observation_keys": sorted(key for key in dataset.features if key.startswith("observation.")),
        "camera_keys": dataset.camera_keys,
        "camera_shapes": {key: dataset.features[key]["shape"] for key in dataset.camera_keys},
        "selected_camera": dataset.camera_key,
        "selected_image_resolution": dataset.features[dataset.camera_key]["shape"][:2],
        "action_shape": [len(trajectory["actions"]), action_dim],
        "state_shape": [len(trajectory["observations"]), state_dim],
        "action_dim": action_dim,
        "state_dim": state_dim,
        "action_feature_names": dataset.features["action"].get("names"),
        "state_feature_names": dataset.features["observation.state"].get("names"),
        "action_dtype": dataset.features["action"]["dtype"],
        "state_dtype": dataset.features["observation.state"]["dtype"],
        "fps": dataset.info["fps"],
        "language_or_task": metadata.get("tasks"),
        "canonical_task_id": trajectory["task_id"],
        "success_metadata": None,
        "source_metadata": dataset.source,
        "note": "next.done is an episode boundary, not a task-success label",
    }


def compare_schemas(real: RoboCasaLeRobotDataset, synthetic: RoboCasaLeRobotDataset,
                    real_schema: dict, synthetic_schema: dict) -> dict:
    checks = {
        "action_dim_match": real_schema["action_dim"] == synthetic_schema["action_dim"],
        "state_dim_match": real_schema["state_dim"] == synthetic_schema["state_dim"],
        "action_dtype_match": real_schema["action_dtype"] == synthetic_schema["action_dtype"],
        "state_dtype_match": real_schema["state_dtype"] == synthetic_schema["state_dtype"],
        "action_feature_names_match": real_schema["action_feature_names"] == synthetic_schema["action_feature_names"],
        "state_feature_names_match": real_schema["state_feature_names"] == synthetic_schema["state_feature_names"],
        "selected_camera_present_in_both": real.camera_key in real.camera_keys and synthetic.camera_key in synthetic.camera_keys,
        "selected_camera_key_match": real.camera_key == synthetic.camera_key,
        "selected_camera_shape_match": real.features[real.camera_key]["shape"] == synthetic.features[synthetic.camera_key]["shape"],
        "fps_match": real_schema["fps"] == synthetic_schema["fps"],
        "task_class_match": real_schema["canonical_task_id"] == synthetic_schema["canonical_task_id"],
        "instruction_text_match": real_schema["language_or_task"] == synthetic_schema["language_or_task"],
    }
    common_cameras = sorted(set(real.camera_keys) & set(synthetic.camera_keys))
    checks["common_camera_shapes_match"] = all(real.features[key]["shape"] == synthetic.features[key]["shape"]
                                               for key in common_cameras)
    common_tasks = sorted(set(real.get_task_ids()) & set(synthetic.get_task_ids()))
    structural = all(value for key, value in checks.items()
                     if key not in {"fps_match", "instruction_text_match"}) and bool(common_tasks)
    if not structural:
        verdict = "mismatch: not ready for the same ACT training pipeline"
    elif not checks["fps_match"]:
        verdict = "schema compatible, but temporal resampling or explicit FPS handling is required before joint ACT training"
    else:
        verdict = "schema compatible for joint ACT input; task success and action quality remain unverified"
    return {"checks": checks, "common_camera_keys": common_cameras,
            "common_tasks": common_tasks, "verdict": verdict}


def export_previews(dataset: RoboCasaLeRobotDataset, task: str, output: Path,
                    count: int, seed: int) -> list[dict]:
    task_dataset = dataset.filter_by_task(task)
    ids = random.Random(seed).sample(task_dataset.get_episode_ids(),
                                   min(count, len(task_dataset)))
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for episode_id in ids:
        meta = task_dataset.get_episode_metadata(episode_id)
        video, start, duration = task_dataset.video_segment(episode_id)
        destination = output / f"{dataset.source}_episode_{episode_id}.mp4"
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                   "-ss", f"{start:.6f}", "-i", str(video), "-t", f"{duration:.6f}",
                   "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", str(destination)]
        subprocess.run(command, check=True)
        rows.append({"source": dataset.source, "episode_id": episode_id, "task_id": task,
                     "length": meta["length"], "action_dim": dataset.features["action"]["shape"][-1],
                     "state_dim": dataset.features["observation.state"]["shape"][-1],
                     "instruction": " | ".join(meta["tasks"]),
                     "video_path": str(destination.resolve())})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dataset/pnp_counter_to_cab.yaml")
    parser.add_argument("--real-count", type=int, default=5)
    parser.add_argument("--synthetic-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-video", action="store_true", help="Inspect metadata and episode arrays without downloading MP4")
    parser.add_argument("--offline", action="store_true", help="Use only files already present in the local Hugging Face cache")
    args = parser.parse_args()
    if args.real_count < 1 or args.synthetic_count < 1:
        parser.error("preview counts must be positive")
    config = load_dataset_config(args.config)
    task = config["task_id"]
    output = ROOT / "results/dataset_inspection" / task

    datasets = {}
    errors = {}
    for label in ("real", "synthetic"):
        try:
            datasets[label] = RoboCasaLeRobotDataset(**config[label], task_id=task,
                camera_key=config["camera_key"],
                cache_dir=ROOT / "data/raw/huggingface", offline=args.offline)
            print(f"{label} dataset loaded: yes ({datasets[label].repo_id})")
        except Exception as exc:
            errors[label] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            print(f"{label} dataset loaded: no ({errors[label]})")

    report = {"task_id": task, "task_match_rules": {label: config[label].get("task_pattern") or config[label].get("task_label")
                                                   for label in ("real", "synthetic")},
              "loaded": {label: label in datasets for label in ("real", "synthetic")},
              "errors": errors, "episode_counts": {"real": None, "synthetic": None},
              "instruction_variants": {}, "schemas": {}, "compatibility": None, "preview_rows": []}
    for label, dataset in datasets.items():
        report["episode_counts"][label] = len(dataset)
        report["instruction_variants"][label] = sorted({instruction for episode_id in dataset.get_episode_ids()
                                                         for instruction in dataset.get_episode_metadata(episode_id)["tasks"]})
        print(f"{label} {task} episodes: {len(dataset)}")
        print(f"{label} instruction variants: {len(report['instruction_variants'][label])}")
        if dataset:
            report["schemas"][label] = inspect_schema(dataset, args.seed)
            print(f"{label} sample schema: {json.dumps(report['schemas'][label], ensure_ascii=False)}")
        else:
            print(f"[WARNING] No {task} episodes matched in {label} episode metadata")

    real, synthetic = datasets.get("real"), datasets.get("synthetic")
    if real and synthetic:
        report["compatibility"] = compare_schemas(real, synthetic, report["schemas"]["real"],
                                                   report["schemas"]["synthetic"])
        print(f"\n{task}: real={len(real)}, synthetic={len(synthetic)}")
        for key, value in report["compatibility"]["checks"].items():
            print(f"[{'OK' if value else 'WARNING'}] {key}: {value}")
        print("Compatibility verdict:", report["compatibility"]["verdict"])
    else:
        print("\nPaired task: unverified (both datasets and task episodes are required)")
        print("Compatibility verdict: unverified")

    if not args.skip_video:
        for label, count in (("real", args.real_count), ("synthetic", args.synthetic_count)):
            dataset = datasets.get(label)
            if dataset:
                report["preview_rows"].extend(export_previews(dataset, task, output / label, count, args.seed))
        output.mkdir(parents=True, exist_ok=True)
        with (output / "episodes.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source", "episode_id", "length",
                                                      "instruction", "video_path"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(report["preview_rows"])
        print(f"Preview CSV: {output / 'episodes.csv'}")

    report_path = output / "inspection.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Inspection report: {report_path}")
    return 0 if real and synthetic else 2


if __name__ == "__main__":
    raise SystemExit(main())
