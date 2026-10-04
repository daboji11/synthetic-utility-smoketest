"""Download and verify only OpenDrawer raw shards used by the current repo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import av
import yaml

from src.datasets import RoboCasaLeRobotDataset


CONFIG = ROOT / "configs/dataset/open_drawer.yaml"
CACHE = ROOT / "data/raw/huggingface"
CAMERA = "observation.images.robot0_agentview_left_image"


def referenced_files(dataset: RoboCasaLeRobotDataset) -> tuple[set[str], set[str]]:
    data_files, video_files = set(), set()
    for episode_id in dataset.get_episode_ids():
        episode = dataset.get_episode_metadata(episode_id)
        data_files.add(dataset.info["data_path"].format(
            chunk_index=episode["data/chunk_index"],
            file_index=episode["data/file_index"],
        ))
        prefix = f"videos/{dataset.camera_key}"
        video_files.add(dataset.info["video_path"].format(
            video_key=dataset.camera_key,
            chunk_index=episode[f"{prefix}/chunk_index"],
            file_index=episode[f"{prefix}/file_index"],
        ))
    return data_files, video_files


def verify_dataset(label: str, spec: dict, expected_fps: int, offline: bool) -> dict:
    dataset = RoboCasaLeRobotDataset(
        **spec, task_id="OpenDrawer", camera_key=CAMERA, cache_dir=CACHE,
        offline=offline,
    )
    assert dataset.camera_key == CAMERA
    assert dataset.info["fps"] == expected_fps
    assert dataset.features["action"]["shape"][-1] == 12
    assert dataset.features["observation.state"]["shape"][-1] == 16
    assert dataset.features[CAMERA]["shape"] == [128, 128, 3]
    data_files, video_files = referenced_files(dataset)
    print(f"{label}: {len(dataset)} episodes, {len(data_files)} parquet, "
          f"{len(video_files)} left-camera videos", flush=True)
    paths = {}
    for filename in sorted(data_files | video_files):
        paths[filename] = dataset._download(filename)
        print(f"downloaded {label}: {filename}", flush=True)
    for episode_id in dataset.get_episode_ids():
        episode = dataset.get_episode_metadata(episode_id)
        trajectory = dataset.get_trajectory(episode_id)
        assert len(trajectory["actions"]) == episode["length"]
        assert len(trajectory["observations"]) == episode["length"]
        assert all(len(action) == 12 for action in trajectory["actions"])
        assert all(len(state) == 16 for state in trajectory["observations"])
        video, start, duration = dataset.video_segment(episode_id)
        assert video.is_file() and start >= 0 and duration > 0
    for filename in sorted(video_files):
        with av.open(str(paths[filename])) as container:
            stream = container.streams.video[0]
            assert (stream.width, stream.height) == (128, 128)
            assert float(stream.average_rate) == expected_fps
            assert next(container.decode(video=0), None) is not None
    print(f"verified {label}: all {len(dataset)} episodes", flush=True)
    return {
        "repo_id": dataset.repo_id,
        "revision": dataset.revision,
        "local_path": str(CACHE / f"datasets--{dataset.repo_id.replace('/', '--')}"),
        "episodes": len(dataset),
        "fps": expected_fps,
        "action_dim": 12,
        "state_dim": 16,
        "camera_key": dataset.camera_key,
        "image_shape": [128, 128, 3],
        "data_files": sorted(data_files),
        "video_files": sorted(video_files),
        "referenced_bytes": sum(paths[filename].stat().st_size for filename in paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text())
    assert config["task_id"] == "OpenDrawer"
    results = {
        "real": verify_dataset("real", config["real"], 20, args.offline),
        "synthetic": verify_dataset("synthetic", config["synthetic"], 16, args.offline),
    }
    output = ROOT / "data/raw/open_drawer_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"manifest: {output}", flush=True)


if __name__ == "__main__":
    main()
