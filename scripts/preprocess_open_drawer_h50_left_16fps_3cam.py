"""Prepare the same 28 H50 left-drawer episodes at 16 fps with all three cameras."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torchcodec.decoders import VideoDecoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.datasets import RoboCasaLeRobotDataset
from preprocess_open_drawer_h50_left_16fps import selected_indices

CAMERAS = [
    "observation.images.robot0_agentview_left_image",
    "observation.images.robot0_agentview_right_image",
    "observation.images.robot0_eye_in_hand_image",
]
REPO_ID = "local/open_drawer_h50_human_left_16fps_3cam"
RAW_CACHE = ROOT / "data/raw/huggingface"
REFERENCE = ROOT / "data/processed/OpenDrawer/h50_human_left_16fps"
DESTINATION = ROOT / "data/processed/OpenDrawer/h50_human_left_16fps_3cam"
STAGING = DESTINATION.with_name(DESTINATION.name + ".incomplete")
INSTRUCTION = "open the left drawer"


def parquet_values(path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    tables = [pq.read_table(file, columns=columns)
              for file in sorted((path / "data").glob("**/*.parquet"))]
    table = pa.concat_tables(tables)
    return {key: np.asarray(table.column(key).to_pylist()) for key in columns}


def verify_and_publish() -> None:
    manifest = json.loads((STAGING / "resampling_manifest.json").read_text())
    reference_manifest = json.loads((REFERENCE / "resampling_manifest.json").read_text())
    verified = LeRobotDataset(REPO_ID, root=STAGING, video_backend="torchcodec")
    if (verified.num_episodes != 28 or verified.meta.fps != 16
        or verified.meta.camera_keys != CAMERAS
        or list(verified.meta.tasks.index) != [INSTRUCTION]
        or tuple(verified.features["action"]["shape"]) != (12,)
        or tuple(verified.features["observation.state"]["shape"]) != (16,)):
        raise ValueError("Three-camera dataset schema mismatch")
    expected_frames = sum(item["target_length"] for item in manifest["episodes"])
    if len(verified) != expected_frames or len(manifest["episodes"]) != 28:
        raise ValueError("Episode/frame count mismatch")
    for item, ref in zip(manifest["episodes"], reference_manifest["episodes"], strict=True):
        for key in ("new_episode_index", "source_episode_index", "source_length",
                    "target_length", "source_indices"):
            if item[key] != ref[key]:
                raise ValueError(f"Human base or resampling map differs: {key}")
    columns = ["episode_index", "frame_index", "timestamp", "observation.state", "action"]
    actual, reference = parquet_values(STAGING, columns), parquet_values(REFERENCE, columns)
    for key in columns:
        if not np.array_equal(actual[key], reference[key]):
            raise ValueError(f"Three-camera {key} differs from single-camera baseline")
    offset = 0
    for item in manifest["episodes"]:
        length = item["target_length"]
        for local_index in (0, length - 1):
            frame = verified[offset + local_index]
            if (int(frame["episode_index"]) != item["new_episode_index"]
                or abs(float(frame["timestamp"]) - local_index / 16) > 1e-4
                or any(tuple(frame[key].shape) != (3, 128, 128) for key in CAMERAS)):
                raise ValueError("Decoded camera/frame alignment mismatch")
        offset += length
    manifest["verification"] = {
        "episodes": 28, "fps": 16, "frames": expected_frames,
        "all_camera_endpoints_decoded": True,
        "same_human_episodes_and_source_indices_as_single_camera": True,
        "state_action_timestamps_identical_to_single_camera": True,
    }
    (STAGING / "resampling_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    STAGING.rename(DESTINATION)
    print(f"VERIFIED episodes=28 fps=16 frames={expected_frames} action_dim=12 state_dim=16 "
          f"cameras={CAMERAS} resolution=128x128", flush=True)
    print(f"DATASET {DESTINATION}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    if not REFERENCE.is_dir():
        raise FileNotFoundError("Existing single-camera human-left dataset is required for comparison")
    if args.finalize_existing:
        if DESTINATION.exists() or not STAGING.is_dir():
            raise FileNotFoundError("Expected only completed staging dataset")
        verify_and_publish()
        return
    if DESTINATION.exists() or STAGING.exists():
        raise FileExistsError("Preserving existing three-camera dataset or staging")
    source = RoboCasaLeRobotDataset(
        repo_id="DAVIAN-Robotics/robocasa-H50", source="human", task_id="OpenDrawer",
        task_label=INSTRUCTION, camera_key=CAMERAS[0], cache_dir=RAW_CACHE, offline=True,
    )
    if (len(source) != 28 or source.info["fps"] != 20
        or any(source.get_episode_metadata(eid)["tasks"] != [INSTRUCTION]
               for eid in source.get_episode_ids())
        or set(source.camera_keys) != set(CAMERAS)):
        raise ValueError("Unexpected raw H50 human-left source")
    # Download only shared video files referenced by these 28 selected episodes.
    # Use the exact metadata revision already cached, including for the extra cameras.
    video_paths = set()
    for eid in source.get_episode_ids():
        episode = source.get_episode_metadata(eid)
        for camera in CAMERAS:
            prefix = f"videos/{camera}"
            video_paths.add(source.info["video_path"].format(
                video_key=camera, chunk_index=episode[f"{prefix}/chunk_index"],
                file_index=episode[f"{prefix}/file_index"]))
    for relative_path in sorted(video_paths):
        if not (source.snapshot / relative_path).is_file():
            print(f"DOWNLOAD revision={source.revision} file={relative_path}", flush=True)
            hf_hub_download(repo_id=source.repo_id, filename=relative_path, repo_type="dataset",
                            revision=source.revision, cache_dir=RAW_CACHE)
        print(f"RAW_CAMERA_READY {relative_path}", flush=True)
    features = {
        "action": source.features["action"].copy(),
        "observation.state": source.features["observation.state"].copy(),
        **{key: {"dtype": "video", "shape": [128, 128, 3],
                 "names": ["height", "width", "channel"]} for key in CAMERAS},
    }
    writer = LeRobotDataset.create(
        repo_id=REPO_ID, root=STAGING, fps=16, features=features,
        robot_type=source.info.get("robot_type"), use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p",
                                     g=16, crf=18, preset="veryfast"),
    )
    manifest = {
        "source_repo": source.repo_id, "source_revision": source.revision,
        "source_type": "human", "source_instruction": INSTRUCTION,
        "source_fps": 20, "target_fps": 16,
        "resampling": "nearest timestamp j/16, ties to earlier source frame",
        "camera_keys": CAMERAS, "single_camera_reference": str(REFERENCE), "episodes": [],
    }
    decoders = {}
    timestamp_tables = {}
    for new_episode_index, source_id in enumerate(source.get_episode_ids()):
        episode = source.get_episode_metadata(source_id)
        length, start = int(episode["length"]), int(episode["dataset_from_index"])
        indices = selected_indices(length)
        data_path = source.info["data_path"].format(
            chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"])
        if data_path not in timestamp_tables:
            timestamp_tables[data_path] = pq.read_table(
                source._download(data_path), columns=["timestamp"]).column("timestamp").to_numpy()
        source_times = timestamp_tables[data_path][start:start + length]
        if len(source_times) != length or not np.allclose(
                source_times, np.arange(length) / 20, rtol=0, atol=1e-4):
            raise ValueError(f"Unexpected source timestamps for episode {source_id}")
        images, camera_sources = {}, {}
        for camera in CAMERAS:
            video, video_start, _ = source.video_segment(source_id, camera)
            first_frame = round(video_start * 20)
            if abs(first_frame / 20 - video_start) > 1e-3:
                raise ValueError("Non-frame-aligned video start")
            if video not in decoders:
                decoders[video] = VideoDecoder(str(video))
            frames = decoders[video].get_frames_at([first_frame + index for index in indices])
            if not np.allclose(frames.pts_seconds.numpy(),
                               video_start + np.asarray(indices) / 20, rtol=0, atol=1e-3):
                raise ValueError(f"Camera timestamps misaligned: episode={source_id} camera={camera}")
            images[camera] = frames.data.permute(0, 2, 3, 1).numpy()
            camera_sources[camera] = {"source_video_file": str(video),
                                      "source_video_start_seconds": video_start}
        trajectory = source.get_trajectory(source_id)
        states = np.asarray(trajectory["observations"], dtype=np.float32)
        actions = np.asarray(trajectory["actions"], dtype=np.float32)
        if states.shape != (length, 16) or actions.shape != (length, 12):
            raise ValueError("Unexpected state/action shape")
        for j, index in enumerate(indices):
            writer.add_frame({
                "observation.state": states[index], "action": actions[index],
                **{key: images[key][j] for key in CAMERAS}, "task": INSTRUCTION,
            })
        writer.save_episode()
        manifest["episodes"].append({
            "new_episode_index": new_episode_index, "source_episode_index": int(source_id),
            "source_length": length, "target_length": len(indices),
            "source_indices": indices, "camera_sources": camera_sources,
        })
        print(f"EPISODE {new_episode_index + 1}/28 source={source_id} "
              f"frames={length}->{len(indices)} synchronized_cameras=3", flush=True)
    writer.finalize()
    (STAGING / "resampling_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_and_publish()


if __name__ == "__main__":
    main()
