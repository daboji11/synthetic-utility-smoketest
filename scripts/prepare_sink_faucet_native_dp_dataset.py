#!/usr/bin/env python3
"""Export the 54 original 20 Hz H50 faucet demos as a 3-camera OpenPI dataset."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.datasets import RoboCasaLeRobotDataset  # noqa: E402

RAW_CACHE = ROOT / "data/raw/huggingface"
TARGET = ROOT / "data/processed/TurnOnSinkFaucet/h50_human_native_dp_20fps_256"
STAGING = TARGET.with_name(TARGET.name + ".incomplete")
PROMPT = "turn on the sink faucet"
SOURCE_REPO = "DAVIAN-Robotics/robocasa-H50"
SOURCE_IDS = tuple(range(1132, 1186))
TOTAL_FRAMES = 12070
CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
SOURCE_CAMERAS = tuple(f"observation.images.{camera}_image" for camera in CAMERAS)
COLUMNS = ("action", "observation.state", "timestamp", "frame_index",
           "episode_index", "index", "task_index")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def source_dataset() -> RoboCasaLeRobotDataset:
    source = RoboCasaLeRobotDataset(
        repo_id=SOURCE_REPO, source="human", task_id="TurnOnSinkFaucet",
        task_label=PROMPT, camera_key=SOURCE_CAMERAS[0],
        cache_dir=RAW_CACHE, offline=True)
    ids = tuple(int(item) for item in source.get_episode_ids())
    if ids != SOURCE_IDS or source.info["fps"] != 20:
        raise ValueError(f"Unexpected H50 Faucet source IDs/FPS: {ids}")
    if set(source.camera_keys) != set(SOURCE_CAMERAS):
        raise ValueError("Source camera keys changed")
    if sum(int(source.get_episode_metadata(i)["length"]) for i in ids) != TOTAL_FRAMES:
        raise ValueError("Source frame count changed")
    for source_id in ids:
        if source.get_episode_metadata(source_id)["tasks"] != [PROMPT]:
            raise ValueError(f"Unexpected task at episode {source_id}")
        for camera in SOURCE_CAMERAS:
            source.video_segment(source_id, camera)
    return source


def modality_metadata() -> dict:
    state = {"end_effector_position_relative": (0, 3),
             "end_effector_rotation_relative": (3, 7),
             "gripper_qpos": (7, 9), "base_position": (9, 12),
             "base_rotation": (12, 16)}
    action = {"end_effector_position": (0, 3), "end_effector_rotation": (3, 6),
              "gripper_close": (6, 7), "base_motion": (7, 11),
              "control_mode": (11, 12)}

    def fields(ranges: dict, column: str) -> dict:
        return {key: {"original_key": column, "start": first, "end": last,
                      "dtype": "float32", "absolute": True}
                for key, (first, last) in ranges.items()}

    return {"state": fields(state, "observation.state"),
            "action": fields(action, "action"),
            "video": {camera: {"original_key": f"observation.images.{camera}_image"}
                      for camera in CAMERAS},
            "annotation": {"human.task_description": {"original_key": "task_index"}}}


def video_clip(source_path: Path, start: float, length: int, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", f"{start:.8f}",
        "-i", str(source_path), "-frames:v", str(length),
        "-vf", "scale=256:256:flags=bilinear", "-c:v", "libx264",
        "-threads", "2", "-crf", "0", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", str(destination),
    ], check=True)
    capture = cv2.VideoCapture(str(destination))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if count != length or abs(fps - 20) > 0.001:
            raise ValueError(f"Bad video length/FPS {destination}: {count}, {fps}")
        for frame_index in (0, length - 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame.shape != (256, 256, 3):
                raise ValueError(f"Cannot decode endpoint {frame_index}: {destination}")
    finally:
        capture.release()


def export() -> None:
    if TARGET.exists() or STAGING.exists():
        raise FileExistsError("Preserving existing Faucet dataset or staging directory")
    source = source_dataset()
    STAGING.mkdir(parents=True)
    features = {key: source.features[key].copy() for key in COLUMNS}
    for camera in CAMERAS:
        features[f"observation.images.{camera}_image"] = {
            "dtype": "video", "shape": [256, 256, 3],
            "names": ["height", "width", "channel"],
            "info": {"video.fps": 20, "video.channels": 3,
                     "video.codec": "h264", "video.pix_fmt": "yuv420p"}}
    info = {"codebase_version": "v2.1", "fps": 20, "total_episodes": 54,
            "total_frames": TOTAL_FRAMES, "total_tasks": 1,
            "total_videos": 162, "chunks_size": 1000, "features": features,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"}
    write_json(STAGING / "meta/info.json", info)
    write_json(STAGING / "meta/modality.json", modality_metadata())
    (STAGING / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": PROMPT}) + "\n")
    table_cache: dict[str, pa.Table] = {}
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    episode_rows = []
    provenance_rows = []
    total = 0
    for new_index, source_id in enumerate(SOURCE_IDS):
        episode = source.get_episode_metadata(source_id)
        length = int(episode["length"])
        relative = source.info["data_path"].format(
            chunk_index=episode["data/chunk_index"],
            file_index=episode["data/file_index"])
        if relative not in table_cache:
            table_cache[relative] = pq.read_table(source._download(relative), columns=list(COLUMNS))
        raw = table_cache[relative].slice(int(episode["dataset_from_index"]), length)
        if raw.num_rows != length or set(raw["episode_index"].to_pylist()) != {source_id}:
            raise ValueError(f"Source parquet episode mismatch: {source_id}")
        states = np.asarray(raw["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(raw["action"].to_pylist(), dtype=np.float32)
        if states.shape != (length, 16) or actions.shape != (length, 12):
            raise ValueError(f"State/action shape mismatch: {source_id}")
        times = np.asarray(raw["timestamp"].to_pylist())
        if not np.allclose(times, np.arange(length) / 20, rtol=0, atol=1e-4):
            raise ValueError(f"Source 20 Hz timestamps changed: {source_id}")
        if raw["frame_index"].to_pylist() != list(range(length)):
            raise ValueError(f"Source frame order changed: {source_id}")
        new_table = pa.Table.from_arrays([
            raw["action"].combine_chunks(), raw["observation.state"].combine_chunks(),
            raw["timestamp"].combine_chunks(), raw["frame_index"].combine_chunks(),
            pa.array([new_index] * length, type=pa.int64()),
            pa.array(range(total, total + length), type=pa.int64()),
            pa.array([0] * length, type=pa.int64()),
        ], names=COLUMNS)
        parquet = STAGING / f"data/chunk-000/episode_{new_index:06d}.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new_table, parquet)
        check = pq.read_table(parquet)
        for key in ("action", "observation.state", "timestamp", "frame_index"):
            if not check[key].equals(raw[key]):
                raise ValueError(f"Numeric source values changed: {source_id}/{key}")
        camera_sources = {}
        for camera in SOURCE_CAMERAS:
            source_video, start, _ = source.video_segment(source_id, camera)
            if abs(start * 20 - round(start * 20)) > 1e-3:
                raise ValueError(f"Camera start is not 20 Hz aligned: {source_id}/{camera}")
            output_video = STAGING / f"videos/chunk-000/{camera}/episode_{new_index:06d}.mp4"
            video_clip(source_video, start, length, output_video)
            camera_sources[camera] = {"source_video": str(source_video),
                                      "start_seconds": start}
        all_states.append(states)
        all_actions.append(actions)
        episode_rows.append({"episode_index": new_index, "tasks": [PROMPT], "length": length})
        provenance_rows.append({"episode_index": new_index,
                                "source_episode_index": source_id, "length": length,
                                "camera_sources": camera_sources})
        total += length
        print(f"EXPORT episode={new_index + 1}/54 source={source_id} "
              f"frames={length} total={total} cameras=3", flush=True)
    if total != TOTAL_FRAMES:
        raise ValueError(f"Unexpected exported frame count: {total}")
    stats = {}
    for key, arrays in (("observation.state", all_states), ("action", all_actions)):
        values = np.concatenate(arrays).astype(np.float64)
        stats[key] = {name: value.tolist() for name, value in (
            ("min", values.min(0)), ("max", values.max(0)),
            ("mean", values.mean(0)), ("std", values.std(0)),
            ("q01", np.quantile(values, .01, axis=0)),
            ("q99", np.quantile(values, .99, axis=0)))}
    write_json(STAGING / "meta/stats.json", stats)
    (STAGING / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episode_rows))
    write_json(STAGING / "provenance_manifest.json", {
        "source_repo": SOURCE_REPO, "source_revision": source.revision,
        "source_type": "human", "source_instruction": PROMPT,
        "source_fps": 20, "target_fps": 20, "total_episodes": 54,
        "total_frames": TOTAL_FRAMES, "source_episode_ids": list(SOURCE_IDS),
        "camera_order": list(SOURCE_CAMERAS), "action_dim": 12,
        "state_dim": 16, "all_numeric_values_preserved": True,
        "all_162_videos_endpoints_decoded": True,
        "image_transform": "128x128 to 256x256 bilinear; no recovered detail",
        "episodes": provenance_rows,
    })
    STAGING.rename(TARGET)
    print(f"DATASET_READY {TARGET} episodes=54 frames={TOTAL_FRAMES} "
          "fps=20 cameras=3 action=12 state=16", flush=True)


if __name__ == "__main__":
    export()
