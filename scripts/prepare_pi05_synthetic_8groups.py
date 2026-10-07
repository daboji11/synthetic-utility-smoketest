#!/usr/bin/env python3
"""Build three fixed 20 Hz Pi0.5 datasets from pinned 16 Hz DreamGen episodes.

Synthetic arm OSC_POSE controls are per-step deltas. Linear time interpolation
followed by 16/20 scaling preserves approximate commanded arm motion per
second; discrete gripper and control-mode fields are held. This conversion is
an approximation, recorded in provenance, and is identical across groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / ("data/raw/huggingface/"
    "datasets--Keh0t0--dreamgen_robocasa_p3_init0_v2_droid/snapshots/"
    "686cd91d4502b0606e473b07adb73c920b05d4e3")
CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
SOURCE_FPS, TARGET_FPS = 16, 20
STATE_SLICES = {
    "end_effector_position_relative": (0, 3),
    "end_effector_rotation_relative": (3, 7),
    "gripper_qpos": (7, 9),
    "base_position": (9, 12),
    "base_rotation": (12, 16),
}
ACTION_SLICES = {
    "end_effector_position": (0, 3),
    "end_effector_rotation": (3, 6),
    "gripper_close": (6, 7),
    "base_motion": (7, 11),
    "control_mode": (11, 12),
}

def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

def interpolate(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    old = np.arange(len(values), dtype=np.float64)
    return np.stack([np.interp(positions, old, values[:, j])
                     for j in range(values.shape[1])], axis=1).astype(np.float32)

def transform_numeric(raw_action: np.ndarray, raw_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = len(raw_action)
    target_length = round(length * TARGET_FPS / SOURCE_FPS)
    positions = np.minimum(np.arange(target_length) * SOURCE_FPS / TARGET_FPS, length - 1)
    action = interpolate(raw_action, positions)
    action[:, :6] *= SOURCE_FPS / TARGET_FPS
    nearest = np.minimum(np.rint(positions).astype(int), length - 1)
    action[:, 6] = raw_action[nearest, 6]
    action[:, 11] = raw_action[nearest, 11]
    state_raw = raw_state.copy()
    for start, stop in ((3, 7), (12, 16)):
        quat = state_raw[:, start:stop]
        for i in range(1, len(quat)):
            if np.dot(quat[i - 1], quat[i]) < 0:
                quat[i] *= -1
    state = interpolate(state_raw, positions)
    for start, stop in ((3, 7), (12, 16)):
        quat = state[:, start:stop]
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    if (action.shape != (target_length, 12) or state.shape != (target_length, 16)
        or not np.isfinite(action).all() or not np.isfinite(state).all()
        or np.max(np.abs(action)) > 1.0001
        or not np.all(np.isin(action[:, (6, 11)], (-1, 1)))):
        raise ValueError("Resampled action/state integrity failure")
    return action, state

def video_stats(path: Path) -> dict:
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,avg_frame_rate,width,height",
        "-of", "json", str(path)], text=True))["streams"][0]

def convert_video(source: Path, offset: float, source_frames: int,
                  target_frames: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".building.mp4")
    if output.exists():
        raise FileExistsError(f"Preserving existing output: {output}")
    try:
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-ss", f"{offset:.9f}", "-i", str(source),
            "-vf", f"trim=end_frame={source_frames},setpts=PTS-STARTPTS,"
                   "fps=20,scale=256:256:flags=bilinear",
            "-frames:v", str(target_frames), "-an", "-c:v", "libx264",
            "-threads", "2", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp)],
            check=True)
        stats = video_stats(temp)
        num, den = map(int, stats["avg_frame_rate"].split("/"))
        if ((stats["width"], stats["height"], int(stats["nb_frames"]))
            != (256, 256, target_frames) or num / den != TARGET_FPS):
            raise ValueError(f"Video mismatch {output}: {stats}")
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)

def build_group(group: str, selected: list[int], selection: dict, meta: dict[int, dict],
                tables: dict[tuple[int, int], pa.Table], info: dict, output_root: Path) -> dict:
    output = output_root / group
    if output.exists():
        raise FileExistsError(f"Preserving existing group: {output}")
    output.mkdir(parents=True)
    features = {key: info["features"][key] for key in
                ("action", "observation.state", "timestamp", "frame_index",
                 "episode_index", "index", "task_index")}
    for cam in CAMERAS:
        features[f"observation.images.{cam}_image"] = {
            "dtype": "video", "shape": [256, 256, 3],
            "names": ["height", "width", "channel"],
            "info": {"video.fps": TARGET_FPS, "video.channels": 3,
                     "video.codec": "h264", "video.pix_fmt": "yuv420p"}}
    parts = []
    global_index = 0
    lengths = []
    for new_ep, source_ep in enumerate(selected):
        ep = meta[source_ep]
        if ep["tasks"] != ["open the left drawer"] or ep["length"] != 279:
            raise ValueError(f"Unexpected source episode {source_ep}")
        table = tables[(ep["data/chunk_index"], ep["data/file_index"])]
        raw = table.slice(ep["dataset_from_index"], ep["length"])
        if set(raw["episode_index"].to_pylist()) != {source_ep}:
            raise ValueError(f"Source parquet slice mismatch: {source_ep}")
        original_action = np.asarray(raw["action"].to_pylist(), dtype=np.float32)
        original_state = np.asarray(raw["observation.state"].to_pylist(), dtype=np.float32)
        original_time = np.asarray(raw["timestamp"].to_pylist(), dtype=np.float64)
        if (original_action.shape != (279, 12) or original_state.shape != (279, 16)
            or not np.allclose(original_time, np.arange(279) / 16, atol=1e-4)):
            raise ValueError(f"Source numeric schema mismatch: {source_ep}")
        action, state = transform_numeric(original_action, original_state)
        length = len(action)
        timestamps = np.arange(length, dtype=np.float64) / TARGET_FPS
        out_table = pa.table({
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 12)),
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 16)),
            "timestamp": pa.array(timestamps),
            "frame_index": pa.array(np.arange(length, dtype=np.int64)),
            "episode_index": pa.array(np.full(length, new_ep, dtype=np.int64)),
            "index": pa.array(np.arange(global_index, global_index + length, dtype=np.int64)),
            "task_index": pa.array(np.zeros(length, dtype=np.int64)),
        })
        parquet = output / f"data/chunk-000/episode_{new_ep:06d}.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(out_table, parquet)
        if not pq.read_table(parquet)["action"].equals(out_table["action"]):
            raise ValueError(f"Parquet round trip failure: {source_ep}")
        for cam in CAMERAS:
            key = f"observation.images.{cam}_image"
            prefix = f"videos/{key}/"
            start = ep[prefix + "from_timestamp"]
            stop = ep[prefix + "to_timestamp"]
            if abs((stop - start) * SOURCE_FPS - ep["length"]) > 1e-4:
                raise ValueError(f"Video episode interval mismatch: {source_ep}/{cam}")
            source_video = SNAPSHOT / info["video_path"].format(
                video_key=key, chunk_index=ep[prefix + "chunk_index"],
                file_index=ep[prefix + "file_index"])
            target_video = output / f"videos/chunk-000/{key}/episode_{new_ep:06d}.mp4"
            convert_video(source_video, start, ep["length"], length, target_video)
        lengths.append({"episode_index": new_ep, "tasks": ["open the left drawer"],
                        "length": length, "source_episode_index": source_ep})
        parts.append((action, state))
        global_index += length
        print(f"GROUP_EPISODE {group} {new_ep + 1}/8 source={source_ep} frames={length}", flush=True)

    write_json(output / "meta/info.json", {
        "codebase_version": "v2.1", "fps": TARGET_FPS,
        "total_episodes": len(selected), "total_frames": global_index,
        "total_tasks": 1, "total_videos": 3 * len(selected), "chunks_size": 1000,
        "features": features,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    })
    def slices(values: dict[str, tuple[int, int]], column: str) -> dict:
        return {name: {"original_key": column, "start": start, "end": end,
                       "dtype": "float32", "absolute": True}
                for name, (start, end) in values.items()}
    write_json(output / "meta/modality.json", {
        "state": slices(STATE_SLICES, "observation.state"),
        "action": slices(ACTION_SLICES, "action"),
        "video": {cam: {"original_key": f"observation.images.{cam}_image"}
                  for cam in CAMERAS},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    })
    stats = {}
    for key, values in (("action", np.concatenate([x[0] for x in parts])),
                        ("observation.state", np.concatenate([x[1] for x in parts]))):
        stats[key] = {name: value.tolist() for name, value in (
            ("min", values.min(0)), ("max", values.max(0)), ("mean", values.mean(0)),
            ("std", values.std(0)), ("q01", np.quantile(values, .01, axis=0)),
            ("q99", np.quantile(values, .99, axis=0)))}
    write_json(output / "meta/stats.json", stats)
    (output / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "open the left drawer"}) + "\n")
    (output / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in lengths))
    manifest = {
        "selection_sha256": hashlib.sha256(
            json.dumps(selection, sort_keys=True).encode()).hexdigest(),
        "selection_status": selection["selection_status"], "source_revision": SNAPSHOT.name,
        "group": group, "source_episode_indices": selected,
        "source_fps": SOURCE_FPS, "fps": TARGET_FPS, "episodes": len(selected),
        "frames": global_index, "cameras": list(CAMERAS),
        "images": "source 128x128 to 256x256 bilinear; fps=20 duplicates/interpolates timestamps",
        "action_time_conversion": "linear 16->20 Hz; arm OSC_POSE deltas 0.8x; discrete gripper/control nearest",
        "state_time_conversion": "linear 16->20 Hz; quaternion sign continuity and renormalization",
        "simulator_success_verified": False, "idm_action_semantics_verified": False,
        "instruction_distribution": {"open the left drawer": len(selected)},
    }
    write_json(output / "provenance_manifest.json", manifest)
    return {"group": group, "path": str(output), "episodes": len(selected),
            "frames": global_index, "source_episode_indices": selected}

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("selection", type=Path)
    p.add_argument("--output-root", type=Path)
    args = p.parse_args()
    selected = json.loads(args.selection.read_text())
    if selected["selection_status"] != "EXPLORATORY_HUMAN_OVERRIDES_AND_PROVISIONAL_VISUAL":
        raise ValueError("Expected frozen exploratory selection after manual overrides")
    if selected["source_revision"] != SNAPSHOT.name:
        raise ValueError("Unexpected DreamGen source revision")
    output_root = args.output_root or ROOT / (
        "data/processed/OpenDrawer/synthetic_8group_20fps_" +
        hashlib.sha256(args.selection.read_bytes()).hexdigest()[:8])
    info = json.loads((SNAPSHOT / "meta/info.json").read_text())
    if info["fps"] != SOURCE_FPS:
        raise ValueError("Source FPS changed")
    rows = pq.read_table(SNAPSHOT / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    meta = {int(row["episode_index"]): row for row in rows}
    keys = {(meta[episode]["data/chunk_index"], meta[episode]["data/file_index"])
            for group in selected["groups"].values()
            for episode in group["episode_indices"]}
    tables = {key: pq.read_table(SNAPSHOT / info["data_path"].format(
        chunk_index=key[0], file_index=key[1]),
        columns=["action", "observation.state", "timestamp", "episode_index"])
        for key in keys}
    results = [build_group(group, selected["groups"][group]["episode_indices"],
                           selected, meta, tables, info, output_root)
               for group in selected["group_order"]]
    write_json(output_root / "conversion_summary.json",
               {"selection": str(args.selection), "groups": results})
    print("SYNTHETIC_DATASETS_READY " + json.dumps(
        {"output": str(output_root), "groups": results}), flush=True)

if __name__ == "__main__":
    main()
