#!/usr/bin/env python3
"""Check selected DreamGen episode metadata, state/action arrays and 3-camera previews; no GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / (
    "data/raw/huggingface/datasets--Keh0t0--dreamgen_robocasa_p3_init0_v2_droid/"
    "snapshots/686cd91d4502b0606e473b07adb73c920b05d4e3"
)
PREVIEWS = ROOT / "results/dataset_inspection/OpenDrawer/synthetic_3cam/manifest.json"
CAMERAS = (
    "observation.images.robot0_agentview_left_image",
    "observation.images.robot0_agentview_right_image",
    "observation.images.robot0_eye_in_hand_image",
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("selection", type=Path)
    args = p.parse_args()
    selection = json.loads(args.selection.read_text())
    if selection["source_revision"] != SNAPSHOT.name:
        raise ValueError("Selection and cached source revisions disagree")
    selected = sorted({int(i) for group in selection["groups"].values()
                       for i in group["episode_indices"]})
    if len(selected) < 8:
        raise ValueError("No eight-episode group selected")

    info = json.loads((SNAPSHOT / "meta/info.json").read_text())
    if info["fps"] != 16 or info["total_episodes"] != 1440:
        raise ValueError("Unexpected DreamGen source metadata")
    for field, shape in (("action", [12]), ("observation.state", [16])):
        if info["features"][field]["shape"] != shape:
            raise ValueError(f"Unexpected {field} shape")
    for camera in CAMERAS:
        if info["features"][camera]["shape"] != [128, 128, 3]:
            raise ValueError(f"Unexpected camera shape: {camera}")

    meta = pq.read_table(SNAPSHOT / "meta/episodes/chunk-000/file-000.parquet",
                         columns=["episode_index", "tasks", "length", "dataset_from_index",
                                  "data/chunk_index", "data/file_index"]).to_pylist()
    by_id = {int(row["episode_index"]): row for row in meta}
    preview_report = json.loads(PREVIEWS.read_text())
    by_preview = {int(row["episode_index"]): row for row in preview_report["previews"]}
    if preview_report["source_revision"] != SNAPSHOT.name:
        raise ValueError("Preview revision differs from source")

    grouped = {}
    for episode in selected:
        ep = by_id[episode]
        if ep["tasks"] != ["open the left drawer"] or ep["length"] != 279:
            raise ValueError(f"Unexpected task/duration for {episode}")
        key = (ep["data/chunk_index"], ep["data/file_index"])
        grouped.setdefault(key, []).append(episode)
    arrays = {}
    for (chunk, file), episodes in grouped.items():
        path = SNAPSHOT / f"data/chunk-{chunk:03d}/file-{file:03d}.parquet"
        table = pq.read_table(path, columns=["action", "observation.state",
                                             "timestamp", "frame_index", "episode_index"])
        for episode in episodes:
            ep = by_id[episode]
            start, length = int(ep["dataset_from_index"]), int(ep["length"])
            block = table.slice(start, length)
            if set(block["episode_index"].to_pylist()) != {episode}:
                raise ValueError(f"Parquet episode slice mismatch: {episode}")
            arrays[episode] = block
    details = []
    for episode in selected:
        table = arrays[episode]
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
        frames = np.asarray(table["frame_index"].to_numpy())
        if (actions.shape != (279, 12) or states.shape != (279, 16)
                or not np.isfinite(actions).all() or not np.isfinite(states).all()
                or np.max(np.abs(actions)) > 1.0001
                or not np.allclose(timestamps, np.arange(279) / 16, rtol=0, atol=1e-4)
                or not np.array_equal(frames, np.arange(279))):
            raise ValueError(f"Numeric source check failed: {episode}")
        for dim in (6, 11):
            if not np.all(np.isin(actions[:, dim], (-1.0, 1.0))):
                raise ValueError(f"Non-discrete gripper/control value: {episode}, {dim}")
        preview = by_preview[episode]
        if (preview["fps"] != 16 or preview["frames"] != 279
                or preview["layout"] != ["left", "right", "wrist"]
                or len(preview["camera_segments"]) != 3
                or {part["camera_key"] for part in preview["camera_segments"]} != set(CAMERAS)
                or not (ROOT / preview["video_path"]).is_file()):
            raise ValueError(f"Three-camera preview unavailable: {episode}")
        details.append({
            "episode_index": episode, "frames": 279,
            "action_dim": 12, "state_dim": 16,
            "max_abs_action": float(np.max(np.abs(actions))),
            "arm_abs_p99": float(np.quantile(np.abs(actions[:, :6]), 0.99)),
            "three_camera_preview": preview["video_path"],
            "numeric_and_temporal_integrity": True,
            "simulator_success_verified": False,
            "idm_action_semantics_verified": False,
        })
    result = {
        "selection_file": str(args.selection.resolve()),
        "source_revision": SNAPSHOT.name,
        "source_fps": 16,
        "selection_status": selection["selection_status"],
        "unique_episodes_checked": len(selected),
        "all_numeric_and_temporal_checks_passed": True,
        "simulator_success_verified": False,
        "idm_action_semantics_verified": False,
        "episodes": details,
    }
    suffix = "draft" if "DRAFT" in selection["selection_status"] else args.selection.stem
    out = ROOT / f"records/opendrawer_synthetic_source_audit_{suffix}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checked": len(selected), "status": result["selection_status"],
                      "source_fps": 16, "output": str(out),
                      "simulator_success_verified": False}))


if __name__ == "__main__":
    main()
