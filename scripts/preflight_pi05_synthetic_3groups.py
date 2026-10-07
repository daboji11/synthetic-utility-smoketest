#!/usr/bin/env python3
"""Read-only remote preflight for three fixed synthetic Pi0.5 LoRA datasets."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pi05_h50_config import CHECKPOINT, make_config, verify_source

SELECTION = Path(__file__).resolve().parents[1] / (
    "records/opendrawer_synthetic_8group_selection_exploratory_fee2a10f.json")
GROUPS = ("redundant", "diverse", "random")
CAMERAS = ("observation/image", "observation/wrist_image", "observation/right_image")

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--check-weights", action="store_true")
    args = p.parse_args()
    dataset_root = args.dataset_root.resolve()
    selection = json.loads(SELECTION.read_text())
    digest = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
    if selection["selection_status"] != "EXPLORATORY_HUMAN_OVERRIDES_AND_PROVISIONAL_VISUAL":
        raise ValueError("Wrong selection status")
    if selection["group_order"] != list(GROUPS):
        raise ValueError("Group order changed")
    if args.check_weights:
        checkpoint = json.loads((CHECKPOINT / "source_manifest.json").read_text())
        for item in checkpoint["files"]:
            rel = Path(item["path"]).relative_to(checkpoint["prefix"])
            file = CHECKPOINT / rel
            if not file.exists() or file.stat().st_size != item["size"]:
                raise ValueError(f"Missing or truncated pretrained weight: {file}")
    import jax
    from openpi.training import data_loader
    source = verify_source()
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("CUDA JAX device required")
    checks = []
    for group in GROUPS:
        dataset = dataset_root / group
        info = json.loads((dataset / "meta/info.json").read_text())
        provenance = json.loads((dataset / "provenance_manifest.json").read_text())
        if (info["fps"], info["total_episodes"], info["total_frames"], info["total_videos"]) != (20, 8, 2792, 24):
            raise ValueError(f"Wrong data counts in {group}")
        if (provenance["selection_sha256"] != digest
            or provenance["group"] != group
            or provenance["source_episode_indices"] != selection["groups"][group]["episode_indices"]):
            raise ValueError(f"Selection differs from dataset: {group}")
        parquet = sorted(dataset.glob("data/**/*.parquet"))
        video = sorted(dataset.glob("videos/**/*.mp4"))
        if len(parquet) != 8 or len(video) != 24:
            raise ValueError(f"Incomplete dataset transfer: {group}")
        for episode_index, file in enumerate(parquet):
            table = pq.read_table(file, columns=[
                "action", "observation.state", "timestamp", "frame_index",
                "episode_index", "index", "task_index"])
            if (table.num_rows != 349
                or set(table["episode_index"].to_pylist()) != {episode_index}
                or not np.allclose(table["timestamp"].to_numpy(), np.arange(349) / 20, atol=1e-6)):
                raise ValueError(f"Episode time/index mismatch: {file}")
            action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            if action.shape != (349, 12) or not np.isfinite(action).all():
                raise ValueError(f"Invalid actions: {file}")
        cfg = make_config(dataset=dataset, batch_size=1, steps=4000, num_workers=0)
        data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
        raw = data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
        sample = raw[0]
        if any(sample[camera].shape != (256, 256, 3) or sample[camera].dtype != np.uint8
               for camera in CAMERAS):
            raise ValueError(f"Three-camera reader mismatch: {group}")
        if sample["actions"].shape != (50, 12) or sample["observation/state"].shape != (16,):
            raise ValueError(f"Policy state/action shape mismatch: {group}")
        if sample["prompt"] != "open the left drawer":
            raise ValueError(f"Task prompt mismatch: {group}")
        source_action = np.asarray(pq.read_table(parquet[0], columns=["action"])["action"].to_pylist()[:50], dtype=np.float32)
        if not np.array_equal(sample["actions"], source_action):
            raise ValueError(f"Action order/magnitude altered by reader: {group}")
        transformed = data_loader.transform_dataset(raw, data_config)[0]
        if (set(transformed["image"]) != {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
            or not all(bool(value) for value in transformed["image_mask"].values())):
            raise ValueError(f"Model camera masking mismatch: {group}")
        checks.append({"group": group, "episodes": 8, "frames": 2792,
                       "cameras": 3, "source_episode_indices": provenance["source_episode_indices"],
                       "first_batch_action_window": list(sample["actions"].shape)})
        print("GROUP_PREFLIGHT_OK " + json.dumps(checks[-1]), flush=True)
    print("ALL_GROUPS_PREFLIGHT_OK " + json.dumps({
        "groups": checks, "source": str(source), "gpu": [str(device) for device in jax.devices()],
        "optimizer_updates": 0, "rollouts": 0}), flush=True)

if __name__ == "__main__":
    main()
