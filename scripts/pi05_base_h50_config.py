"""Clean pi05_base + 28 H50 OpenDrawer interface. No human300 checkpoint use."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/processed/OpenDrawer/h50_human_left_native_dp_20fps_256"
BASE = ROOT / "data/raw/models/pi05_base_h50_interface"
SOURCE_COMMIT = "5a6beda9ff99da30b4e1b59320f6a32971d7c397"
UPSTREAM_CONFIG = "pi05-og_robocasa_pickplacetoastertocounter"
SEEDS = (3, 4, 5, 8, 10, 13, 16, 18, 20, 21, 22, 23, 26, 27, 28, 31, 32, 33, 34, 35)
STATE_KEYS = ("state.end_effector_position_relative", "state.end_effector_rotation_relative",
              "state.base_position", "state.base_rotation", "state.gripper_qpos")
CAMERAS = {"observation/image": "video.robot0_agentview_left",
           "observation/wrist_image": "video.robot0_eye_in_hand",
           "observation/right_image": "video.robot0_agentview_right"}
STATE_ORDER = tuple(range(7)) + tuple(range(9, 16)) + (7, 8)


def verify_source() -> Path:
    import openpi
    source = Path(openpi.__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCE_COMMIT:
        raise ValueError(f"Wrong OpenPI revision: {revision}")
    return source


def verify_4090() -> dict:
    line = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"
    ], text=True).strip().splitlines()
    if len(line) != 1:
        raise RuntimeError(f"Expected one RTX 4090, found: {line}")
    name, memory = [part.strip() for part in line[0].split(",", 1)]
    if "4090" not in name or int(memory) < 24000:
        raise RuntimeError(f"Use a 24 GB RTX 4090 for this experiment: {line[0]}")
    return {"name": name, "memory_mib": int(memory)}


def official_base_params_uri() -> str:
    from openpi.training import config, weight_loaders
    upstream = config.get_config(UPSTREAM_CONFIG)
    loader = upstream.weight_loader
    if not isinstance(loader, weight_loaders.CheckpointWeightLoader):
        raise ValueError("Expected upstream base checkpoint weight loader")
    uri = loader.params_path
    if not uri.startswith("gs://openpi-assets/checkpoints/") or not uri.endswith("/pi05_base/params"):
        raise ValueError(f"Upstream no longer points to pi05_base: {uri}")
    return uri


def verify_dataset() -> dict:
    info = json.loads((DATASET / "meta/info.json").read_text())
    provenance = json.loads((DATASET / "provenance_manifest.json").read_text())
    episodes = [json.loads(line) for line in (DATASET / "meta/episodes.jsonl").read_text().splitlines()]
    tasks = [json.loads(line) for line in (DATASET / "meta/tasks.jsonl").read_text().splitlines()]
    modality = json.loads((DATASET / "meta/modality.json").read_text())
    if (info["fps"], info["total_episodes"], info["total_frames"], info["total_videos"]) != (20, 28, 5916, 84):
        raise ValueError("Expected exactly 28 original 20 Hz H50 demos and 84 camera videos")
    if len(episodes) != 28 or sum(ep["length"] for ep in episodes) != 5916:
        raise ValueError("Episode count/length mismatch")
    if [row["task"] for row in tasks] != ["open the left drawer"]:
        raise ValueError("Prompt mismatch")
    if provenance.get("source_type") != "human" or provenance.get("source_repo") != "DAVIAN-Robotics/robocasa-H50":
        raise ValueError("Not the original H50 human dataset")
    if provenance.get("source_fps") != 20 or provenance.get("target_fps") != 20 or len(provenance["episodes"]) != 28:
        raise ValueError("Unexpected dataset provenance")
    expected_state = ("end_effector_position_relative", "end_effector_rotation_relative",
                      "gripper_qpos", "base_position", "base_rotation")
    expected_action = ("end_effector_position", "end_effector_rotation", "gripper_close",
                       "base_motion", "control_mode")
    if tuple(modality["state"]) != expected_state or tuple(modality["action"]) != expected_action:
        raise ValueError("State/action modality order mismatch")
    state_slices = ((0, 3), (3, 7), (7, 9), (9, 12), (12, 16))
    action_slices = ((0, 3), (3, 6), (6, 7), (7, 11), (11, 12))
    for kind, names, ranges in (("state", expected_state, state_slices),
                                ("action", expected_action, action_slices)):
        for name, (start, end) in zip(names, ranges, strict=True):
            row = modality[kind][name]
            if (row["start"], row["end"]) != (start, end):
                raise ValueError(f"Wrong {kind} slice for {name}")
    if tuple(modality["video"]) != ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"):
        raise ValueError("Camera keys/order mismatch")
    if info["features"]["action"]["shape"] != [12] or info["features"]["observation.state"]["shape"] != [16]:
        raise ValueError("Action/state feature dimensions mismatch")
    if len(list(DATASET.glob("data/**/*.parquet"))) != 28 or len(list(DATASET.glob("videos/**/*.mp4"))) != 84:
        raise ValueError("Missing demo data or camera videos")
    return info


def expected_norm_stats() -> dict:
    """H50 raw action order is already model order; state alone needs reordering."""
    verify_dataset()
    stats = json.loads((DATASET / "meta/stats.json").read_text())

    def values(key: str, order: tuple[int, ...]) -> dict:
        row = stats[key]
        return {
            "mean": [float(row["mean"][i]) for i in order] + [0.0] * (32 - len(order)),
            "std": [float(row["std"][i]) for i in order] + [1.0] * (32 - len(order)),
        }

    return {"norm_stats": {
        "state": values("observation.state", STATE_ORDER),
        "actions": values("action", tuple(range(12))),
    }}


def verify_base_checkpoint() -> dict:
    manifest = json.loads((BASE / "source_manifest.json").read_text())
    if manifest.get("model") != "pi05_base" or manifest.get("params_uri") != official_base_params_uri():
        raise ValueError("Base manifest does not identify official pi05_base")
    if "human300" in str((BASE / "params").resolve()):
        raise ValueError("Base params resolve into a human300 checkpoint")
    if not (BASE / "params/_METADATA").is_file():
        raise FileNotFoundError("Incomplete base params")
    if hashlib.sha256((BASE / "params/_METADATA").read_bytes()).hexdigest() != manifest["checkpoint_metadata_sha256"]:
        raise ValueError("Base params metadata hash mismatch")
    saved = json.loads((BASE / "assets/norm_stats.json").read_text())
    if saved != expected_norm_stats():
        raise ValueError("Interface stats differ from exact 28-H50-derived stats")
    if hashlib.sha256((BASE / "assets/norm_stats.json").read_bytes()).hexdigest() != manifest["norm_sha256"]:
        raise ValueError("Norm stats hash mismatch")
    return manifest


def observation(raw: dict, prompt: str = "open the left drawer") -> dict:
    result = {key: np.ascontiguousarray(raw[value]) for key, value in CAMERAS.items()}
    for key, image in result.items():
        if image.shape != (256, 256, 3) or image.dtype != np.uint8:
            raise ValueError(f"Bad camera {key}: {image.shape}, {image.dtype}")
    result["observation/state"] = np.concatenate([raw[key] for key in STATE_KEYS]).astype(np.float32)
    if result["observation/state"].shape != (16,) or not np.isfinite(result["observation/state"]).all():
        raise ValueError("Bad 16D state")
    result["prompt"] = prompt
    return result


def make_config(*, lora: bool, batch_size: int = 4, steps: int = 4000,
                lr: float = 5e-6, seed: int = 0, exp_name: str = "seed_0", num_workers: int = 4):
    from flax import nnx
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared import normalize
    from openpi.shared.nnx_utils import PathRegex
    from openpi.training import config, optimizer, weight_loaders

    verify_base_checkpoint()
    norm = normalize.load(BASE / "assets")
    upstream = config.get_config(UPSTREAM_CONFIG)
    model = Pi0Config(pi05=True, max_token_len=200,
                      paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
                      action_expert_variant="gemma_300m_lora" if lora else "gemma_300m")
    data = config.LeRobotRobocasaDataConfig(
        data_dirs=[{"path": str(DATASET.resolve()), "filter_key": None}] if lora else [],
        base_config=config.DataConfig(norm_stats=norm, use_quantile_norm=False))
    return dataclasses.replace(
        upstream, name="pi05_base_h50_left_lora" if lora else "pi05_base_h50_left_eval",
        exp_name=exp_name, model=model, data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(BASE / "params")),
        freeze_filter=nnx.All(nnx.Param, nnx.Not(PathRegex(".*lora.*"))),
        ema_decay=None, batch_size=batch_size, num_train_steps=steps,
        num_workers=num_workers, seed=seed,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=min(200, max(1, steps // 10)),
            peak_lr=lr, decay_steps=steps, decay_lr=lr / 10),
        checkpoint_base_dir=str(ROOT / "outputs/OpenDrawer"),
        log_interval=10, save_interval=4000, keep_period=1000, wandb_enabled=False,
        policy_metadata={"task": "OpenDrawer", "instruction": "open the left drawer",
                         "training_data": "28 original H50 human demos" if lora else "none",
                         "base": official_base_params_uri(), "fps": 20, "lora_only": lora})
