"""Fixed native-20Hz H50 human-left Diffusion workflow schema."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(__file__).resolve().parents[2]
DATASET_REL = Path("data/processed/OpenDrawer/h50_human_left_20fps_3cam")
DATASET = ROOT / DATASET_REL
REPO_ID = "local/open_drawer_h50_human_left_20fps_3cam"
BASE_OUTPUT = ROOT / "outputs/OpenDrawer/diffusion_human_left_3cam"
INSTRUCTION = "open the left drawer"
CAMERAS = [
    "observation.images.robot0_agentview_left_image",
    "observation.images.robot0_agentview_right_image",
    "observation.images.robot0_eye_in_hand_image",
]
FPS = 20
REQUIRED_MODEL_FILES = (
    "model.safetensors", "config.json", "train_config.json",
    "policy_preprocessor.json", "policy_postprocessor.json",
)

def validate_dataset(root=DATASET):
    dataset = LeRobotDataset(REPO_ID, root=root, video_backend="torchcodec")
    manifest = json.loads((Path(root)/"provenance_manifest.json").read_text())
    if (dataset.num_episodes != 28 or dataset.meta.fps != FPS
        or dataset.meta.camera_keys != CAMERAS
        or list(dataset.meta.tasks.index) != [INSTRUCTION]
        or tuple(dataset.features["action"]["shape"]) != (12,)
        or tuple(dataset.features["observation.state"]["shape"]) != (16,)
        or any(tuple(dataset.features[key]["shape"]) != (128,128,3) for key in CAMERAS)):
        raise ValueError("Expected exactly 28 H50 human-left native-20fps three-camera demonstrations")
    if (manifest["source_repo"] != "DAVIAN-Robotics/robocasa-H50"
        or manifest["source_type"] != "human" or manifest["source_instruction"] != INSTRUCTION
        or manifest["source_fps"] != FPS or manifest["target_fps"] != FPS
        or len(manifest["episodes"]) != 28
        or not manifest["verification"]["all_source_frames_preserved"]
        or not manifest["verification"]["state_action_exactly_equal_original_H50"]):
        raise ValueError("Native H50 provenance invalid")
    for episode in manifest["episodes"]:
        if episode["source_indices"] != list(range(episode["source_length"])):
            raise ValueError("Resampling prohibited in this workflow")
    return dataset

def policy_protocol(config):
    return {
        "policy": config.type, "observation_horizon":config.n_obs_steps,
        "prediction_horizon":config.horizon, "action_horizon":config.n_action_steps,
        "denoising_steps":config.num_inference_steps or config.num_train_timesteps,
        "noise_scheduler":config.noise_scheduler_type, "beta_schedule":config.beta_schedule,
        "prediction_type":config.prediction_type,
        "normalization":{key:value.value for key,value in config.normalization_mapping.items()},
        "vision_backbone":config.vision_backbone,
        "pretrained_visual_backbone":config.pretrained_backbone_weights,
        "pretrained_robot_policy":str(config.pretrained_path) if config.pretrained_path else None,
        "separate_encoder_per_camera":config.use_separate_rgb_encoder_per_camera,
        "camera_keys":list(config.image_features), "resolution":[128,128],
        "down_dims":list(config.down_dims), "drop_n_last_frames":config.drop_n_last_frames,
        "train_fps":FPS, "env_control_hz":FPS, "effective_dt_seconds":1/FPS,
        "policy_select_action_hz":FPS, "model_replan_hz":FPS/config.n_action_steps,
        "actions_generated_per_diffusion_sample":config.horizon,
        "actions_retained_and_executed":config.n_action_steps,
        "observation_queue_updated_every_env_step":True,
        "mixed_precision":"bf16" if config.use_amp else "no",
    }

def action_statistics(actions):
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 12 or not np.isfinite(actions).all():
        raise ValueError("Expected finite (N,12) actions")
    result = []
    for dim in range(12):
        values = actions[:,dim]
        result.append({
            "dim":dim, "raw_min":float(values.min()), "raw_max":float(values.max()),
            "raw_mean":float(values.mean()), "raw_std":float(values.std()),
            "p95_abs":float(np.percentile(abs(values),95)),
            "p99_abs":float(np.percentile(abs(values),99)),
            "clipped_count":int((abs(values)>1).sum()),
            "clipped_pct":float(100*(abs(values)>1).mean())})
    return result
