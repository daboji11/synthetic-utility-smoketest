#!/usr/bin/env python
"""Manual H50 human-left ACT evaluation with saved processors and explicit statistics."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.utils import prepare_observation_for_inference
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action
from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASET = ROOT / "data/processed/OpenDrawer/h50_human_left_16fps"
CHECKPOINT = ROOT / "outputs/OpenDrawer/human_left_only/seed_0/checkpoints/last/pretrained_model"
EVAL_ROOT = ROOT / "outputs/OpenDrawer/human_left_only/seed_0/robocasa_eval"
CAMERA = "observation.images.robot0_agentview_left_image"
CAMERAS = [CAMERA]
REPO_ID = "local/open_drawer_h50_human_left_16fps"
RAW_CAMERA = "video.robot0_agentview_left"
LEFT_TASK = "open the left drawer"
HORIZON = get_task_horizon("OpenDrawer")
CONTROL_HZ = 16


def raw_camera_key(key):
    return "video." + key.removeprefix("observation.images.").removesuffix("_image")


def processor_state(checkpoint, filename, registry_name):
    config = json.loads((checkpoint / filename).read_text())
    matches = [step for step in config["steps"] if step["registry_name"] == registry_name]
    if len(matches) != 1 or not matches[0].get("state_file"):
        raise ValueError(f"Missing saved processor state: {registry_name}")
    state_file = (checkpoint / matches[0]["state_file"]).resolve()
    if not state_file.is_relative_to(checkpoint) or not state_file.is_file():
        raise FileNotFoundError(f"Invalid processor state file: {state_file}")
    with safe_open(state_file, framework="pt", device="cpu") as reader:
        state = {key: reader.get_tensor(key).numpy() for key in reader.keys()}
    return state, state_file


def make_policy(device="cuda", require_training_step=None):
    global CHECKPOINT
    # Pin last to its resolved immutable checkpoint before reading any files.
    CHECKPOINT = CHECKPOINT.resolve(strict=True)
    required = ("model.safetensors", "config.json", "train_config.json",
                "policy_preprocessor.json", "policy_postprocessor.json")
    if any(not (CHECKPOINT / name).is_file() for name in required):
        raise FileNotFoundError("Incomplete formal H50 ACT checkpoint")
    step_file = CHECKPOINT.parent / "training_state/training_step.json"
    trained_step = json.loads(step_file.read_text())["step"] if step_file.is_file() else None
    if require_training_step is not None and trained_step != require_training_step:
        raise ValueError(f"Checkpoint is step {trained_step}; expected {require_training_step}. "
                         "Wait until training has saved the final checkpoint.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    dataset = LeRobotDataset(REPO_ID, root=DATASET, video_backend="torchcodec")
    if (dataset.num_episodes != 28 or dataset.meta.fps != CONTROL_HZ
        or dataset.meta.camera_keys != CAMERAS
        or list(dataset.meta.tasks.index) != [LEFT_TASK]):
        raise ValueError("Evaluation dataset is not H50 human-left 16fps")
    manifest = json.loads((DATASET / "resampling_manifest.json").read_text())
    if (manifest["source_repo"] != "DAVIAN-Robotics/robocasa-H50"
        or manifest["source_type"] != "human"
        or manifest["source_instruction"] != LEFT_TASK):
        raise ValueError("Dataset provenance mismatch")
    train_config = json.loads((CHECKPOINT / "train_config.json").read_text())
    train_root = Path(train_config["dataset"]["root"])
    if not train_root.is_absolute():
        train_root = ROOT / train_root
    if train_root.resolve() != DATASET.resolve() or train_config["dataset"]["repo_id"] != REPO_ID:
        raise ValueError("Checkpoint was trained on a different dataset")
    if train_config["dataset"].get("image_transforms", {}).get("enable", False):
        raise ValueError("Training image transforms need an explicit evaluation compatibility check")
    saved_policy_type = json.loads((CHECKPOINT / "config.json").read_text())["type"]
    if saved_policy_type == "act_camera_id":
        from src.policies.act_camera_id.configuration_act_camera_id import ACTCameraIdConfig
        from src.policies.act_camera_id.modeling_act_camera_id import ACTCameraIdPolicy
        config_class, policy_class = ACTCameraIdConfig, ACTCameraIdPolicy
    elif saved_policy_type == "act":
        config_class, policy_class = ACTConfig, ACTPolicy
    else:
        raise ValueError(f"Unsupported checkpoint policy type: {saved_policy_type}")
    config = config_class.from_pretrained(str(CHECKPOINT), local_files_only=True)
    if (list(config.input_features) != ["observation.state", *CAMERAS]
        or tuple(config.input_features["observation.state"].shape) != (16,)
        or any(tuple(config.input_features[key].shape) != (3, 128, 128) for key in CAMERAS)
        or tuple(config.output_features["action"].shape) != (12,)):
        raise ValueError("Checkpoint input/output schema mismatch")
    pre_state, pre_file = processor_state(
        CHECKPOINT, "policy_preprocessor.json", "normalizer_processor")
    post_state, post_file = processor_state(
        CHECKPOINT, "policy_postprocessor.json", "unnormalizer_processor")
    for feature in ("action", "observation.state"):
        for stat in ("mean", "std"):
            saved = pre_state[f"{feature}.{stat}"]
            expected = np.asarray(dataset.meta.stats[feature][stat])
            if not np.allclose(saved, expected, rtol=1e-4, atol=1e-5):
                raise ValueError(f"Saved {feature}.{stat} does not match the training dataset")
    for stat in ("mean", "std"):
        if not np.array_equal(pre_state[f"action.{stat}"], post_state[f"action.{stat}"]):
            raise ValueError(f"Action normalizer/unnormalizer {stat} mismatch")
    for camera in CAMERAS:
        for stat, imagenet in (("mean", [0.485, 0.456, 0.406]),
                               ("std", [0.229, 0.224, 0.225])):
            expected = (np.asarray(imagenet).reshape(3, 1, 1)
                        if train_config["dataset"].get("use_imagenet_stats", True)
                        else np.asarray(dataset.meta.stats[camera][stat]))
            if not np.allclose(pre_state[f"{camera}.{stat}"], expected, atol=1e-6):
                raise ValueError(f"Saved image normalization mismatch: {camera}.{stat}")
    config.device = device  # In-memory placement only; never resave checkpoint config.
    policy = policy_class.from_pretrained(str(CHECKPOINT), config=config,
                                      local_files_only=True, strict=True).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(CHECKPOINT),
        preprocessor_overrides={"device_processor": {"device": device}})
    policy._eval_validation = {
        "resolved_checkpoint": str(CHECKPOINT), "training_step": trained_step,
        "dataset": str(DATASET), "episodes": 28, "fps": CONTROL_HZ,
        "camera_mapping": {key: raw_camera_key(key) for key in CAMERAS},
        "normalizer": str(pre_file), "unnormalizer": str(post_file),
        "saved_normalization_matches_training": True, "action_pre_post_stats_match": True,
        "images": "RGB uint8 HWC -> float32 CHW /255 -> saved normalization",
        "vertical_flip": "RoboCasaGymEnv flips once; evaluator does not flip again",
        "state_order": "ee_pos3,ee_quat4,gripper_qpos2,base_pos3,base_quat4",
        "action_order": "ee_pos3,ee_rot3,gripper1,base_motion4,control_mode1",
        "action_scaling": "No extra interpolation or scaling",
        "device": device,
    }
    return dataset.features, policy, preprocessor, postprocessor


def obs_to_policy(raw_obs, features, device="cuda"):
    state = np.concatenate((
        raw_obs["state.end_effector_position_relative"],
        raw_obs["state.end_effector_rotation_relative"],
        raw_obs["state.gripper_qpos"],
        raw_obs["state.base_position"],
        raw_obs["state.base_rotation"],
    )).astype(np.float32)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError(f"Observation state shape/type mismatch: {state.shape}")
    frame = {"observation.state": state}
    for key in features:
        if not key.startswith("observation.images."):
            continue
        image = raw_obs[raw_camera_key(key)]
        if image.shape != (128, 128, 3) or image.dtype != np.uint8:
            raise ValueError(f"Camera shape/type mismatch: {key}, {image.shape}, {image.dtype}")
        frame[key] = image.copy()
    return prepare_observation_for_inference(
        frame, device=torch.device(device), task=LEFT_TASK, robot_type="robomimic")


def preflight_forward(features, policy, preprocessor, postprocessor):
    # One saved training observation, no simulator, no optimizer and no gradients.
    dataset = LeRobotDataset(REPO_ID, root=DATASET, video_backend="torchcodec")
    sample = dataset[0]
    state = sample["observation.state"].numpy()
    raw = {
        "state.end_effector_position_relative": state[:3],
        "state.end_effector_rotation_relative": state[3:7],
        "state.gripper_qpos": state[7:9],
        "state.base_position": state[9:12],
        "state.base_rotation": state[12:16],
    }
    for key in CAMERAS:
        raw[raw_camera_key(key)] = np.rint(sample[key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    frame = obs_to_policy(raw, features, device=policy.config.device)
    if not np.array_equal(frame["observation.state"].cpu().numpy().reshape(-1), state):
        raise ValueError("State ordering differs from the saved training sample")
    for key in CAMERAS:
        if not torch.allclose(frame[key].cpu().squeeze(0), sample[key], atol=1e-6):
            raise ValueError(f"Evaluation image conversion differs from training: {key}")
    prepared = preprocessor(frame)
    policy.reset()
    with torch.inference_mode():
        action = policy.select_action(prepared)
    applied = postprocessor(action)
    if tuple(applied.shape) != (1, 12) or not torch.isfinite(applied).all():
        raise ValueError("Saved checkpoint produces an invalid action")
    queued = len(policy._action_queue)
    if queued != policy.config.n_action_steps - 1:
        raise ValueError("ACT action queue length mismatch")
    policy.reset()
    return {"observation_matches_saved_training_sample": True,
            "prepared_shapes": {key: list(prepared[key].shape)
                                for key in ["observation.state", *CAMERAS]},
            "postprocessed_action_shape": list(applied.shape), "finite_action": True,
            "predicted_chunk_size": policy.config.chunk_size, "queued_after_first_action": queued,
            "simulation_started": False}


def video_image(raw_obs, cameras, seed, step, reward, success):
    import cv2
    images = [raw_obs[raw_camera_key(key)] for key in cameras]
    if any(image.shape != (128, 128, 3) or image.dtype != np.uint8 for image in images):
        raise ValueError("Video camera shape/type mismatch")
    canvas = np.concatenate(images, axis=1)
    canvas = np.concatenate((canvas, np.zeros((32, canvas.shape[1], 3), dtype=np.uint8)), axis=0)
    cv2.putText(canvas, f"seed={seed} step={step} success={int(success)} reward={reward:.1f}",
                (3, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


def run_episode(env, seed, features, policy, preprocessor, postprocessor,
                video_path=None, steps_path=None):
    raw_obs, _ = env.reset(seed=seed)
    metadata = env.env.get_ep_meta()
    instruction = metadata.get("lang", "").strip().lower().rstrip(".")
    if instruction != LEFT_TASK:
        return None
    policy.reset()
    raw_actions = []
    clipped_by_dim = np.zeros(12, dtype=np.int64)
    replan_steps = []
    max_drawer = 0.0
    started = time.monotonic()
    print(f"accepted seed={seed} instruction={instruction}", flush=True)
    video = None
    step_log = None
    try:
        if video_path is not None:
            import cv2
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                    CONTROL_HZ, (128 * len(CAMERAS), 160))
            if not video.isOpened():
                raise RuntimeError(f"VideoWriter failed: {video_path}")
            video.write(video_image(raw_obs, CAMERAS, seed, 0, 0.0, False))
        if steps_path is not None:
            steps_path.parent.mkdir(parents=True, exist_ok=True)
            step_log = steps_path.open("w", buffering=1)
        for step in range(1, HORIZON + 1):
            frame = obs_to_policy(raw_obs, features)
            prepared = preprocessor(frame)
            replanned = len(policy._action_queue) == 0
            if replanned:
                replan_steps.append(step)
            with torch.inference_mode():
                predicted = policy.select_action(prepared)
            action = postprocessor(predicted).detach().cpu().numpy().reshape(-1)
            if action.shape != (12,) or not np.isfinite(action).all():
                raise ValueError(f"Invalid ACT action at seed={seed} step={step}: {action}")
            raw_actions.append(action.copy())
            clipped_by_dim += (np.abs(action) > 1.0).astype(np.int64)
            applied = np.clip(action, -1.0, 1.0).astype(np.float32)
            action_dict = convert_action(applied)
            if set(action_dict) != set(env.action_space.spaces):
                raise ValueError("RoboCasa action keys mismatch")
            raw_obs, reward, terminated, truncated, info = env.step(action_dict)
            success = bool(info.get("success", False))
            door_state = env.env.drawer.get_door_state(env=env.env)
            max_drawer = max(max_drawer, *(float(value) for value in door_state.values()))
            if step_log is not None:
                step_log.write(json.dumps({
                    "step": step, "reward": float(reward), "success": success,
                    "replanned": replanned, "raw_action": action.tolist(),
                    "applied_action": applied.tolist(),
                    "door_state": {key: float(value) for key, value in door_state.items()},
                }) + "\n")
            if video is not None:
                video.write(video_image(raw_obs, CAMERAS, seed, step, float(reward), success))
            if step == 1:
                print(f"INTERFACE_OK state=(16,) cameras={len(CAMERAS)}x(128,128,3) "
                      f"action=(12,) control_hz={env.env.control_freq} "
                      f"replan_every={policy.config.n_action_steps}", flush=True)
            if step % 100 == 0:
                print(f"seed={seed} step={step}/{HORIZON} replans={len(replan_steps)} "
                      f"clips={int(clipped_by_dim.sum())} success={success} reward={float(reward):.4f} "
                      f"max_drawer_open={max_drawer:.4f} elapsed_s={time.monotonic()-started:.1f}", flush=True)
            if success or terminated or truncated:
                break
    finally:
        if video is not None:
            video.release()
        if step_log is not None:
            step_log.close()
    actions = np.stack(raw_actions)
    per_dim = []
    for dim in range(12):
        values = actions[:, dim]
        per_dim.append({
            "dim": dim, "raw_min": float(values.min()), "raw_max": float(values.max()),
            "raw_mean": float(values.mean()), "raw_std": float(values.std()),
            "p95_abs": float(np.percentile(np.abs(values), 95)),
            "p99_abs": float(np.percentile(np.abs(values), 99)),
            "clipped_count": int(clipped_by_dim[dim]),
            "clipped_pct": float(100 * clipped_by_dim[dim] / step),
        })
    result = {
        "seed": seed, "instruction": instruction, "steps": step, "success": success,
        "reward": float(reward), "clipped_action_values": int(clipped_by_dim.sum()),
        "max_abs_raw_action": float(np.abs(actions).max()),
        "model_replans": len(replan_steps), "replan_steps": replan_steps,
        "action_stats_by_dim": per_dim, "max_drawer_open_fraction": max_drawer,
        "wall_clock_s": round(time.monotonic() - started, 2),
        "video": str(video_path) if video_path is not None else None,
        "steps_log": str(steps_path) if steps_path is not None else None,
    }
    print(f"EPISODE_STATS seed={seed} success={success} reward={result['reward']:.4f} "
          f"steps={step} replans={len(replan_steps)} clips={result['clipped_action_values']} "
          f"max_abs_raw_action={result['max_abs_raw_action']:.4f} "
          f"max_drawer_open={max_drawer:.4f} wall_clock_s={result['wall_clock_s']:.2f}", flush=True)
    print("dim raw_min raw_max raw_mean raw_std p95_abs p99_abs clipped_count clipped_pct", flush=True)
    for d in per_dim:
        print(f"{d['dim']:>3} {d['raw_min']:>7.3f} {d['raw_max']:>7.3f} "
              f"{d['raw_mean']:>8.3f} {d['raw_std']:>7.3f} {d['p95_abs']:>7.3f} "
              f"{d['p99_abs']:>7.3f} {d['clipped_count']:>13} {d['clipped_pct']:>10.2f}", flush=True)
    if video_path is not None:
        print(f"VIDEO {video_path} columns={CAMERAS}", flush=True)
    return result


def main():
    global DATASET, CHECKPOINT, EVAL_ROOT, CAMERAS, REPO_ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--replan-every", type=int, default=100)
    parser.add_argument("--multicamera", action="store_true")
    parser.add_argument("--camera-id", action="store_true",
                        help="Use the separate ACT camera-identity checkpoint")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--require-training-step", type=int)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--preflight", action="store_true",
                        help="CPU checkpoint/input forward check; no simulator or rollout")
    parser.add_argument("--preflight-report", type=Path)
    args = parser.parse_args()
    if (args.episodes < 1 or args.start_seed < 0
        or not args.run_name.replace("_", "").isascii()
        or not args.run_name.replace("_", "").isalnum()
        or (args.seeds is not None and (len(args.seeds) != args.episodes
            or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0))):
        parser.error("Invalid episodes, seed or run name")
    if args.preflight_report is not None and not args.preflight:
        parser.error("--preflight-report requires --preflight")
    if args.camera_id:
        args.multicamera = True
    if args.multicamera:
        DATASET = ROOT / "data/processed/OpenDrawer/h50_human_left_16fps_3cam"
        CHECKPOINT = ROOT / "outputs/OpenDrawer/human_left_only_3cam/seed_0/checkpoints/last/pretrained_model"
        EVAL_ROOT = ROOT / "outputs/OpenDrawer/human_left_only_3cam/seed_0/robocasa_eval"
        CAMERAS = [CAMERA, "observation.images.robot0_agentview_right_image",
                   "observation.images.robot0_eye_in_hand_image"]
        REPO_ID += "_3cam"
        if args.require_training_step is None:
            args.require_training_step = 10000
    if args.camera_id:
        CHECKPOINT = ROOT / "outputs/OpenDrawer/human_left_only_3cam_camera_id/seed_0/checkpoints/last/pretrained_model"
        EVAL_ROOT = ROOT / "outputs/OpenDrawer/human_left_only_3cam_camera_id/seed_0/robocasa_eval"
    if args.checkpoint is not None:
        CHECKPOINT = args.checkpoint
    output = EVAL_ROOT / args.run_name
    if output.exists() and not args.preflight:
        raise FileExistsError(f"Preserving existing evaluation: {output}")
    features, policy, preprocessor, postprocessor = make_policy(
        device="cpu" if args.preflight else "cuda",
        require_training_step=args.require_training_step)
    checkpoint_n_action_steps = policy.config.n_action_steps
    if policy.config.temporal_ensemble_coeff is not None:
        raise ValueError("Temporal ensembling is not supported by this evaluator")
    if not 1 <= args.replan_every <= policy.config.chunk_size:
        parser.error(f"--replan-every must be in 1..{policy.config.chunk_size}")
    policy.config.n_action_steps = args.replan_every
    policy.reset()
    versions = {}
    for name in ("torch", "lerobot", "robocasa", "robosuite"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "local source"
    protocol = {
        "checkpoint": str(CHECKPOINT), "dataset": str(DATASET),
        "task": "OpenDrawer", "instruction": LEFT_TASK, "requested_episodes": args.episodes,
        "start_seed": args.start_seed, "requested_seeds": args.seeds, "horizon": HORIZON,
        "control_freq": CONTROL_HZ, "effective_dt_seconds": 1 / CONTROL_HZ,
        "camera": CAMERA, "camera_keys": CAMERAS, "resolution": [128, 128],
        "split": "all", "obj_registries": ["lightwheel"],
        "action_order": "ee_pos3,ee_rot3,gripper1,base_motion4,control_mode1",
        "state_order": "ee_pos3,ee_quat4,gripper_qpos2,base_pos3,base_quat4",
        "action_clip": [-1.0, 1.0], "success_source": "RoboCasaGymEnv info.success",
        "policy_type": policy.config.type,
        "camera_identity_encoding": hasattr(policy.model, "camera_identity_embed"),
        "act_chunk_size": policy.config.chunk_size, "checkpoint_n_action_steps": checkpoint_n_action_steps,
        "eval_n_action_steps": policy.config.n_action_steps,
        "temporal_ensemble_coeff": policy.config.temporal_ensemble_coeff,
        "save_video": args.save_video, "video_columns": CAMERAS, "video_fps": CONTROL_HZ,
        "versions": versions, "interface_validation": policy._eval_validation,
    }
    print("EVAL_PROTOCOL " + json.dumps(protocol), flush=True)
    if args.preflight:
        report = {**protocol, "forward_check": preflight_forward(features, policy, preprocessor, postprocessor)}
        if args.preflight_report is not None:
            args.preflight_report.parent.mkdir(parents=True, exist_ok=True)
            args.preflight_report.write_text(json.dumps(report, indent=2) + "\n")
        print("PREFLIGHT_OK " + json.dumps(report["forward_check"]), flush=True)
        return
    output.mkdir(parents=True)
    results, skipped = [], []
    started = time.monotonic()
    env = None
    try:
        env = RoboCasaGymEnv(env_name="OpenDrawer", camera_widths=128, camera_heights=128,
                             split="all", obj_registries=("lightwheel",), control_freq=CONTROL_HZ)
        if env.env.control_freq != CONTROL_HZ:
            raise ValueError("Actual RoboCasa control frequency mismatch")
        for key in CAMERAS:
            if tuple(env.observation_space[raw_camera_key(key)].shape) != (128, 128, 3):
                raise ValueError(f"Actual RoboCasa camera space mismatch: {key}")
        protocol["actual_env_control_freq"] = env.env.control_freq
        protocol["actual_env_camera_names"] = list(env.env.camera_names)
        protocol["controller_types"] = [
            {key: type(value).__name__ for key, value in robot.composite_controller.part_controllers.items()}
            for robot in env.env.robots]
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        print("ENV_INTERFACE_OK control_hz=16 cameras=" + json.dumps(CAMERAS), flush=True)
        with (output / "episodes.jsonl").open("w", buffering=1) as log:
            seed = args.start_seed
            while len(results) < args.episodes:
                if seed >= args.start_seed + max(100, 10 * args.episodes):
                    raise RuntimeError("Could not find enough left-drawer seeds")
                current_seed = args.seeds[len(results)] if args.seeds is not None else seed
                result = run_episode(
                    env, current_seed, features, policy, preprocessor, postprocessor,
                    video_path=output / "videos" / f"seed_{current_seed}.mp4" if args.save_video else None,
                    steps_path=output / "steps" / f"seed_{current_seed}.jsonl")
                if result is None:
                    if args.seeds is not None:
                        raise ValueError(f"Requested seed={current_seed} is not left-drawer")
                    skipped.append(seed)
                    print(f"skipped right-drawer seed={seed}", flush=True)
                else:
                    results.append(result)
                    log.write(json.dumps(result) + "\n")
                    successes_so_far = sum(r["success"] for r in results)
                    print(f"PROGRESS {len(results)}/{args.episodes} successes={successes_so_far} "
                          f"success_rate={successes_so_far / len(results):.1%}", flush=True)
                seed += 1
    finally:
        if env is not None:
            env.close()
    successes = sum(r["success"] for r in results)
    summary = {**protocol, "completed_episodes": len(results), "successes": successes,
               "success_rate": successes / len(results), "skipped_non_left_seeds": skipped,
               "selected_seeds": [r["seed"] for r in results],
               "wall_clock_s": round(time.monotonic() - started, 2)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
