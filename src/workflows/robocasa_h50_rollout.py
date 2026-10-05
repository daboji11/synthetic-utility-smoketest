"""Reuse the validated H50 observation, streaming video and per-dimension rollout diagnostics."""
from __future__ import annotations
import json
import time
import numpy as np
import torch
from lerobot.policies.utils import prepare_observation_for_inference
from robocasa.utils.env_utils import convert_action

LEFT_TASK = "open the left drawer"
def raw_camera_key(key):
    return "video." + key.removeprefix("observation.images.").removesuffix("_image")

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
                video_path=None, steps_path=None, *, cameras, control_hz, horizon):
    CAMERAS, CONTROL_HZ, HORIZON = cameras, control_hz, horizon
    raw_obs, _ = env.reset(seed=seed)
    metadata = env.env.get_ep_meta()
    instruction = metadata.get("lang", "").strip().lower().rstrip(".")
    if instruction != LEFT_TASK:
        return None
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
            replanned = len(policy._queues["action"]) == 0
            if replanned:
                replan_steps.append(step)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=policy.config.use_amp):
                predicted = policy.select_action(prepared)
            action = postprocessor(predicted).detach().cpu().numpy().reshape(-1)
            if action.shape != (12,) or not np.isfinite(action).all():
                raise ValueError(f"Invalid Diffusion action at seed={seed} step={step}: {action}")
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
    result["arm_clipped_count"] = int(clipped_by_dim[:6].sum())
    result["gripper_clipped_count"] = int(clipped_by_dim[6])
    result["other_clipped_count"] = int(clipped_by_dim[7:].sum())
    print(f"CLIP_GROUPS arm={result['arm_clipped_count']} gripper={result['gripper_clipped_count']} "
          f"base_and_mode={result['other_clipped_count']}", flush=True)
    print("dim raw_min raw_max raw_mean raw_std p95_abs p99_abs clipped_count clipped_pct", flush=True)
    for d in per_dim:
        print(f"{d['dim']:>3} {d['raw_min']:>7.3f} {d['raw_max']:>7.3f} "
              f"{d['raw_mean']:>8.3f} {d['raw_std']:>7.3f} {d['p95_abs']:>7.3f} "
              f"{d['p99_abs']:>7.3f} {d['clipped_count']:>13} {d['clipped_pct']:>10.2f}", flush=True)
    if video_path is not None:
        print(f"VIDEO {video_path} columns={CAMERAS}", flush=True)
    return result
