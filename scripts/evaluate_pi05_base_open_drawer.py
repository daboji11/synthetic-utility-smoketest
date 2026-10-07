#!/usr/bin/env python3
"""Matched 20-seed OpenDrawer rollout for untouched pi05_base or its 28-H50 LoRA."""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYNPUT_BACKEND", "dummy")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pi05_base_h50_config import (  # noqa: E402
    BASE, CAMERAS, ROOT, SEEDS, SOURCE_COMMIT, STATE_KEYS,
    make_config, observation, official_base_params_uri, verify_base_checkpoint,
    verify_source, verify_4090,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=750)
    parser.add_argument("--replan-steps", type=int, default=5)
    args = parser.parse_args()
    if args.horizon != 750 or args.replan_steps != 5:
        parser.error("This clean comparison fixes horizon=750 and replan-steps=5")
    verify_source()
    gpu = verify_4090()
    base = verify_base_checkpoint()
    checkpoint = args.checkpoint.resolve()
    if args.lora:
        training_root = (ROOT / "outputs/OpenDrawer/pi05_base_h50_left_lora/seed_0").resolve()
        if not checkpoint.is_relative_to(training_root):
            parser.error("LoRA checkpoint must come from this clean pi05_base + H50 run")
        actual = json.loads((training_root / "actual_config.json").read_text())
        if actual["base_params_uri"] != official_base_params_uri() or actual["norm_sha256"] != base["norm_sha256"]:
            parser.error("LoRA checkpoint provenance differs from base baseline")
    elif checkpoint != BASE.resolve():
        parser.error("Untouched baseline must use the verified pi05_base directory")
    if args.output.exists():
        parser.error(f"Preserving existing output: {args.output}")

    import cv2
    import jax
    import numpy as np
    from openpi.policies import policy_config
    from robocasa.utils.env_utils import convert_action
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("CUDA JAX device required")
    cfg = make_config(lora=args.lora)
    policy = policy_config.create_trained_policy(cfg, checkpoint)
    args.output.mkdir(parents=True)
    protocol = {
        "model": "pi05_base + 28 H50 LoRA" if args.lora else "untouched pi05_base",
        "checkpoint": str(checkpoint), "official_base_params_uri": official_base_params_uri(),
        "norm_sha256": base["norm_sha256"], "source_commit": SOURCE_COMMIT,
        "seeds": list(SEEDS), "episodes": 20, "horizon": 750, "replan_steps": 5,
        "control_hz": 20, "prompt": "open the left drawer", "state_keys": STATE_KEYS,
        "camera_keys": CAMERAS, "action_horizon": 50, "action_dim": 12,
        "normalization": "same 28-H50-derived z-score stats for A and C",
        "devices": [str(device) for device in jax.devices()],
        "gpu": gpu,
    }
    (args.output / "eval_config.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (args.output / "native_config_snapshot.json").write_text(json.dumps(
        dataclasses.asdict(cfg), indent=2,
        default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)) + "\n")
    episodes = []
    initial_conditions = []
    for seed in SEEDS:
        env = writer = None
        try:
            np.random.seed(seed)
            policy._rng = jax.random.key(seed)
            env = RoboCasaGymEnv(env_name="OpenDrawer", split="all", control_freq=20,
                                 camera_widths=256, camera_heights=256,
                                 obj_registries=("lightwheel",))
            raw, _ = env.reset(seed=seed)
            if raw["annotation.human.task_description"].strip().lower().rstrip(".") != "open the left drawer":
                raise ValueError(f"Wrong task for seed {seed}")
            if env.env.control_freq != 20:
                raise ValueError("Wrong control frequency")
            first = observation(raw)
            image_sha = {key: hashlib.sha256(first[key].tobytes()).hexdigest() for key in CAMERAS}
            initial = {"seed": seed, "state": first["observation/state"].tolist(),
                       "image_sha256": image_sha,
                       "task": raw["annotation.human.task_description"]}
            initial_conditions.append(initial)
            (args.output / "initial_conditions.json").write_text(json.dumps(initial_conditions, indent=2) + "\n")
            ep_dir = args.output / f"seed_{seed}"
            ep_dir.mkdir()
            writer = cv2.VideoWriter(str(ep_dir / "rollout.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     20, (768, 256))
            if not writer.isOpened():
                raise RuntimeError("Video writer failed")

            def frame(obs):
                rgb = np.concatenate([obs["video.robot0_agentview_left"],
                                      obs["video.robot0_agentview_right"],
                                      obs["video.robot0_eye_in_hand"]], axis=1)
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            frame(raw)
            queue = collections.deque()
            replans = clips = 0
            start = time.perf_counter()
            actions = []
            with (ep_dir / "steps.jsonl").open("w") as trace:
                for step in range(1, args.horizon + 1):
                    if not queue:
                        chunk = np.asarray(policy.infer(observation(raw))["actions"])
                        if chunk.shape != (50, 12) or not np.isfinite(chunk).all():
                            raise ValueError(f"Invalid action chunk: {chunk.shape}")
                        queue.extend(chunk[:args.replan_steps])
                        replans += 1
                    action = queue.popleft()
                    actions.append(action)
                    clips += int((np.abs(action) > 1).sum())
                    raw, reward, done, truncated, info = env.step(convert_action(np.clip(action, -1, 1)))
                    success = bool(info["success"])
                    frame(raw)
                    trace.write(json.dumps({"step": step, "success": success,
                                            "reward": float(reward), "replans": replans,
                                            "action": action.tolist()}) + "\n")
                    if step == 1 or step % 100 == 0 or success:
                        print(f"PROGRESS seed={seed} step={step}/750 replans={replans} "
                              f"clips={clips} success={success} elapsed_s={time.perf_counter()-start:.1f}",
                              flush=True)
                    if success or done or truncated:
                        break
            actions = np.asarray(actions)
            row = {"seed": seed, "success": success, "steps": step,
                   "reward": float(reward), "replans": replans,
                   "clipped_count": clips, "wall_clock_s": time.perf_counter() - start,
                   "video": str(ep_dir / "rollout.mp4"),
                   "action_mean": actions.mean(0).tolist(),
                   "action_std": actions.std(0).tolist()}
            (ep_dir / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
            episodes.append(row)
            (args.output / "episodes.json").write_text(json.dumps(episodes, indent=2) + "\n")
            print("EPISODE_STATS " + json.dumps(row), flush=True)
            print(f"PROGRESS episodes={len(episodes)}/20 successes={sum(x['success'] for x in episodes)}",
                  flush=True)
        finally:
            if writer is not None:
                writer.release()
            if env is not None:
                env.close()
    report = {"episodes": 20, "successes": sum(x["success"] for x in episodes),
              "seeds": list(SEEDS), "checkpoint": str(checkpoint),
              "norm_sha256": base["norm_sha256"], "episode_stats": episodes}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.lora:
        baseline = args.output.parent / "base" / "initial_conditions.json"
        if not baseline.is_file():
            raise FileNotFoundError("Matched base initial conditions are missing")
        first = json.loads(baseline.read_text())
        if len(first) != 20:
            raise ValueError("Incomplete base initial conditions")
        for base_initial, lora_initial in zip(first, initial_conditions, strict=True):
            if base_initial["seed"] != lora_initial["seed"] or base_initial["task"] != lora_initial["task"]:
                raise ValueError("Seed or task mismatch")
            if not np.allclose(base_initial["state"], lora_initial["state"], rtol=0, atol=1e-5):
                raise ValueError(f"Initial state mismatch at seed {base_initial['seed']}")
    print(f"FINAL_EVAL model={protocol['model']} successes={report['successes']}/20 output={args.output}",
          flush=True)


if __name__ == "__main__":
    main()
