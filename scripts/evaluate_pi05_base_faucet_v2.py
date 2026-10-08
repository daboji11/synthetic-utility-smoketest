#!/usr/bin/env python3
"""20-seed v2 Faucet rollout; seed-matched IDs do not imply identical scenes."""
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

from pi05_base_faucet_v2_config import (  # noqa: E402
    BASE, CAMERAS, ROOT, SEEDS, SOURCE_COMMIT, STATE_KEYS, DEFAULT_RUN, SAVE_STEPS,
    run_root, norm_fingerprint, verify_saved_norm,
    make_config, observation, official_base_params_uri, verify_base_checkpoint,
    verify_source, verify_5090,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=750)
    parser.add_argument("--replan-steps", type=int, default=5)
    args = parser.parse_args()
    if args.horizon != 750 or args.replan_steps != 5:
        parser.error("This clean comparison fixes horizon=750 and replan-steps=5")
    verify_source()
    gpu = verify_5090()
    verify_base_checkpoint()
    checkpoint = args.checkpoint.resolve()
    training_root = run_root(args.run_id).resolve()
    if checkpoint.parent != training_root or checkpoint.name not in {str(s) for s in SAVE_STEPS}:
        parser.error("Checkpoint must be one of this v2 run's scheduled intermediate/final saves")
    actual = json.loads((training_root / "actual_config.json").read_text())
    if (actual["base_params_uri"] != official_base_params_uri()
            or actual["norm_fingerprint"] != norm_fingerprint()
            or actual["only_lora"] is not True or actual["episodes"] != 54):
        parser.error("Wrong v2 checkpoint provenance")
    if not (checkpoint / "params/_METADATA").is_file():
        raise FileNotFoundError("Incomplete checkpoint")
    saved_norm_sha = verify_saved_norm(checkpoint)
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
    cfg = make_config(run_id=args.run_id)
    policy = policy_config.create_trained_policy(cfg, checkpoint)
    args.output.mkdir(parents=True)
    protocol = {
        "model": "pi05_base + human54 LoRA state-range v2",
        "checkpoint": str(checkpoint), "official_base_params_uri": official_base_params_uri(),
        "norm_sha256": saved_norm_sha, "source_commit": SOURCE_COMMIT,
        "seeds": list(SEEDS), "episodes": 20, "horizon": 750, "replan_steps": 5,
        "control_hz": 20, "prompt": "turn on the sink faucet", "state_keys": STATE_KEYS,
        "camera_keys": CAMERAS, "action_horizon": 50, "action_dim": 12,
        "normalization": "v2 state q01/q99 affine; unchanged action z-score",
        "norm_fingerprint": norm_fingerprint(), "paired_initial_conditions": False,
        "pairing_note": "Same seed IDs; exact scene replay is not validated",
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
            __import__("random").seed(seed)
            np.random.seed(seed)
            policy._rng = jax.random.key(seed)
            env = RoboCasaGymEnv(env_name="TurnOnSinkFaucet", split="all", control_freq=20,
                                 camera_widths=256, camera_heights=256,
                                 obj_registries=("lightwheel",))
            raw, _ = env.reset(seed=seed)
            if raw["annotation.human.task_description"].strip().lower().rstrip(".") != "turn on the sink faucet":
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
            # Capture scene/state for subsequent diagnosis without claiming this
            # is already a validated portable replay bank.
            (ep_dir / "initial_scene.xml").write_text(env.env.sim.model.get_xml())
            np.save(ep_dir / "initial_sim_state.npy", env.env.sim.get_state().flatten())
            (ep_dir / "initial_ep_meta.json").write_text(json.dumps(env.env.get_ep_meta(),
                indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")
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
    report = {"episodes": 20, "paired_initial_conditions": False, "norm_fingerprint": norm_fingerprint(), "successes": sum(x["success"] for x in episodes),
              "seeds": list(SEEDS), "checkpoint": str(checkpoint),
              "norm_sha256": saved_norm_sha, "episode_stats": episodes}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"FINAL_EVAL model={protocol['model']} successes={report['successes']}/20 output={args.output}",
          flush=True)


if __name__ == "__main__":
    main()
