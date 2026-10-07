#!/usr/bin/env python3
"""Train one independent Pi0.5 LoRA adapter on exactly eight synthetic OpenDrawer episodes."""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYNPUT_BACKEND", "dummy")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from pi05_h50_config import CHECKPOINT, ROOT, make_config, verify_source

def verify_dataset(path: Path, group: str) -> dict:
    info = json.loads((path / "meta/info.json").read_text())
    manifest = json.loads((path / "provenance_manifest.json").read_text())
    episodes = [json.loads(s) for s in (path / "meta/episodes.jsonl").read_text().splitlines() if s]
    tasks = [json.loads(s) for s in (path / "meta/tasks.jsonl").read_text().splitlines() if s]
    if (info["fps"], info["total_episodes"], info["total_frames"], info["total_videos"]) != (20, 8, 2792, 24):
        raise ValueError("Expected precisely eight 349-frame episodes at 20 Hz with three cameras")
    if len(episodes) != 8 or any(row["length"] != 349 for row in episodes):
        raise ValueError("Episode lengths differ")
    if [row["task"] for row in tasks] != ["open the left drawer"]:
        raise ValueError("Task prompt mismatch")
    if (manifest["group"] != group or manifest["episodes"] != 8
        or manifest["source_fps"] != 16 or manifest["fps"] != 20
        or len(manifest["source_episode_indices"]) != 8
        or manifest["selection_status"] != "EXPLORATORY_HUMAN_OVERRIDES_AND_PROVISIONAL_VISUAL"):
        raise ValueError("Synthetic group provenance mismatch")
    if len(list(path.glob("videos/**/*.mp4"))) != 24 or len(list(path.glob("data/**/*.parquet"))) != 8:
        raise ValueError("Incomplete synthetic group files")
    return manifest

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", action="store_true", help="Required to start optimizer updates")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--group", choices=("redundant", "diverse", "random"), required=True)
    p.add_argument("--exp-name", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    a = p.parse_args()
    if not a.train:
        p.error("--train is required")
    if a.steps < 2 or a.batch_size < 1 or a.lr <= 0:
        p.error("Positive training hyperparameters required")
    dataset = a.dataset.resolve()
    manifest = verify_dataset(dataset, a.group)
    source = verify_source()
    if not (CHECKPOINT / "source_manifest.json").exists():
        raise FileNotFoundError("Pinned vanilla Pi0.5 checkpoint is incomplete")
    cfg = make_config(dataset=dataset, batch_size=a.batch_size, steps=a.steps, lr=a.lr,
                      seed=a.seed, exp_name=a.exp_name, num_workers=a.num_workers)
    cfg = dataclasses.replace(
        cfg, name="pi05_synthetic_8_lora", save_interval=a.steps, keep_period=a.steps,
        policy_metadata={
            "task": "OpenDrawer", "instruction": "open the left drawer",
            "training_data": f"8 synthetic DreamGen episodes, group={a.group}",
            "fps": 20, "lora_only": True,
            "source_episode_indices": manifest["source_episode_indices"],
            "source_selection_sha256": manifest["selection_sha256"],
        })
    if cfg.checkpoint_dir.exists():
        raise FileExistsError(f"Preserving existing experiment {cfg.checkpoint_dir}")

    import jax
    from flax.traverse_util import flatten_dict
    from openpi.training import checkpoints, data_loader, sharding
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("CUDA JAX device required")
    module_spec = importlib.util.spec_from_file_location("pinned_pi05_train", source / "scripts/train.py")
    upstream = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(upstream)
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    if cfg.batch_size % jax.device_count():
        raise ValueError("Batch size not divisible by GPU count")
    manager, _ = checkpoints.initialize_checkpoint_dir(
        cfg.checkpoint_dir, keep_period=cfg.keep_period, overwrite=False, resume=False)
    run = cfg.checkpoint_dir
    (run / "native_config_snapshot.json").write_text(
        json.dumps(dataclasses.asdict(cfg), indent=2,
                   default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)) + "\n")
    (run / "actual_config.json").write_text(json.dumps({
        **{name: str(value) if isinstance(value, Path) else value for name, value in vars(a).items()},
        "dataset": str(dataset), "checkpoint": str(CHECKPOINT),
        "episodes": 8, "frames": 2792, "fps": 20,
        "source_episode_indices": manifest["source_episode_indices"],
        "selection_sha256": manifest["selection_sha256"],
        "only_lora": True, "vision_frozen": True, "ema_decay": None,
        "normalization": "unchanged native pretrained z-score stats",
        "normalizer_sha256": hashlib.sha256((CHECKPOINT / "assets/norm_stats.json").read_bytes()).hexdigest(),
        "checkpoint_dir": str(cfg.checkpoint_dir),
    }, indent=2) + "\n")
    started = time.perf_counter()
    loader = data_loader.create_data_loader(cfg, sharding=data_sharding, shuffle=True)
    iterator = iter(loader)
    batch = next(iterator)
    rng = jax.random.key(cfg.seed)
    train_rng, init_rng = jax.random.split(rng)
    state, state_sharding = upstream.init_train_state(cfg, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    trainables = flatten_dict(state.params.filter(cfg.trainable_filter).to_pure_dict(), sep="/")
    if not trainables or any("lora" not in key for key in trainables):
        raise ValueError("Unexpected trainable parameter outside LoRA")
    (run / "trainable_parameters.json").write_text(
        json.dumps({key: list(value.shape) for key, value in trainables.items()}, indent=2) + "\n")
    step_fn = jax.jit(
        __import__("functools").partial(upstream.train_step, cfg),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated), donate_argnums=(1,))
    schedule = cfg.lr_schedule.create()
    print("PI05_SYNTHETIC_LORA_START " + json.dumps({
        "group": a.group, "batch": a.batch_size, "steps": a.steps, "lr": a.lr,
        "trainable": sum(value.size for value in trainables.values()),
        "output": str(run)}), flush=True)
    with (run / "metrics.jsonl").open("x") as log:
        for step in range(1, cfg.num_train_steps + 1):
            tic = time.perf_counter()
            with sharding.set_mesh(mesh):
                state, details = step_fn(train_rng, state, batch)
            details = jax.device_get(details)
            if any(not __import__("numpy").isfinite(value).all() for value in details.values()):
                raise FloatingPointError(f"Nonfinite metrics at step {step}")
            record = {"step": step, "train_loss": float(details["loss"]),
                      "learning_rate": float(schedule(step - 1)),
                      "compute_seconds": time.perf_counter() - tic,
                      "wall_clock_s": time.perf_counter() - started}
            log.write(json.dumps(record) + "\n")
            if step <= 3 or step % cfg.log_interval == 0:
                log.flush()
                print("TRAIN_STEP " + json.dumps(record), flush=True)
            if step % cfg.save_interval == 0 or step == cfg.num_train_steps:
                checkpoints.save_state(manager, state, loader, step)
                manager.wait_until_finished()
                last = run / "last"
                if last.is_symlink():
                    last.unlink()
                elif last.exists():
                    raise FileExistsError(f"Preserving non-symlink {last}")
                last.symlink_to(str(step), target_is_directory=True)
                print(f"CHECKPOINT_SAVED step={step} path={run / str(step)}", flush=True)
            if step < cfg.num_train_steps:
                batch = next(iterator)
    manager.wait_until_finished()
    (run / "training_summary.json").write_text(json.dumps({
        "group": a.group, "updates": cfg.num_train_steps, "final_checkpoint": str(run / "last"),
        "wall_clock_s": time.perf_counter() - started, "batch_size": cfg.batch_size,
        "only_lora": True, "vision_frozen": True,
    }, indent=2) + "\n")
    print("PI05_SYNTHETIC_LORA_COMPLETE", flush=True)

if __name__ == "__main__":
    main()
