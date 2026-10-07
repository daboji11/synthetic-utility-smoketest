#!/usr/bin/env python3
"""Explicit 28-human-demo LoRA from untouched pi05_base; checkpoints at 1000/2000/4000."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import shutil
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYNPUT_BACKEND", "dummy")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pi05_base_h50_config import (  # noqa: E402
    BASE, DATASET, ROOT, SEEDS, make_config, official_base_params_uri,
    verify_base_checkpoint, verify_dataset, verify_source, verify_4090,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="Required to update parameters")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if not args.train:
        parser.error("--train is required; preflight and baseline run separately")
    if (args.batch_size, args.steps, args.lr, args.seed) != (4, 4000, 5e-6, 0):
        parser.error("Clean smoke test fixes batch4, 4000 steps, peak LR5e-6, seed0")
    source = verify_source()
    gpu = verify_4090()
    if shutil.disk_usage(ROOT).free < 45_000_000_000:
        raise RuntimeError("Need at least 45 GB free for 1000/2000/4000 checkpoints")
    verify_dataset()
    base = verify_base_checkpoint()
    baseline_path = ROOT / "results/robocasa_eval/pi05_base_h50_smoketest/base/summary.json"
    baseline = json.loads(baseline_path.read_text())
    if baseline["episodes"] != 20 or baseline["seeds"] != list(SEEDS):
        raise ValueError("Complete matched pi05_base 20-seed baseline is required before training")
    if baseline["checkpoint"] != str(BASE.resolve()) or baseline["norm_sha256"] != base["norm_sha256"]:
        raise ValueError("Baseline provenance differs from current pi05_base + H50 interface")
    cfg = make_config(lora=True, batch_size=args.batch_size, steps=args.steps,
                      lr=args.lr, seed=args.seed, exp_name="seed_0", num_workers=args.num_workers)
    if cfg.checkpoint_dir.exists():
        raise FileExistsError(f"Preserving existing experiment {cfg.checkpoint_dir}")

    import jax
    import numpy as np
    from flax.traverse_util import flatten_dict
    from openpi.training import checkpoints, data_loader, sharding

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("CUDA JAX device required")
    spec = importlib.util.spec_from_file_location("pinned_pi05_base_train", source / "scripts/train.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    if cfg.batch_size % jax.device_count():
        raise ValueError("Batch size must divide GPU device count")
    manager, _ = checkpoints.initialize_checkpoint_dir(
        cfg.checkpoint_dir, keep_period=cfg.keep_period, overwrite=False, resume=False)
    run = cfg.checkpoint_dir
    (run / "native_config_snapshot.json").write_text(json.dumps(
        dataclasses.asdict(cfg), indent=2,
        default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)) + "\n")
    (run / "actual_config.json").write_text(json.dumps({
        **vars(args), "base_params_uri": official_base_params_uri(),
        "base_params_local": str((BASE / "params").resolve()),
        "dataset": str(DATASET.resolve()), "episodes": 28, "frames": 5916,
        "gpu": gpu,
        "fps": 20, "action_dim": 12, "state_dim": 16, "action_horizon": 50,
        "only_lora": True, "vision_frozen": True, "ema_decay": None,
        "norm_sha256": hashlib.sha256((BASE / "assets/norm_stats.json").read_bytes()).hexdigest(),
        "normalization": "same 28-H50-derived z-score stats as untouched base baseline",
        "checkpoint_dir": str(run),
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
    if not trainables or any("lora" not in key.lower() for key in trainables):
        raise ValueError("Trainable parameters include something outside LoRA")
    (run / "trainable_parameters.json").write_text(json.dumps(
        {key: list(value.shape) for key, value in trainables.items()}, indent=2) + "\n")
    step_fn = jax.jit(
        __import__("functools").partial(upstream.train_step, cfg),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated), donate_argnums=(1,))
    schedule = cfg.lr_schedule.create()
    print("PI05_BASE_H50_LORA_START " + json.dumps({
        "batch_size": args.batch_size, "steps": args.steps, "peak_lr": args.lr,
        "trainable_parameters": sum(value.size for value in trainables.values()),
        "base_params_uri": official_base_params_uri(), "output": str(run)}), flush=True)
    with (run / "metrics.jsonl").open("x") as metrics:
        for step in range(1, cfg.num_train_steps + 1):
            tic = time.perf_counter()
            with sharding.set_mesh(mesh):
                state, details = step_fn(train_rng, state, batch)
            details = jax.device_get(details)
            if any(not np.isfinite(value).all() for value in details.values()):
                raise FloatingPointError(f"Nonfinite metrics at step {step}")
            loss = float(details["loss"])
            if step <= 10 and loss > 100:
                raise ValueError(f"Abnormally large early loss {loss}; stop and recheck normalization")
            row = {"step": step, "train_loss": loss,
                   "learning_rate": float(schedule(step - 1)),
                   "compute_seconds": time.perf_counter() - tic,
                   "wall_clock_s": time.perf_counter() - started}
            metrics.write(json.dumps(row) + "\n")
            if step <= 3 or step % cfg.log_interval == 0:
                metrics.flush()
                print("TRAIN_STEP " + json.dumps(row), flush=True)
            if step in (1000, 2000, 4000):
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
        "updates": cfg.num_train_steps, "checkpoint_1000": str(run / "1000"),
        "checkpoint_2000": str(run / "2000"),
        "final_checkpoint": str(run / "4000"), "wall_clock_s": time.perf_counter() - started,
        "batch_size": cfg.batch_size, "only_lora": True,
    }, indent=2) + "\n")
    print("PI05_BASE_H50_LORA_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
