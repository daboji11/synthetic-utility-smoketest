#!/usr/bin/env python3
"""Fresh-base v2: two-demo learning probe, then human54 LoRA for 7000 steps."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYNPUT_BACKEND", "dummy")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pi05_base_faucet_v2_config import (  # noqa: E402
    BASE, DATASET, ROOT, SOURCE_COMMIT, DEFAULT_RUN, NAME, SAVE_STEPS, make_config,
    norm_payload, norm_fingerprint, require_probe, official_base_params_uri,
    verify_base_checkpoint, verify_dataset, verify_source, verify_5090,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="Required to update parameters")
    parser.add_argument("--probe", action="store_true", help="300 updates on two human demos")
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if not args.train:
        parser.error("--train is required")
    source = verify_source()
    gpu = verify_5090()
    verify_dataset()
    verify_base_checkpoint()
    cfg = make_config(run_id=args.run_id, probe=args.probe, num_workers=args.num_workers)
    budget = 300 if args.probe else 7000
    fingerprint = norm_fingerprint()
    probe_root = ROOT / "outputs/TurnOnSinkFaucet" / (NAME + "_probe") / args.run_id
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if not args.probe:
        report = json.loads((probe_root / "probe_summary.json").read_text())
        require_probe(report, fingerprint)
        if report.get("training_script_sha256") != script_sha:
            raise ValueError("Probe was run with different training code")
        if shutil.disk_usage(ROOT).free < 30_000_000_000:
            raise RuntimeError("Need 30 GB free for four v2 checkpoints; preserve old runs separately")
    if cfg.checkpoint_dir.exists():
        raise FileExistsError(f"Preserving existing experiment {cfg.checkpoint_dir}; use a new FAUCET_V2_RUN_ID")

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
    run = cfg.checkpoint_dir
    manager = None
    if args.probe:
        run.mkdir(parents=True)
    else:
        manager, _ = checkpoints.initialize_checkpoint_dir(
            run, keep_period=cfg.keep_period, overwrite=False, resume=False)
    (run / "native_config_snapshot.json").write_text(json.dumps(
        dataclasses.asdict(cfg), indent=2,
        default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)) + "\n")
    (run / "normalization_recipe.json").write_text(json.dumps({
        "state": "center=(q01+q99)/2, scale=(q99-q01)/2; narrow dimensions scale=1",
        "actions": "unchanged empirical human54 mean/std", "use_quantile_norm": False,
        "note": "state mean/std asset fields encode affine centers/scales, not empirical statistics",
        "norm_fingerprint": fingerprint, "payload": norm_payload(),
    }, indent=2) + "\n")
    (run / "actual_config.json").write_text(json.dumps({
        **vars(args), "base_params_uri": official_base_params_uri(),
        "base_params_local": str((BASE / "params").resolve()),
        "dataset": str(DATASET.resolve()), "episodes": 2 if args.probe else 54,
        "dataset_filter": "2_demos, pinned seed0" if args.probe else None,
        "gpu": gpu, "fps": 20, "action_dim": 12, "state_dim": 16, "action_horizon": 50,
        "prompt": "turn on the sink faucet", "batch_size": 8, "steps": budget,
        "peak_lr": 1e-5, "tail_lr": 1e-6, "warmup_steps": 200, "decay_steps": 7000,
        "only_lora": True, "vision_frozen": True, "ema_decay": None,
        "norm_fingerprint": fingerprint, "source_commit": SOURCE_COMMIT,
        "training_script_sha256": script_sha, "fresh_optimizer": True,
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
    def measured_step(rng, old_state, batch):
        import optax
        new_state, details = upstream.train_step(cfg, rng, old_state, batch)
        delta = jax.tree.map(lambda a, b: a - b,
            new_state.params.filter(cfg.trainable_filter), old_state.params.filter(cfg.trainable_filter))
        return new_state, {**details, "update_norm": optax.global_norm(delta)}

    step_fn = jax.jit(
        measured_step,
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated), donate_argnums=(1,))
    schedule = cfg.lr_schedule.create()
    probe_before = None
    if args.probe:
        from flax import nnx
        data_cfg = loader.data_config()
        raw_dataset = data_loader.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)
        selected = [int(x) for x in raw_dataset.trajectory_ids]
        if len(selected) != 2:
            raise ValueError(f"Probe must use exactly two demos, got {selected}")
        # The fixed batch contains training observations. This measures learning
        # response, not held-out generalization or scene-replayed closed-loop fit.
        probe_batch = batch
        np.savez_compressed(run / "probe_observations.npz",
            state=np.asarray(probe_batch[0].state), actions=np.asarray(probe_batch[1]),
            **{f"image_{k}": np.asarray(v) for k, v in probe_batch[0].images.items()},
            tokens=np.asarray(probe_batch[0].tokenized_prompt))
        model_def = state.model_def

        def sample(params, obs, key):
            model = nnx.merge(model_def, params)
            model.eval()
            return model.sample_actions(key, obs)

        sample_fn = jax.jit(sample,
            in_shardings=(state_sharding.params, data_sharding, replicated),
            out_shardings=data_sharding)
        scales = np.asarray(norm_payload()["norm_stats"]["actions"]["std"][:6]) + 1e-6

        def action_errors(params):
            errors = []
            target = np.asarray(probe_batch[1])
            for noise_seed in (101, 102, 103):
                predicted = np.asarray(sample_fn(params, probe_batch[0], jax.random.key(noise_seed)))
                if predicted.shape != target.shape or not np.isfinite(predicted).all():
                    raise ValueError("Invalid fixed-observation action prediction")
                errors.append((predicted - target)[..., :6])
            errors = np.stack(errors)
            return {"normalized_first5_arm_mse": float(np.mean(errors[:, :, :5] ** 2)),
                "raw_arm_rmse_per_dim": {
                    str(h): np.sqrt(np.mean((errors[:, :, :h] * scales) ** 2, axis=(0, 1, 2))).tolist()
                    for h in (1, 5, 50)}, "noise_seeds": [101, 102, 103]}

        print("PROBE_ACTION_ERROR_BEFORE_START (first inference compiles)", flush=True)
        probe_before = action_errors(state.params)
        print("PROBE_BEFORE " + json.dumps(probe_before), flush=True)
    print("PI05_BASE_FAUCET_V2_START " + json.dumps({
        "probe": args.probe, "batch_size": 8, "steps": budget, "peak_lr": 1e-5,
        "trainable_parameters": sum(value.size for value in trainables.values()),
        "base_params_uri": official_base_params_uri(), "output": str(run)}), flush=True)
    with (run / "metrics.jsonl").open("x") as metrics:
        for step in range(1, budget + 1):
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
                   "grad_norm": float(details["grad_norm"]), "update_norm": float(details["update_norm"]),
                   "learning_rate": float(schedule(step - 1)),
                   "compute_seconds": time.perf_counter() - tic,
                   "wall_clock_s": time.perf_counter() - started}
            metrics.write(json.dumps(row) + "\n")
            if step <= 3 or step % cfg.log_interval == 0:
                metrics.flush()
                print("TRAIN_STEP " + json.dumps(row), flush=True)
            if not args.probe and step in SAVE_STEPS:
                checkpoints.save_state(manager, state, loader, step)
                manager.wait_until_finished()
                last = run / "last"
                if last.is_symlink():
                    last.unlink()
                elif last.exists():
                    raise FileExistsError(f"Preserving non-symlink {last}")
                last.symlink_to(str(step), target_is_directory=True)
                print(f"CHECKPOINT_SAVED step={step} path={run / str(step)}", flush=True)
            if step < budget:
                batch = next(iterator)
    if args.probe:
        after = action_errors(state.params)
        initial = probe_before["normalized_first5_arm_mse"]
        final = after["normalized_first5_arm_mse"]
        relative_gain = (initial - final) / max(initial, 1e-12)
        passed = bool(np.isfinite(relative_gain) and relative_gain >= 0.10)
        report = {"passed": passed, "probe_steps": 300, "batch_size": 8,
            "base_params_uri": official_base_params_uri(), "source_commit": SOURCE_COMMIT,
            "training_script_sha256": script_sha, "norm_fingerprint": fingerprint,
            "selected_episode_indices": selected, "before": probe_before, "after": after,
            "relative_gain": relative_gain, "minimum_relative_gain": 0.10,
            "interpretation": "training-batch action learning response only; not a generalization test"}
        (run / "probe_summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print("PROBE_RESULT " + json.dumps(report), flush=True)
        if not passed:
            raise RuntimeError("Probe action error did not improve by 10%; full training stopped. Inspect probe_summary.json")
        print("PI05_BASE_FAUCET_V2_PROBE_COMPLETE", flush=True)
    else:
        manager.wait_until_finished()
        manager.close()
        (run / "training_summary.json").write_text(json.dumps({
            "updates": budget, "save_steps": SAVE_STEPS,
            "final_checkpoint": str(run / "7000"), "wall_clock_s": time.perf_counter() - started,
            "batch_size": 8, "only_lora": True, "norm_fingerprint": fingerprint,
            "base_params_uri": official_base_params_uri(),
        }, indent=2) + "\n")
        print("PI05_BASE_FAUCET_V2_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
