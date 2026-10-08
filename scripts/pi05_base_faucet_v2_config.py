"""Fresh pi05_base, human54, state range fix and a full-budget LR schedule."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from pi05_base_faucet_config import (
    BASE, CAMERAS, DATASET, ROOT, SEEDS, SOURCE_COMMIT, STATE_KEYS, STATE_ORDER,
    expected_norm_stats, make_config as original_config, observation,
    official_base_params_uri, verify_base_checkpoint, verify_dataset, verify_source, verify_5090,
)

NAME = "pi05_base_faucet_state_range_v2"
DEFAULT_RUN = "human54_lr1e5_15000_seed0"
TOTAL_STEPS = 15000
SAVE_STEPS = (1000, 3000, 5000, 7000, 10000, 12000, 15000)
REQUIRED_FREE_BYTES = 50_000_000_000


def check_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("Run ID must contain only letters, digits, underscores and hyphens")
    return run_id


def run_root(run_id: str = DEFAULT_RUN) -> Path:
    return ROOT / "outputs/TurnOnSinkFaucet" / NAME / check_run_id(run_id)


def build_norm_payload(stats: dict) -> dict:
    """Store state affine centers/scales in the upstream mean/std fields.

    For state: center=(q01+q99)/2, scale=(q99-q01)/2. These are deliberately
    NOT empirical means/stds. Actions retain their exact empirical z-score
    normalization. One upstream Normalize/Unnormalize pair handles both keys,
    and the same assets are saved in every checkpoint. No extra action transform.
    """
    state = stats["observation.state"]
    low = np.asarray(state["q01"], dtype=np.float64)
    high = np.asarray(state["q99"], dtype=np.float64)
    if low.shape != (16,) or high.shape != (16,) or not np.isfinite([low, high]).all():
        raise ValueError("Expected finite 16D state quantiles")
    if np.any(high < low):
        raise ValueError("Reversed state quantiles")
    centers = (low + high) / 2
    scales = (high - low) / 2
    # Constant/narrow dimensions carry no resolvable state variation. Avoid
    # amplifying tiny quaternion/base/gripper numerical noise into extreme tokens.
    scales = np.where(scales < 1e-6, 1.0, scales)
    action = stats["action"]
    if len(action["mean"]) != 12 or len(action["std"]) != 12:
        raise ValueError("Expected 12D action statistics")
    if not np.isfinite([action["mean"], action["std"]]).all() or np.any(np.asarray(action["std"]) < 0):
        raise ValueError("Invalid action statistics")
    return {"norm_stats": {
        "state": {"mean": centers[list(STATE_ORDER)].tolist() + [0.0] * 16,
                  "std": scales[list(STATE_ORDER)].tolist() + [1.0] * 16},
        "actions": {"mean": list(action["mean"]) + [0.0] * 20,
                    "std": list(action["std"]) + [1.0] * 20},
    }}


def norm_payload() -> dict:
    verify_dataset()
    payload = build_norm_payload(json.loads((DATASET / "meta/stats.json").read_text()))
    if payload["norm_stats"]["actions"] != expected_norm_stats()["norm_stats"]["actions"]:
        raise ValueError("Action normalization changed")
    return payload


def norm_fingerprint(payload: dict | None = None) -> str:
    encoded = json.dumps(payload or norm_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def verify_saved_norm(checkpoint: Path) -> str:
    from openpi.shared import normalize
    saved = normalize.load(checkpoint / "assets")
    expected = norm_payload()["norm_stats"]
    for key in ("state", "actions"):
        for field in ("mean", "std"):
            if not np.array_equal(getattr(saved[key], field), expected[key][field]):
                raise ValueError(f"Checkpoint normalization differs: {key}.{field}")
    return hashlib.sha256((checkpoint / "assets/norm_stats.json").read_bytes()).hexdigest()


def make_config(*, lora: bool = True, run_id: str = DEFAULT_RUN, probe: bool = False,
                num_workers: int = 4):
    from openpi.shared import normalize
    from openpi.training import optimizer
    payload = norm_payload()
    norm = {key: normalize.NormStats(mean=np.asarray(row["mean"]), std=np.asarray(row["std"]))
            for key, row in payload["norm_stats"].items()}
    cfg = original_config(lora=lora, batch_size=8, steps=TOTAL_STEPS, lr=1e-5,
                          seed=0, exp_name=check_run_id(run_id), num_workers=num_workers)
    # The pinned Groot reader interprets 2_demos as two seed-0-selected episodes.
    # Selection affects the probe only; its stats still come from all 54 humans.
    data = dataclasses.replace(cfg.data,
        data_dirs=[{"path": str(DATASET.resolve()), "filter_key": "2_demos" if probe else None}],
        base_config=dataclasses.replace(cfg.data.base_config, norm_stats=norm, use_quantile_norm=False))
    return dataclasses.replace(cfg, name=NAME + ("_probe" if probe else ""), data=data,
        keep_period=1000, save_interval=1000,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=200, peak_lr=1e-5,
                                                 decay_steps=TOTAL_STEPS, decay_lr=1e-6),
        policy_metadata={**cfg.policy_metadata, "protocol": "faucet_state_range_v2",
                         "norm_fingerprint": norm_fingerprint(payload),
                         "state_scaling": "q01/q99 affine; constant dimensions scale=1",
                         "action_scaling": "unchanged human54 empirical z-score"})


def require_probe(report: dict, fingerprint: str, *, allow_no_gain: bool = False) -> None:
    if (type(report.get("passed")) is not bool
            or (not allow_no_gain and report["passed"] is not True)
            or report.get("norm_fingerprint") != fingerprint
            or report.get("base_params_uri") != official_base_params_uri()
            or report.get("source_commit") != SOURCE_COMMIT
            or report.get("probe_steps") != 300 or report.get("batch_size") != 8):
        raise ValueError("A valid 300-step v2 probe from the same base and stats is required")
    before = report["before"]["normalized_first5_arm_mse"]
    after = report["after"]["normalized_first5_arm_mse"]
    if (not np.isfinite([before, after]).all() or before <= 0 or after < 0
            or (not allow_no_gain and (before - after) / before < 0.10)
            or len(set(report.get("selected_episode_indices", []))) != 2
            or report["before"].get("noise_seeds") != [101, 102, 103]
            or report["after"].get("noise_seeds") != [101, 102, 103]):
        raise ValueError("Invalid fixed-noise probe metrics or missing required action improvement")


def audit_inputs() -> dict:
    """CPU check of real tensor dimensions, finite values and token saturation."""
    import pandas as pd
    payload = norm_payload()["norm_stats"]
    frames = [pd.read_parquet(path, columns=["observation.state", "action"])
              for path in sorted(DATASET.glob("data/**/*.parquet"))]
    raw_state = np.vstack([np.stack(frame["observation.state"]) for frame in frames])
    actions = np.vstack([np.stack(frame["action"]) for frame in frames])
    if raw_state.shape != (12070, 16) or actions.shape != (12070, 12):
        raise ValueError("Actual state/action arrays differ from metadata")
    if not np.isfinite(raw_state).all() or not np.isfinite(actions).all():
        raise ValueError("Nonfinite demonstration values")
    stats = json.loads((DATASET / "meta/stats.json").read_text())
    for key, values in (("observation.state", raw_state), ("action", actions)):
        for field, observed in (("mean", values.mean(0, dtype=np.float64)),
                                ("std", values.std(0, dtype=np.float64))):
            if not np.allclose(observed, stats[key][field], rtol=1e-4, atol=1e-6):
                raise ValueError(f"Stored {key}.{field} differs from the actual human54 arrays")
    state = raw_state[:, list(STATE_ORDER)]
    center = np.asarray(payload["state"]["mean"][:16])
    scale = np.asarray(payload["state"]["std"][:16])
    normalized = (state - center) / (scale + 1e-6)
    # Exact endpoint bins used by the pinned state tokenizer.
    saturated = (normalized < -1) | (normalized >= np.linspace(-1, 1, 257)[-2])
    fractions = saturated[:, :3].mean(0)
    if np.any(fractions > 0.10):
        raise ValueError(f"State fix still saturates over 10% of position tokens: {fractions}")
    return {"frames": len(actions), "state_position_endpoint_fractions": fractions.tolist(),
            "action_max_abs": np.abs(actions).max(0).tolist(),
            "action_stats_match_actual_data": True}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    args = parser.parse_args()
    verify_source()
    verify_base_checkpoint()
    import shutil
    if shutil.disk_usage(ROOT).free < REQUIRED_FREE_BYTES:
        raise RuntimeError("Need 50 GB free for seven 15000-step-run checkpoint saves")
    audit = audit_inputs()
    cfg = make_config(run_id=args.run_id)
    print(json.dumps({"checkpoint": str((BASE / "params").resolve()),
        "official_base": official_base_params_uri(), "dataset": str(DATASET.resolve()),
        "episodes": 54, "batch_size": 8, "steps": TOTAL_STEPS,
        "peak_lr": 1e-5, "tail_lr": 1e-6, "warmup_steps": 200,
        "save_steps": SAVE_STEPS, "only_lora": True, "vision_frozen": True,
        "norm_fingerprint": norm_fingerprint(), "input_audit": audit,
        "output": str(cfg.checkpoint_dir)}, indent=2))
