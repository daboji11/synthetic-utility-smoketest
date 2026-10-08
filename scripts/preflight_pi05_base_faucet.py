#!/usr/bin/env python3
"""CPU preflight for the clean pi05_base / 54-H50 smoke test; no model run."""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pyarrow.parquet as pq

from pi05_base_faucet_config import (  # noqa: E402
    BASE, CAMERAS, DATASET, ROOT, SEEDS, STATE_ORDER,
    expected_norm_stats, make_config, official_base_params_uri,
    verify_base_checkpoint, verify_dataset, verify_source,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-base", action="store_true", help="Require official weights to be downloaded")
    args = parser.parse_args()
    info = verify_dataset()
    expected = expected_norm_stats()["norm_stats"]
    actions = []
    states = []
    for path in sorted(DATASET.glob("data/**/*.parquet")):
        table = pq.read_table(path, columns=["action", "observation.state", "episode_index", "task_index"])
        if len(set(table["episode_index"].to_pylist())) != 1 or set(table["task_index"].to_pylist()) != {0}:
            raise ValueError(f"Mixed episode/task rows: {path}")
        actions.append(np.asarray(table["action"].to_pylist(), dtype=np.float32))
        states.append(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32))
    action = np.concatenate(actions)
    state_raw = np.concatenate(states)
    state = state_raw[:, STATE_ORDER]
    if action.shape != (12070, 12) or state.shape != (12070, 16):
        raise ValueError("Wrong action/state shape")
    if not np.isfinite(action).all() or not np.isfinite(state).all():
        raise ValueError("Nonfinite action/state")
    for key, values in (("actions", action), ("state", state)):
        mean = np.asarray(expected[key]["mean"][:values.shape[1]])
        std = np.asarray(expected[key]["std"][:values.shape[1]])
        if not np.allclose(values.astype(np.float64).mean(0), mean, atol=2e-5):
            raise ValueError(f"{key} mean/order mismatch")
        if not np.allclose(values.astype(np.float64).std(0), std, atol=2e-5):
            raise ValueError(f"{key} std/order mismatch")
        if np.any(np.abs(values[:, std == 0] - mean[std == 0]) > 1e-7):
            raise ValueError(f"{key} zero-variance dimension contains noise")
        normalized = (values - mean) / (std + 1e-6)
        if np.max(np.abs(normalized)) > 100:
            raise ValueError(f"{key} normalization produces an outlier over 100σ")
    if set(np.unique(action[:, 6])) != {-1.0} or set(np.unique(action[:, 11])) != {-1.0}:
        raise ValueError("Gripper/control-mode value mismatch")
    if np.any(action[:, 7:11] != 0):
        raise ValueError("Unexpected base motion in original H50 demos")
    from openpi.groot_utils.groot_openpi_dataset import GrootOpenpiSingleDataset

    reader = GrootOpenpiSingleDataset(
        {"path": str(DATASET), "filter_key": None}, action_horizon=50)
    if len(reader) != 12070:
        raise ValueError("Groot/OpenPI reader length mismatch")
    for index in (0, len(reader) - 1):
        sample = reader[index]
        if (sample["prompt"] != "turn on the sink faucet"
                or sample["actions"].shape != (50, 12)
                or not np.allclose(sample["actions"][0], action[index])
                or not np.allclose(sample["observation/state"], state[index])):
            raise ValueError(f"Groot/OpenPI sample mismatch at frame {index}")
        for key in ("observation/image", "observation/wrist_image",
                    "observation/right_image"):
            if sample[key].shape != (256, 256, 3):
                raise ValueError(f"Groot/OpenPI camera mismatch: {key}")
    report = {
        "passed": True, "dataset": str(DATASET), "episodes": 54,
        "frames": 12070, "fps": info["fps"], "source": "H50 human only",
        "action_dim": 12, "state_dim": 16, "action_horizon": 50,
        "seeds": list(SEEDS), "control_hz": 20,
        "camera_order": list(CAMERAS.values()), "prompt": "turn on the sink faucet",
        "normalization": "z-score from exactly 54 H50 human demos; state reordered, action identity",
        "official_base_params_uri": official_base_params_uri(),
        "dataset_stats_sha256": hashlib.sha256((DATASET / "meta/stats.json").read_bytes()).hexdigest(),
        "weights_checked": args.check_base,
    }
    if args.check_base:
        verify_source()
        manifest = verify_base_checkpoint()
        cfg = make_config(lora=True)
        if cfg.weight_loader.params_path != str(BASE / "params"):
            raise ValueError("Training loader is not the verified base checkpoint")
        report["norm_sha256"] = manifest["norm_sha256"]
        report["source_commit"] = __import__("pi05_base_faucet_config").SOURCE_COMMIT
    (ROOT / "records/pi05_base_faucet_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PI05_BASE_FAUCET_PREFLIGHT_OK " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
