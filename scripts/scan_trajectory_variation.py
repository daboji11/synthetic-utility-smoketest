"""Offline state/action variation scan for all DreamGen RoboCasa tasks."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/raw/huggingface"
OUTPUT = ROOT / "results/dataset_inspection"
SYNTHETIC_REPO = "Keh0t0--dreamgen_robocasa_p3_init0_v2_droid"
REAL_REPO = "DAVIAN-Robotics--robocasa-MG_30"
STEPS = 64


def snapshot(repo: str) -> Path:
    matches = [p for p in (CACHE / f"datasets--{repo}" / "snapshots").iterdir() if p.is_dir()]
    if len(matches) != 1:
        raise ValueError(f"Expected one cached snapshot of {repo}, found {len(matches)}")
    return matches[0]


def aligned(values: np.ndarray) -> np.ndarray:
    x = np.linspace(0, 1, len(values))
    target = np.linspace(0, 1, STEPS)
    return np.stack([np.interp(target, x, values[:, dim]) for dim in range(values.shape[1])], axis=-1)


def mean_variance(values: np.ndarray) -> float:
    return float(np.var(values, axis=0, ddof=1).mean())


def pairwise_ee_distances(paths: np.ndarray) -> np.ndarray:
    # RMS Euclidean end-effector position separation in meters, aligned by progress.
    delta = paths[:, None, :, :] - paths[None, :, :, :]
    matrix = np.sqrt(np.mean(np.sum(delta * delta, axis=-1), axis=-1))
    return matrix[np.triu_indices(len(paths), k=1)]


def percentile_rank(values: np.ndarray) -> np.ndarray:
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    ranks = (starts + (counts - 1) / 2)[inverse]
    return ranks / (len(values) - 1)


def main() -> None:
    synthetic = snapshot(SYNTHETIC_REPO)
    real = snapshot(REAL_REPO)
    info = json.loads((synthetic / "meta/info.json").read_text())
    expected_ee = [f"robot0_base_to_eef_pos_{axis}" for axis in range(3)]
    if info["features"]["observation.state"]["names"][:3] != expected_ee:
        raise ValueError("The first three state channels are not the expected end-effector position")
    if info["features"]["action"]["shape"] != [12]:
        raise ValueError("Unexpected action dimension")
    episodes = pq.read_table(synthetic / "meta/episodes/chunk-000/file-000.parquet",
                             columns=["episode_index", "tasks", "length", "dataset_from_index", "dataset_to_index"]).to_pylist()
    real_episodes = pq.read_table(real / "meta/episodes/chunk-000/file-000.parquet",
                                  columns=["tasks"]).to_pylist()
    real_instructions = Counter(task for episode in real_episodes for task in episode["tasks"])
    overlap = {row["task_name"]: row for row in csv.DictReader((OUTPUT / "task_overlap_scan.csv").open(newline=""))}

    table = pq.read_table(synthetic / "data/chunk-000/file-000.parquet",
                          columns=["observation.state", "action"])
    state = table["observation.state"].combine_chunks().values.to_numpy().reshape(len(table), 16)
    action = table["action"].combine_chunks().values.to_numpy().reshape(len(table), 12)
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("Non-finite state or action values")

    groups = defaultdict(list)
    for episode in episodes:
        if len(episode["tasks"]) != 1:
            raise ValueError(f"Expected one instruction for episode {episode['episode_index']}")
        instruction = episode["tasks"][0]
        start, stop = episode["dataset_from_index"], episode["dataset_to_index"]
        if stop - start != episode["length"]:
            raise ValueError(f"Length mismatch in episode {episode['episode_index']}")
        groups[instruction].append({"id": episode["episode_index"], "length": episode["length"],
                                    "initial": state[start], "state": aligned(state[start:stop]),
                                    "action": aligned(action[start:stop])})

    rows = []
    for task_name, previous in overlap.items():
        instruction = previous["synthetic_instruction"]
        items = groups.pop(instruction)
        initial = np.stack([item["initial"] for item in items])
        state_paths = np.stack([item["state"] for item in items])
        action_paths = np.stack([item["action"] for item in items])
        ee_paths = state_paths[:, :, :3]  # robot0_base_to_eef_pos_x/y/z from feature names
        relative_ee_paths = ee_paths - ee_paths[:, :1, :]
        lengths = np.array([item["length"] for item in items])
        pairwise = pairwise_ee_distances(ee_paths)
        exact_real = real_instructions[instruction]
        if exact_real != int(previous["real_exact_match_count"]):
            raise ValueError(f"Real exact-match metadata changed for {task_name}")
        rows.append({
            "task_name": task_name, "synthetic_instruction": instruction,
            "synthetic_episodes": len(items),
            "real_family_episodes": int(previous["real_family_episode_count"]),
            "real_exact_match_episodes": exact_real,
            "real_matching_episodes": int(previous["matching_real_episode_count"]),
            "initial_state_var_mean": mean_variance(initial),
            "initial_ee_pos_var_m2": mean_variance(initial[:, :3]),
            "action_sequence_var_mean": mean_variance(action_paths),
            "motion_action_sequence_var_mean": mean_variance(action_paths[:, :, :6]),
            "gripper_action_sequence_var": mean_variance(action_paths[:, :, 6:7]),
            "ee_pos_trajectory_var_m2": mean_variance(ee_paths),
            "ee_relative_path_var_m2": mean_variance(relative_ee_paths),
            "episode_length_mean_frames": float(lengths.mean()),
            "episode_length_std_frames": float(lengths.std(ddof=1)),
            "episode_length_var_frames2": float(lengths.var(ddof=1)),
            "pairwise_ee_trajectory_distance_mean_m": float(pairwise.mean()),
            "pairwise_ee_trajectory_distance_var_m2": float(pairwise.var(ddof=1)),
        })
    if groups or len(rows) != 24:
        raise ValueError(f"Task mismatch: {len(rows)} scanned, {len(groups)} unmatched instructions")

    # Relative ranks are exploratory and equally reproducible across the same 24 tasks.
    weights = {"initial_ee_pos_var_m2": .10, "motion_action_sequence_var_mean": .25,
               "ee_relative_path_var_m2": .35, "pairwise_ee_trajectory_distance_mean_m": .25,
               "episode_length_std_frames": .05}
    varying_weights = {key: weight for key, weight in weights.items()
                       if np.ptp([row[key] for row in rows]) > 0}
    score = np.zeros(len(rows))
    for key, weight in varying_weights.items():
        score += weight * percentile_rank(np.array([row[key] for row in rows]))
    score /= sum(varying_weights.values())
    for row, value in zip(rows, score):
        row["variation_score_0_100"] = round(float(value * 100), 1)
    rows.sort(key=lambda row: (-row["variation_score_0_100"], -row["real_matching_episodes"]))

    path = OUTPUT / "trajectory_variation_scan.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    method = {
        "task_count": len(rows), "synthetic_episode_count": len(episodes),
        "cached_synthetic_revision": synthetic.name, "cached_real_revision": real.name,
        "source_files": [str(synthetic / "data/chunk-000/file-000.parquet"),
                         str(synthetic / "meta/episodes/chunk-000/file-000.parquet"),
                         str(real / "meta/episodes/chunk-000/file-000.parquet")],
        "time_alignment": f"linear interpolation to {STEPS} normalized-progress samples per episode",
        "initial_state_var_mean": "Mean sample variance across all 16 initial observation.state channels; raw mixed units squared",
        "action_sequence_var_mean": "Mean sample variance across episodes, 64 progress points and all 12 action channels; raw mixed units squared",
        "motion_action_sequence_var_mean": "Same variance restricted to action channels 0:6, inferred motion channels from observed magnitudes; action names are generic",
        "gripper_action_sequence_var": "Same variance for action channel 6 alone; interpreted as gripper-like from observed values, not explicit feature names",
        "ee_pos_trajectory_var_m2": "Mean sample variance across episodes, progress points and the first three state channels (base-frame end-effector position)",
        "ee_relative_path_var_m2": "Same end-effector position variance after subtracting each episode's initial position",
        "pairwise_ee_trajectory_distance_mean_m": "Across all unordered pairs: RMS Euclidean separation of aligned base-frame end-effector positions, in meters",
        "pairwise_ee_trajectory_distance_var_m2": "Sample variance of those pairwise distances",
        "score": "Weighted within-24-task percentile ranks with tied ranks averaged: initial EE position 10%, motion-action sequence (channels 0:6) 25%, relative EE path 35%, pairwise EE distance 25%, episode-length std 5%. Constant metrics are omitted and remaining weights renormalized. Exploration aid, not utility/quality label.",
        "real_matching_episodes": "Previous task-family overlap scan; exact instruction counts rechecked against cached real episode metadata",
        "limitations": ["No object pose/state channels in observation.state; appearance/object variation cannot be measured",
                        "Progress alignment hides timing differences; length spread is reported separately",
                        "State/action variance does not establish task success, semantic diversity, or training utility"],
    }
    (OUTPUT / "trajectory_variation_method.json").write_text(json.dumps(method, indent=2) + "\n")
    print("Wrote", path)
    for row in rows:
        print(f"{row['task_name']:26s} score={row['variation_score_0_100']:5.1f} "
              f"match={row['real_matching_episodes']:2d} initial={row['initial_state_var_mean']:.4f} "
              f"action={row['action_sequence_var_mean']:.4f} ee={row['ee_pos_trajectory_var_m2']:.5f} "
              f"relative={row['ee_relative_path_var_m2']:.5f} "
              f"length_sd={row['episode_length_std_frames']:.1f} "
              f"pair={row['pairwise_ee_trajectory_distance_mean_m']:.3f}±{np.sqrt(row['pairwise_ee_trajectory_distance_var_m2']):.3f}")


if __name__ == "__main__":
    main()
