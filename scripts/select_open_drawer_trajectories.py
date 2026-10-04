"""Cluster OpenDrawer synthetic trajectories and make two offline preview selections."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "results/dataset_inspection/OpenDrawer"
OUT = TASK_DIR / "trajectory_similarity"
CACHE = ROOT / "data/raw/huggingface/datasets--Keh0t0--dreamgen_robocasa_p3_init0_v2_droid/snapshots"
INSTRUCTION = "open the left drawer"
STEPS = 64
WEIGHTS = {"absolute_state": 0.15, "relative_state": 0.45, "action": 0.40}


def align(values: np.ndarray) -> np.ndarray:
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, STEPS)
    return np.stack([np.interp(target, source, values[:, channel])
                     for channel in range(values.shape[1])], axis=-1)


def quaternion_outer(quaternion: np.ndarray) -> np.ndarray:
    # q and -q have the same outer product and represent the same orientation.
    indices = np.triu_indices(4)
    outer = quaternion[..., :, None] * quaternion[..., None, :]
    return outer[..., indices[0], indices[1]]


def state_features(state: np.ndarray) -> np.ndarray:
    return np.concatenate([state[..., :3], quaternion_outer(state[..., 3:7]),
                           state[..., 7:12], quaternion_outer(state[..., 12:16])], axis=-1)


def normalize_block(block: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str], list[float]]:
    mean = block.mean(axis=(0, 1))
    std = block.std(axis=(0, 1))
    informative = std >= 0.01
    if not informative.any():
        raise ValueError("No informative channels in a similarity block")
    normalized = (block[..., informative] - mean[informative]) / std[informative]
    return normalized, [name for name, keep in zip(names, informative) if keep], std[informative].tolist()


def pairwise_distance(blocks: dict[str, np.ndarray]) -> np.ndarray:
    squared = np.zeros((60, 60), dtype=np.float64)
    for name, block in blocks.items():
        # The mean gives equal total weight to each block, regardless of its channel count.
        delta = block[:, None] - block[None, :]
        squared += WEIGHTS[name] * np.mean(delta * delta, axis=(2, 3))
    distances = np.sqrt(squared)
    np.fill_diagonal(distances, 0.0)
    return distances


def k_medoids(distance: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(distance)
    global_medoid = int(np.argmin(distance.sum(axis=1)))
    starts = [global_medoid] + [int(x) for x in np.random.default_rng(20261004 + k).choice(
        [i for i in range(n) if i != global_medoid], size=11, replace=False)]
    best = None
    for start in starts:
        medoids = [start]
        while len(medoids) < k:
            nearest = distance[:, medoids].min(axis=1)
            nearest[medoids] = -1
            medoids.append(int(np.argmax(nearest)))
        for _ in range(100):
            labels = np.argmin(distance[:, medoids], axis=1)
            updated = []
            for cluster in range(k):
                members = np.flatnonzero(labels == cluster)
                if not len(members):
                    nearest = distance[:, medoids].min(axis=1)
                    nearest[medoids + updated] = -1
                    updated.append(int(np.argmax(nearest)))
                else:
                    within = distance[np.ix_(members, members)]
                    updated.append(int(members[np.argmin(within.sum(axis=1))]))
            if updated == medoids:
                break
            medoids = updated
        labels = np.argmin(distance[:, medoids], axis=1)
        cost = float(distance[np.arange(n), np.array(medoids)[labels]].sum())
        if best is None or cost < best[2] - 1e-10:
            best = (np.array(medoids), labels, cost)
    assert best is not None
    return best


def silhouette(distance: np.ndarray, labels: np.ndarray) -> float:
    scores = []
    for i in range(len(distance)):
        own = np.flatnonzero(labels == labels[i])
        if len(own) < 2:
            scores.append(0.0)
            continue
        a = distance[i, own[own != i]].mean()
        b = min(distance[i, labels == other].mean() for other in np.unique(labels)
                if other != labels[i])
        scores.append(float((b - a) / max(a, b)))
    return float(np.mean(scores))


def choose_clusters(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, list[dict]]:
    candidates = []
    for k in range(2, 9):
        medoids, labels, cost = k_medoids(distance, k)
        sizes = np.bincount(labels, minlength=k)
        score = silhouette(distance, labels)
        candidates.append({"k": k, "silhouette": score, "sizes": sizes.tolist(),
                           "cost": cost, "valid": bool(sizes.min() >= 3 and sizes.max() >= 10),
                           "medoids": medoids, "labels": labels})
    valid = [candidate for candidate in candidates if candidate["valid"]]
    if not valid:
        raise ValueError("No stable k-medoids partition with groups of at least three episodes")
    chosen = max(valid, key=lambda candidate: (candidate["silhouette"], -candidate["k"]))
    report = [{key: candidate[key] for key in ("k", "silhouette", "sizes", "cost", "valid")}
              for candidate in candidates]
    return chosen["medoids"], chosen["labels"], chosen["k"], report


def farthest_point(distance: np.ndarray, count: int) -> tuple[list[int], list[float]]:
    selected = [int(np.argmin(distance.sum(axis=1)))]
    gains = [0.0]
    while len(selected) < count:
        nearest = distance[:, selected].min(axis=1)
        nearest[selected] = -1
        chosen = int(np.argmax(nearest))
        gains.append(float(nearest[chosen]))
        selected.append(chosen)
    return selected, gains


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def preview_page(name: str, title: str, rows: list[dict], fps: int) -> None:
    cards = []
    for row in rows:
        episode_id = html.escape(str(row["episode_id"]))
        video = f"../synthetic/synthetic_episode_{episode_id}.mp4"
        if not (OUT / video).is_file():
            raise FileNotFoundError(video)
        duration = int(row["length"]) / fps
        cards.append(f"""
        <article><div class="head"><span class="rank">#{row['rank']}</span><strong>Episode {episode_id}</strong>
          <span class="meta">Cluster {row['cluster_id']} · {duration:.1f} s</span></div>
          <video controls preload="none" playsinline src="{video}" aria-label="Episode {episode_id}"></video></article>""")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · OpenDrawer</title><style>
:root {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #203029; background: #f3f5f0; }}
* {{ box-sizing: border-box; }} body {{ margin: 0; }} header {{ background: #f3f5f0; border-bottom: 1px solid #dce2d8; }}
.wrap {{ max-width: 1250px; margin: auto; padding: 20px 24px; }} h1 {{ margin: 0 0 5px; font-size: 1.7rem; }} p {{ margin: 0; color: #5b6b60; }}
a {{ color: #245b41; }} .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 360px), 1fr)); gap: 16px; margin-top: 20px; }}
article {{ background: white; border: 1px solid #dce2d8; border-radius: 12px; overflow: hidden; }} .head {{ display: flex; align-items: center; gap: 9px; padding: 12px 14px; }}
.rank {{ background: #dcebe2; padding: 3px 7px; border-radius: 5px; font-weight: 700; }} .meta {{ margin-left: auto; font-size: .82rem; color: #617167; }}
video {{ display: block; width: 100%; aspect-ratio: 1/1; max-height: 360px; background: #151b17; object-fit: contain; }}
</style></head><body><header><div class="wrap"><h1>OpenDrawer · {html.escape(title)}</h1><p>{len(rows)} selected synthetic trajectories · left agent view · <a href="redundant_10.html">Redundant 10</a> · <a href="diverse_10.html">Diverse 10</a> · <a href="cluster_representatives.html">Cluster representatives</a></p></div></header>
<main class="wrap"><section class="grid" aria-label="Selected episodes">{''.join(cards)}</section></main></body></html>"""
    (OUT / f"{name}.html").write_text(page)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    snapshots = [path for path in CACHE.iterdir() if path.is_dir()]
    if len(snapshots) != 1:
        raise ValueError("Expected one cached DreamGen snapshot")
    snapshot = snapshots[0]
    info = json.loads((snapshot / "meta/info.json").read_text())
    episodes = pq.read_table(snapshot / "meta/episodes/chunk-000/file-000.parquet",
                             columns=["episode_index", "tasks", "length", "dataset_from_index", "dataset_to_index"]).to_pylist()
    target = sorted((row for row in episodes if row["tasks"] == [INSTRUCTION]),
                    key=lambda row: row["episode_index"])
    if len(target) != 60:
        raise ValueError(f"Expected 60 OpenDrawer synthetic episodes, found {len(target)}")
    preview_rows = {int(row["episode_id"]): row for row in csv.DictReader((TASK_DIR / "episodes.csv").open(newline=""))
                    if row["source"] == "synthetic"}
    if set(preview_rows) != {row["episode_index"] for row in target}:
        raise ValueError("Preview videos do not cover exactly the target episodes")

    table = pq.read_table(snapshot / "data/chunk-000/file-000.parquet",
                          columns=["observation.state", "action"])
    state = table["observation.state"].combine_chunks().values.to_numpy().reshape(len(table), 16)
    action = table["action"].combine_chunks().values.to_numpy().reshape(len(table), 12)
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("Non-finite state or action values")
    state_sequences, action_sequences = [], []
    for row in target:
        start, end = row["dataset_from_index"], row["dataset_to_index"]
        if end - start != row["length"]:
            raise ValueError(f"Length mismatch for episode {row['episode_index']}")
        state_sequences.append(align(state_features(state[start:end])))
        action_sequences.append(align(action[start:end, :7]))
    states = np.stack(state_sequences)
    actions = np.stack(action_sequences)
    relative_states = states - states[:, :1, :]
    state_names = ([f"eef_pos_{i}" for i in range(3)] +
                   [f"eef_quat_outer_{i}" for i in range(10)] +
                   [f"gripper_qpos_{i}" for i in range(2)] +
                   [f"base_pos_{i}" for i in range(3)] +
                   [f"base_quat_outer_{i}" for i in range(10)])
    blocks, channel_report = {}, {}
    for name, values, names in (("absolute_state", states, state_names),
                                ("relative_state", relative_states, state_names),
                                ("action", actions, [f"action_{i}" for i in range(7)])):
        blocks[name], retained, scales = normalize_block(values, names)
        channel_report[name] = {"channels": retained, "pooled_std": scales}
    distance = pairwise_distance(blocks)
    ids = np.array([row["episode_index"] for row in target])
    medoids, labels, k, candidates = choose_clusters(distance)
    # Stable cluster IDs follow representative episode IDs.
    cluster_order = np.argsort(ids[medoids])
    label_map = {int(old): int(new + 1) for new, old in enumerate(cluster_order)}
    cluster_ids = np.array([label_map[int(label)] for label in labels])
    medoid_by_cluster = {label_map[old]: int(medoids[old]) for old in range(k)}
    cluster_rows = []
    for index, row in enumerate(target):
        cluster_id = int(cluster_ids[index])
        medoid = medoid_by_cluster[cluster_id]
        cluster_rows.append({"episode_id": int(ids[index]), "cluster_id": cluster_id,
                             "cluster_representative_episode_id": int(ids[medoid]),
                             "distance_to_representative": float(distance[index, medoid]),
                             "length": row["length"], "video_path": preview_rows[int(ids[index])]["video_path"]})
    write_csv(OUT / "clusters.csv", cluster_rows)

    representatives = []
    eligible = []
    for cluster_id in sorted(medoid_by_cluster):
        members = np.flatnonzero(cluster_ids == cluster_id)
        intra = distance[np.ix_(members, members)]
        mean_pairwise = float(intra[np.triu_indices(len(members), 1)].mean()) if len(members) > 1 else 0.0
        medoid = medoid_by_cluster[cluster_id]
        core = sorted(members, key=lambda i: (distance[i, medoid], ids[i]))[:10]
        core_distances = distance[np.ix_(core, core)]
        core_mean_pairwise = (float(core_distances[np.triu_indices(10, 1)].mean())
                              if len(core) == 10 else None)
        representatives.append({"cluster_id": cluster_id, "size": len(members),
                                "representative_episode_id": int(ids[medoid]),
                                "mean_pairwise_distance": mean_pairwise,
                                "core_10_mean_pairwise_distance": core_mean_pairwise,
                                "mean_distance_to_representative": float(distance[members, medoid].mean()),
                                "video_path": preview_rows[int(ids[medoid])]["video_path"]})
        if len(members) >= 10:
            eligible.append((core_mean_pairwise, -len(members), cluster_id))
    write_csv(OUT / "cluster_representatives.csv", representatives)
    dense_cluster = min(eligible)[2]
    members = np.flatnonzero(cluster_ids == dense_cluster)
    dense_medoid = medoid_by_cluster[dense_cluster]
    redundant = sorted(members, key=lambda i: (distance[i, dense_medoid], ids[i]))[:10]
    diverse, gains = farthest_point(distance, 10)
    redundant_rows = [{"rank": rank, "episode_id": int(ids[index]), "cluster_id": dense_cluster,
                       "distance_to_cluster_representative": float(distance[index, dense_medoid]),
                       "length": int(target[index]["length"]),
                       "video_path": preview_rows[int(ids[index])]["video_path"]}
                      for rank, index in enumerate(redundant, 1)]
    diverse_rows = [{"rank": rank, "episode_id": int(ids[index]), "cluster_id": int(cluster_ids[index]),
                     "distance_to_nearest_previously_selected": gain,
                     "length": int(target[index]["length"]),
                     "video_path": preview_rows[int(ids[index])]["video_path"]}
                    for rank, (index, gain) in enumerate(zip(diverse, gains), 1)]
    write_csv(OUT / "redundant_10.csv", redundant_rows)
    write_csv(OUT / "diverse_10.csv", diverse_rows)
    with (OUT / "pairwise_distances.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode_id"] + ids.tolist())
        for episode_id, distances in zip(ids, distance):
            writer.writerow([int(episode_id)] + [f"{value:.8f}" for value in distances])
    preview_page("redundant_10", "Redundant 10", redundant_rows, info["fps"])
    preview_page("diverse_10", "Diverse 10", diverse_rows, info["fps"])
    representative_rows = [{"rank": row["cluster_id"],
                            "episode_id": row["representative_episode_id"],
                            "cluster_id": row["cluster_id"],
                            "length": int(target[medoid_by_cluster[row["cluster_id"]]]["length"])}
                           for row in representatives]
    preview_page("cluster_representatives", "Cluster representatives", representative_rows, info["fps"])

    tri = np.triu_indices(len(ids), k=1)
    selected_overlap = sorted(set(ids[redundant]) & set(ids[diverse]))
    summary = {
        "task": "OpenDrawer", "synthetic_episode_count": len(ids), "source_revision": snapshot.name,
        "instruction": INSTRUCTION, "time_alignment": f"Linear interpolation to {STEPS} normalized-progress samples",
        "normalization": "Per-channel z-score over all 60 aligned trajectories; channels with pooled standard deviation below 0.01 dropped",
        "state_representation": "End-effector position, sign-invariant quaternion outer products, gripper position, robot base position and sign-invariant base quaternion outer products",
        "action_representation": "First seven action channels; last five near-constant channels omitted",
        "distance": "Square root of weighted mean squared differences across aligned normalized sequences; dimensionless",
        "distance_block_weights": WEIGHTS, "retained_channels": channel_report,
        "clustering": "Deterministic alternating k-medoids, 12 starts per k, k=2..8; highest silhouette among partitions with every cluster >=3 and at least one cluster >=10",
        "cluster_candidates": candidates, "chosen_k": k, "cluster_representatives": representatives,
        "densest_eligible_cluster": dense_cluster,
        "densest_cluster_rule": "Smallest mean pairwise distance in the 10-member medoid-nearest core among clusters with at least 10 episodes",
        "redundant_rule": "Ten closest episodes to the medoid of the densest eligible cluster",
        "diverse_rule": "Farthest-point k-center greedy selection over all 60 episodes, seeded at the global medoid",
        "selected_overlap_episode_ids": [int(x) for x in selected_overlap],
        "all_pairwise_distance_mean": float(distance[tri].mean()),
        "redundant_pairwise_distance_mean": float(distance[np.ix_(redundant, redundant)][np.triu_indices(10, 1)].mean()),
        "diverse_pairwise_distance_mean": float(distance[np.ix_(diverse, diverse)][np.triu_indices(10, 1)].mean()),
        "diverse_coverage_radius": float(distance[:, diverse].min(axis=1).max()),
        "note": "These are geometric sequence selections, not validity, utility, success, or semantic diversity labels. No policy training or FPS resampling was performed.",
    }
    (OUT / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("chosen_k", "densest_eligible_cluster",
                                                  "selected_overlap_episode_ids", "all_pairwise_distance_mean",
                                                  "redundant_pairwise_distance_mean", "diverse_pairwise_distance_mean",
                                                  "diverse_coverage_radius")}, indent=2))
    print("Cluster representatives:", [(row["cluster_id"], row["size"], row["representative_episode_id"])
                                       for row in representatives])
    print("Redundant-10:", [row["episode_id"] for row in redundant_rows])
    print("Diverse-10:", [row["episode_id"] for row in diverse_rows])


if __name__ == "__main__":
    main()
