#!/usr/bin/env python3
"""Select 8-episode trajectory groups from the SAME success-screened DreamGen pool.

This is a geometric trajectory distance, not a KL divergence. Draft output is
for inspection only. Final selection requires human-reviewed outcomes/motion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "records/opendrawer_dreamgen_motion_annotations.csv"
DISTANCES = ROOT / "results/dataset_inspection/OpenDrawer/trajectory_similarity/pairwise_distances.csv"
SUMMARY = ROOT / "results/dataset_inspection/OpenDrawer/trajectory_similarity/selection_summary.json"
PREVIEWS = ROOT / "results/dataset_inspection/OpenDrawer/synthetic_3cam"
OUTPUT = ROOT / "records/opendrawer_synthetic_8group_selection"


def yes(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def load_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = {int(row["episode_index"]): row for row in csv.DictReader(handle)}
    if len(rows) != 60 or set(rows) != set(range(420, 480)):
        raise ValueError("Expected exactly the 60 pinned OpenDrawer synthetic episodes")
    return rows


def effective_outcome(row: dict[str, str]) -> str:
    if yes(row["manual_outcome_reviewed"]):
        if row["manual_visual_label"] not in {"success", "failure", "uncertain"}:
            raise ValueError(f"Invalid manual outcome for {row['episode_index']}")
        return row["manual_visual_label"]
    return row["visual_label"]


def load_distances(path: Path) -> dict[int, dict[int, float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        table = {}
        for row in csv.DictReader(handle):
            episode = int(row["episode_id"])
            table[episode] = {int(k): float(v) for k, v in row.items() if k != "episode_id"}
    if set(table) != set(range(420, 480)):
        raise ValueError("Unexpected trajectory distance IDs")
    for a in table:
        if set(table[a]) != set(table) or abs(table[a][a]) > 1e-6:
            raise ValueError(f"Invalid trajectory distance row {a}")
        for b, distance in table[a].items():
            if distance < 0 or abs(distance - table[b][a]) > 1e-5:
                raise ValueError(f"Asymmetric trajectory distance: {a}, {b}")
    return table


def mean_distance(episodes: tuple[int, ...], distances: dict[int, dict[int, float]]) -> float:
    pairs = itertools.combinations(episodes, 2)
    return sum(distances[a][b] for a, b in pairs) / (len(episodes) * (len(episodes) - 1) / 2)


def extreme_subset(pool: list[int], count: int, distances: dict[int, dict[int, float]],
                   *, largest: bool) -> tuple[tuple[int, ...], float]:
    selected = None
    selected_score = float("-inf") if largest else float("inf")
    for candidate in itertools.combinations(pool, count):
        score = mean_distance(candidate, distances)
        if (score > selected_score + 1e-12 if largest else score < selected_score - 1e-12):
            selected, selected_score = candidate, score
    if selected is None:
        raise ValueError(f"Cannot choose {count} from {len(pool)} episodes")
    return selected, selected_score


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("draft", "final"), default="draft")
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if args.count < 2:
        raise ValueError("At least two episodes needed to measure pairwise distance")

    rows = load_rows(LABELS)
    distances = load_distances(DISTANCES)
    metric = json.loads(SUMMARY.read_text())
    if metric["synthetic_episode_count"] != 60:
        raise ValueError("Trajectory metric provenance changed")
    if any(row["source_revision"] != metric["source_revision"] for row in rows.values()):
        raise ValueError("Labels and distances refer to different source revisions")

    provisional = sorted(episode for episode, row in rows.items()
                         if effective_outcome(row) == "success")
    reviewed = sorted(episode for episode, row in rows.items()
                      if yes(row["manual_outcome_reviewed"])
                      and row["manual_visual_label"] == "success"
                      and yes(row["motion_reviewed"]))
    pool = provisional if args.mode == "draft" else reviewed
    if len(pool) < 2 * args.count:
        raise ValueError(
            f"Need at least {2 * args.count} success episodes to make diverse and redundant disjoint; "
            f"mode={args.mode}, available={len(pool)}, human-reviewed success+motion={len(reviewed)}"
        )

    # First find the globally tightest group; then find a far-apart group
    # from the remaining pool. This keeps the two deliberate contrasts disjoint.
    redundant, redundant_score = extreme_subset(pool, args.count, distances, largest=False)
    remaining = [episode for episode in pool if episode not in redundant]
    diverse, diverse_score = extreme_subset(remaining, args.count, distances, largest=True)
    random_group = tuple(sorted(random.Random(args.seed).sample(pool, args.count)))
    groups = {"redundant": redundant, "diverse": diverse, "random": random_group}

    records = []
    for group, episodes in groups.items():
        for episode in episodes:
            row = rows[episode]
            preview = PREVIEWS / f"synthetic_episode_{episode}_3cam.mp4"
            if not preview.is_file():
                raise FileNotFoundError(preview)
            records.append({
                "group": group, "episode_index": episode,
                "outcome": effective_outcome(row),
                "outcome_source": "human" if yes(row["manual_outcome_reviewed"]) else "provisional_visual",
                "motion_reviewed": yes(row["motion_reviewed"]),
                "idm_action_validity": row["idm_action_validity"],
                "three_camera_preview": str(preview.relative_to(ROOT)),
                "approach_side": row["approach_side"],
                "approach_angle": row["approach_angle"],
                "contact_quality": row["contact_quality"],
                "trajectory_group": row["trajectory_group"],
            })
    report = {
        "selection_status": "DRAFT_NOT_FOR_TRAINING" if args.mode == "draft" else "HUMAN_REVIEWED_SELECTION_ONLY",
        "training_dataset_ready": False,
        "training_dataset_note": "Source is 16 Hz, while prior human training/evaluation used 20 Hz; "
                                 "IDM action semantics and temporal conversion require separate preflight.",
        "source_repo": rows[420]["source_repo"],
        "source_revision": metric["source_revision"],
        "source_fps": 16,
        "source_frames_per_episode": 279,
        "method": "pairwise weighted state/action trajectory distance after 64-point progress alignment; not KL",
        "distance_source": str(DISTANCES.relative_to(ROOT)),
        "distance_sha256": checksum(DISTANCES),
        "labels_source": str(LABELS.relative_to(ROOT)),
        "labels_sha256": checksum(LABELS),
        "eligibility": "draft: provisional visual success with human override; "
                       "final: human outcome success and motion reviewed",
        "eligible_episodes": pool,
        "provisional_success_count": len(provisional),
        "human_reviewed_success_and_motion_count": len(reviewed),
        "group_count": args.count,
        "random_seed": args.seed,
        "group_order": ["redundant", "diverse", "random"],
        "groups": {
            group: {
                "episode_indices": list(episodes),
                "mean_pairwise_distance": mean_distance(episodes, distances),
                "overlap_with_other_groups": {
                    other: sorted(set(episodes) & set(other_episodes))
                    for other, other_episodes in groups.items() if other != group
                },
            } for group, episodes in groups.items()
        },
        "dense_group_global_minimum": redundant_score,
        "diverse_group_maximum_among_non_redundant_pool": diverse_score,
        "records": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT.name + ("_draft" if args.mode == "draft" else "_final_" + checksum(LABELS)[:8])
    out_json = OUTPUT.with_name(stem + ".json")
    out_csv = OUTPUT.with_name(stem + ".csv")
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.mode == "final" and out_json.exists() and out_json.read_text() != payload:
        raise FileExistsError(f"Preserving existing final selection: {out_json}")
    out_json.write_text(payload, encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({
        "mode": args.mode, "eligible": len(pool), "human_reviewed_success_and_motion": len(reviewed),
        "groups": {group: info["episode_indices"] for group, info in report["groups"].items()},
        "mean_distances": {group: round(info["mean_pairwise_distance"], 5)
                           for group, info in report["groups"].items()},
        "output_json": str(out_json), "output_csv": str(out_csv),
        "training_dataset_ready": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
