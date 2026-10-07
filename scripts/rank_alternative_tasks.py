"""Count exact H50 human demonstrations for the 24 cached DreamGen task prompts.

This deliberately does not infer synthetic success from episode boundaries.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/raw/huggingface"
OUTPUT = ROOT / "records/alternative_task_h50_screen.csv"


def snapshot(repo: str) -> Path:
    found = list((CACHE / f"datasets--{repo}" / "snapshots").iterdir())
    assert len(found) == 1, found
    return found[0]


def episodes(repo: str) -> list[dict]:
    path = snapshot(repo) / "meta/episodes/chunk-000/file-000.parquet"
    return pq.read_table(path, columns=["episode_index", "tasks", "length"]).to_pylist()


h50 = episodes("DAVIAN-Robotics--robocasa-H50")
synthetic = episodes("Keh0t0--dreamgen_robocasa_p3_init0_v2_droid")
assert len(h50) == 1293
assert len(synthetic) == 1440

h50_by_prompt = Counter(task for ep in h50 for task in ep["tasks"])
synthetic_by_prompt = Counter(task for ep in synthetic for task in ep["tasks"])
mapping = list(csv.DictReader((ROOT / "results/dataset_inspection/task_overlap_scan.csv").open(newline="")))
assert len(mapping) == 24

rows = []
for row in mapping:
    instruction = row["synthetic_instruction"]
    count = h50_by_prompt[instruction]
    rows.append(
        {
            "task_name": row["task_name"],
            "exact_instruction": instruction,
            "h50_human_exact_episodes": count,
            "mg30_exact_episodes": int(row["real_exact_match_count"]),
            "dreamgen_synthetic_episodes": synthetic_by_prompt[instruction],
            "dreamgen_success_count": "unknown",
            "dreamgen_success_rate": "unverified",
            "h50_status": "exact prompt matches only",
        }
    )

assert len(rows) == 24
assert all(row["dreamgen_synthetic_episodes"] == 60 for row in rows)
rows.sort(key=lambda row: (-row["h50_human_exact_episodes"], row["task_name"]))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

for row in rows:
    print(f"{row['task_name']:26s} H50={row['h50_human_exact_episodes']:3d} MG30={row['mg30_exact_episodes']:2d} synthetic={row['dreamgen_synthetic_episodes']:2d}  {row['exact_instruction']}")
print(f"SAVED {OUTPUT}")
