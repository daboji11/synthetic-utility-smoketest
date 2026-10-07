#!/usr/bin/env python3
"""Check matched initial conditions and report pi05_base versus 28-H50 LoRA."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from pi05_base_h50_config import ROOT, SEEDS

RUN = ROOT / "results/robocasa_eval/pi05_base_h50_smoketest"


def main() -> None:
    base = json.loads((RUN / "base/summary.json").read_text())
    lora = json.loads((RUN / "lora_4000/summary.json").read_text())
    if any(report["episodes"] != 20 or report["seeds"] != list(SEEDS)
           or [row["seed"] for row in report["episode_stats"]] != list(SEEDS)
           for report in (base, lora)):
        raise ValueError("Expected exactly the same 20 seeds in order")
    if base["norm_sha256"] != lora["norm_sha256"]:
        raise ValueError("Different normalization assets")
    initial_base = json.loads((RUN / "base/initial_conditions.json").read_text())
    initial_lora = json.loads((RUN / "lora_4000/initial_conditions.json").read_text())
    if len(initial_base) != 20 or len(initial_lora) != 20:
        raise ValueError("Incomplete initial conditions")
    for first, second in zip(initial_base, initial_lora, strict=True):
        if first["seed"] != second["seed"] or first["task"] != second["task"]:
            raise ValueError("Initial seed/task mismatch")
        if not np.allclose(first["state"], second["state"], rtol=0, atol=1e-5):
            raise ValueError(f"Initial state mismatch at seed {first['seed']}")
        if first["image_sha256"] != second["image_sha256"]:
            raise ValueError(f"Initial three-camera images mismatch at seed {first['seed']}")
    rows = [{"seed": seed, "pi05_base_success": bool(a["success"]),
             "pi05_base_h50_lora_success": bool(b["success"]),
             "pi05_base_steps": a["steps"], "pi05_base_h50_lora_steps": b["steps"]}
            for seed, a, b in zip(SEEDS, base["episode_stats"], lora["episode_stats"], strict=True)]
    with (RUN / "paired_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"pi05_base": base["successes"], "pi05_base_plus_28_human": lora["successes"],
              "episodes": 20, "seeds": list(SEEDS), "initial_state_tolerance": 1e-5,
              "normalizer_sha256": base["norm_sha256"]}
    (RUN / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"pi05_base: {base['successes']}/20")
    print(f"pi05_base + 28 human demos: {lora['successes']}/20")


if __name__ == "__main__":
    main()
