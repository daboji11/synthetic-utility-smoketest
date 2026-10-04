from __future__ import annotations

from typing import Iterable


def aggregate_rollouts(episodes: Iterable[dict]) -> dict:
    episodes = list(episodes)
    if not episodes:
        raise ValueError("No evaluation episodes")
    return {"success_rate": sum(bool(e["success"]) for e in episodes) / len(episodes),
            "mean_return": sum(float(e["return"]) for e in episodes) / len(episodes),
            "num_episodes": len(episodes)}
