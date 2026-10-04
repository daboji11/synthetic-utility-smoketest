"""Deterministic toy sequences for checking the experiment plumbing only."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .base_dataset import BaseRobotDataset, Trajectory


class PlaceholderRobotDataset(BaseRobotDataset):
    def __init__(self, trajectories: Sequence[Trajectory]):
        self.trajectories = list(trajectories)
        self._by_id = {t["trajectory_id"]: t for t in self.trajectories}
        if len(self._by_id) != len(self.trajectories):
            raise ValueError("Trajectory IDs must be unique")

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, index: int) -> Trajectory:
        return self.trajectories[index]

    def get_trajectory(self, trajectory_id: str) -> Trajectory:
        return self._by_id[trajectory_id]

    def get_task_ids(self) -> list[str]:
        return sorted({t["task_id"] for t in self.trajectories})

    def filter_by_task(self, task_id: str) -> PlaceholderRobotDataset:
        return PlaceholderRobotDataset([t for t in self.trajectories if t["task_id"] == task_id])

    @classmethod
    def generate(cls, *, source: str, count: int, task_id: str, seed: int,
                 observation_dim: int = 4, action_dim: int = 2) -> PlaceholderRobotDataset:
        if count < 0 or observation_dim < 1 or action_dim < 1:
            raise ValueError("Invalid toy dataset dimensions or count")
        rng = np.random.default_rng(seed)
        weights = np.arange(1, observation_dim * action_dim + 1, dtype=np.float32).reshape(observation_dim, action_dim) / (observation_dim * action_dim)
        trajectories: list[Trajectory] = []
        for index in range(count):
            observations = rng.normal(size=(6, observation_dim)).astype(np.float32)
            actions = np.tanh(observations @ weights).astype(np.float32)
            trajectories.append({"trajectory_id": f"{source}_{index:05d}", "observations": observations.tolist(),
                                 "actions": actions.tolist(), "task_id": task_id, "success": True,
                                 "metadata": {"source": source}})
        return cls(trajectories)
