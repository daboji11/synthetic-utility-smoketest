"""Minimal trajectory container protocol independent of a robot benchmark."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypedDict


class Trajectory(TypedDict):
    trajectory_id: str
    observations: Any
    actions: Any
    task_id: str
    success: bool | None
    metadata: dict[str, Any]


class BaseRobotDataset(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> Trajectory: ...

    @abstractmethod
    def get_trajectory(self, trajectory_id: str) -> Trajectory: ...

    @abstractmethod
    def get_task_ids(self) -> list[str]: ...

    @abstractmethod
    def filter_by_task(self, task_id: str) -> BaseRobotDataset: ...
