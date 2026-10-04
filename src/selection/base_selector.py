from abc import ABC, abstractmethod

from src.datasets import BaseRobotDataset


class BaseSelector(ABC):
    @abstractmethod
    def select(self, dataset: BaseRobotDataset, budget: int,
               context: dict | None = None) -> list[str]:
        """Return unique trajectory IDs from the input dataset."""
