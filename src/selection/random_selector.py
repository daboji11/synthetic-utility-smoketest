import random

from src.datasets import BaseRobotDataset
from .base_selector import BaseSelector


class RandomSelector(BaseSelector):
    def __init__(self, seed: int = 0):
        self.seed = seed

    def select(self, dataset: BaseRobotDataset, budget: int,
               context: dict | None = None) -> list[str]:
        if budget < 0 or budget > len(dataset):
            raise ValueError("Selection budget must be between zero and dataset size")
        return random.Random(self.seed).sample([dataset[i]["trajectory_id"] for i in range(len(dataset))], budget)
