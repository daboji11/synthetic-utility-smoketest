from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BasePolicy(ABC):
    @abstractmethod
    def train_step(self, observations: Any, actions: Any) -> float: ...

    @abstractmethod
    def predict(self, observations: Any) -> Any: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> None: ...
