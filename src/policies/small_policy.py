"""Tiny state-to-action behavior cloning baseline."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .base_policy import BasePolicy


class SmallPolicy(BasePolicy):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 32,
                 learning_rate: float = 3e-4):
        self.model = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, action_dim))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train_step(self, observations: torch.Tensor, actions: torch.Tensor) -> float:
        self.model.train()
        loss = nn.functional.mse_loss(self.model(observations), actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def predict(self, observations: np.ndarray | torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(observations, dtype=torch.float32)
            return self.model(tensor).cpu().numpy()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
