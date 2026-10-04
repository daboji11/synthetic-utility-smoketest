"""Transition-level toy BC training with a fixed, seeded DataLoader."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.datasets import BaseRobotDataset
from src.policies.base_policy import BasePolicy


class Trainer:
    def __init__(self, policy: BasePolicy, *, batch_size: int, epochs: int, seed: int):
        if batch_size < 1 or epochs < 1:
            raise ValueError("batch_size and epochs must be positive")
        self.policy, self.batch_size, self.epochs, self.seed = policy, batch_size, epochs, seed

    def train(self, dataset: BaseRobotDataset, checkpoint: Path, log: Path) -> dict:
        if len(dataset) == 0:
            raise ValueError("Cannot train on an empty dataset")
        observations, actions = [], []
        for index in range(len(dataset)):
            trajectory = dataset[index]
            obs = np.asarray(trajectory["observations"], dtype=np.float32)
            act = np.asarray(trajectory["actions"], dtype=np.float32)
            if obs.ndim != 2 or act.ndim != 2 or len(obs) != len(act) or not len(obs):
                raise ValueError(f"Invalid observation/action sequence: {trajectory['trajectory_id']}")
            observations.append(obs)
            actions.append(act)
        samples = TensorDataset(torch.tensor(np.concatenate(observations)), torch.tensor(np.concatenate(actions)))
        loader = DataLoader(samples, batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(self.seed))
        log.parent.mkdir(parents=True, exist_ok=True)
        losses = []
        with log.open("w") as handle:
            for epoch in range(self.epochs):
                epoch_losses = [self.policy.train_step(obs, act) for obs, act in loader]
                mean_loss = float(np.mean(epoch_losses))
                losses.append(mean_loss)
                handle.write(f"epoch={epoch + 1} loss={mean_loss:.8f}\n")
        self.policy.save(checkpoint)
        return {"training_loss": losses[-1], "training_loss_by_epoch": losses,
                "optimization_steps": self.epochs * len(loader), "training_transitions": len(samples)}
