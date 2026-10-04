from .base_dataset import BaseRobotDataset, Trajectory
from .placeholder_dataset import PlaceholderRobotDataset
from .robocasa_lerobot_dataset import RoboCasaLeRobotDataset


def load_dataset(name: str, *, count: int, task_id: str, seed: int,
                 observation_dim: int, action_dim: int, **kwargs) -> BaseRobotDataset:
    if name not in {"placeholder_real", "placeholder_synthetic", "placeholder_evaluation"}:
        # TODO: Register RoboCasa, MimicGen, DreamGen, etc. adapters here.
        raise ValueError(f"Unknown dataset: {name}")
    return PlaceholderRobotDataset.generate(source=name, count=count, task_id=task_id,
                                            seed=seed, observation_dim=observation_dim,
                                            action_dim=action_dim)
