"""Dataset proxy today; simulator rollout adapter later."""
import numpy as np

from src.datasets import BaseRobotDataset
from src.policies.base_policy import BasePolicy
from .metrics import aggregate_rollouts


class Evaluator:
    def __init__(self, num_episodes: int, toy_success_mse_threshold: float = 0.05):
        if num_episodes < 1:
            raise ValueError("num_episodes must be positive")
        self.num_episodes = num_episodes
        self.toy_success_mse_threshold = toy_success_mse_threshold

    def evaluate(self, policy: BasePolicy, env_or_dataset) -> dict:
        episodes = []
        if isinstance(env_or_dataset, BaseRobotDataset):
            if len(env_or_dataset) < self.num_episodes:
                raise ValueError("Evaluation dataset is smaller than num_episodes")
            for index in range(self.num_episodes):
                trajectory = env_or_dataset[index]
                target = np.asarray(trajectory["actions"], dtype=np.float32)
                prediction = policy.predict(np.asarray(trajectory["observations"], dtype=np.float32))
                mse = float(np.mean((prediction - target) ** 2))
                episodes.append({"success": mse <= self.toy_success_mse_threshold, "return": -mse})
            metric_type = "toy_action_mse_proxy"
        else:
            # TODO: Supply a simulator adapter whose run_episode executes a closed-loop rollout.
            for _ in range(self.num_episodes):
                episodes.append(env_or_dataset.run_episode(policy))
            metric_type = "environment_rollout"
        return {**aggregate_rollouts(episodes), "evaluation_type": metric_type}
