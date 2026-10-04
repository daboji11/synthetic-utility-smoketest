from .small_policy import SmallPolicy


def make_policy(config: dict, observation_dim: int, action_dim: int, learning_rate: float):
    if config["name"] != "small_policy":
        # TODO: Register new imitation policies here.
        raise ValueError(f"Unknown policy: {config['name']}")
    return SmallPolicy(observation_dim, action_dim, config["hidden_dim"], learning_rate)
