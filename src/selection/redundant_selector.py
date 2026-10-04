from .base_selector import BaseSelector


class RedundantSelector(BaseSelector):
    """Future valid trajectories close to the base real-data distribution."""

    def select(self, dataset, budget, context=None):
        # TODO: Define a task-specific trajectory representation and redundancy rule.
        raise NotImplementedError("Redundant selection is not implemented")
