from .base_selector import BaseSelector


class DiverseSelector(BaseSelector):
    """Future valid trajectories chosen for coverage of the task distribution."""

    def select(self, dataset, budget, context=None):
        # TODO: Add a representation and coverage objective after choosing a dataset.
        raise NotImplementedError("Diverse selection is not implemented")
