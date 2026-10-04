from .base_selector import BaseSelector


class PolicyHardSelector(BaseSelector):
    """Future valid trajectories challenging to a specified current policy."""

    def select(self, dataset, budget, context=None):
        # TODO: Define policy checkpoint and weakness signal in context.
        raise NotImplementedError("Policy-hard selection is not implemented")
