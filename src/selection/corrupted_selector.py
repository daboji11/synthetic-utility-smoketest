from .base_selector import BaseSelector


class CorruptedSelector(BaseSelector):
    """Future negative control with explicit, documented trajectory corruption."""

    def select(self, dataset, budget, context=None):
        # TODO: Generate controlled corruptions; selecting existing IDs alone is insufficient.
        raise NotImplementedError("Corrupted selection is not implemented")
