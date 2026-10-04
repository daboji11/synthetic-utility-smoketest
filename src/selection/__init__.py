from .random_selector import RandomSelector
from .redundant_selector import RedundantSelector
from .diverse_selector import DiverseSelector
from .policy_hard_selector import PolicyHardSelector
from .corrupted_selector import CorruptedSelector


def make_selector(name: str, seed: int = 0):
    selectors = {"random": RandomSelector, "redundant": RedundantSelector,
                 "diverse": DiverseSelector, "policy_hard": PolicyHardSelector,
                 "corrupted": CorruptedSelector}
    if name not in selectors:
        raise ValueError(f"Unknown selector: {name}")
    return selectors[name](seed=seed) if name == "random" else selectors[name]()
