import unittest

from src.datasets import PlaceholderRobotDataset
from src.selection import RandomSelector, make_selector


class SelectorTests(unittest.TestCase):
    def test_random_selection_budget_uniqueness_and_seed(self):
        dataset = PlaceholderRobotDataset.generate(source="toy", count=12, task_id="task", seed=1)
        selected = RandomSelector(seed=7).select(dataset, 5)
        self.assertEqual(len(selected), 5)
        self.assertEqual(len(set(selected)), 5)
        self.assertEqual(selected, RandomSelector(seed=7).select(dataset, 5))
        self.assertLessEqual(set(selected), {dataset[i]["trajectory_id"] for i in range(len(dataset))})
        with self.assertRaises(ValueError):
            RandomSelector().select(dataset, 13)

    def test_unimplemented_selector_is_explicit(self):
        dataset = PlaceholderRobotDataset.generate(source="toy", count=2, task_id="task", seed=1)
        with self.assertRaises(NotImplementedError):
            make_selector("diverse").select(dataset, 1)
