import unittest

from src.datasets import PlaceholderRobotDataset


class DatasetTests(unittest.TestCase):
    def test_placeholder_dataset_access_and_filter(self):
        dataset = PlaceholderRobotDataset.generate(source="toy", count=3, task_id="task_a", seed=4)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.get_trajectory(dataset[0]["trajectory_id"]), dataset[0])
        self.assertEqual(dataset.get_task_ids(), ["task_a"])
        self.assertEqual(len(dataset.filter_by_task("task_a")), 3)
        self.assertEqual(len(dataset.filter_by_task("other")), 0)
