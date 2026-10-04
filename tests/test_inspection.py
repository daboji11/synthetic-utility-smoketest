import unittest
from types import SimpleNamespace

from scripts.inspect_datasets import compare_schemas
from src.datasets import RoboCasaLeRobotDataset


class InspectionTests(unittest.TestCase):
    def test_camera_key_resolves_actual_feature_name(self):
        dataset = object.__new__(RoboCasaLeRobotDataset)
        dataset.camera_keys = ["observation.images.robot0_agentview_left_image"]
        self.assertEqual(dataset._resolve_camera_key("robot0_agentview_left_image"),
                         "observation.images.robot0_agentview_left_image")
        with self.assertRaises(ValueError):
            dataset._resolve_camera_key("missing_camera")

    def test_compatibility_flags_fps_mismatch(self):
        camera = "observation.images.robot0_agentview_left_image"
        def example(fps):
            return {"action_dim": 12, "state_dim": 16, "action_dtype": "float32",
                    "state_dtype": "float32", "action_feature_names": ["a"],
                    "state_feature_names": ["s"], "fps": fps,
                    "language_or_task": ["pick the water bottle from the counter and place it in the cabinet"],
                    "canonical_task_id": "PnPCounterToCab"}
        def dataset():
            return SimpleNamespace(camera_key=camera, camera_keys=[camera],
                                   features={camera: {"shape": [128, 128, 3]}},
                                   get_task_ids=lambda: ["shared task"])
        comparison = compare_schemas(dataset(), dataset(), example(20), example(16))
        self.assertFalse(comparison["checks"]["fps_match"])
        self.assertEqual(comparison["common_tasks"], ["shared task"])
        self.assertIn("resampling", comparison["verdict"])

    def test_task_class_pattern_excludes_reverse_direction(self):
        dataset = object.__new__(RoboCasaLeRobotDataset)
        import re
        dataset.task_pattern = re.compile(r"^pick the .+ from the counter and place it in the cabinet$")
        dataset.canonical_task_id = "PnPCounterToCab"
        dataset.task_aliases = {}
        dataset._episodes = [
            {"episode_index": 1, "tasks": ["pick the apple from the counter and place it in the cabinet"]},
            {"episode_index": 2, "tasks": ["pick the apple from the cabinet and place it on the counter"]},
        ]
        selected = dataset.filter_by_task("PnPCounterToCab")
        self.assertEqual(selected.get_episode_ids(), ["1"])
        self.assertEqual(selected.get_task_ids(), ["PnPCounterToCab"])
