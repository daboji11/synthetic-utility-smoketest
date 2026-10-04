from copy import deepcopy
import unittest
import uuid

from src.pipeline import ROOT, run_smoke_test
from src.utils.config import load_config


class PipelineTests(unittest.TestCase):
    def test_end_to_end_smoke_test(self):
        config = deepcopy(load_config(ROOT / "configs/experiment/smoke_test.yaml"))
        config["experiment_name"] = "test_" + uuid.uuid4().hex
        config["dataset"].update(real_size=2, synthetic_size=4, evaluation_size=2)
        config["selection"]["budget"] = 2
        config["evaluation"]["num_episodes"] = 2
        config["training"].update(batch_size=4, epochs=1)
        try:
            metrics = run_smoke_test(config)
            result = ROOT / "results" / config["experiment_name"]
            self.assertEqual(metrics["num_episodes"], 2)
            self.assertEqual(metrics["evaluation_type"], "toy_action_mse_proxy")
            self.assertTrue(0 <= metrics["success_rate"] <= 1)
            self.assertLessEqual({"config.yaml", "metrics.json", "checkpoint.pt", "train.log"}, {p.name for p in result.iterdir()})
        finally:
            import shutil
            shutil.rmtree(ROOT / "results" / config["experiment_name"], ignore_errors=True)
            shutil.rmtree(ROOT / "data/processed" / config["experiment_name"], ignore_errors=True)
