"""CPU contract checks: real normalization semantics, freeze range, probe gate."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pi05_base_faucet_v2_config as v2


class FaucetV2Tests(unittest.TestCase):
    def stats(self):
        return {"observation.state": {"q01": list(range(16)), "q99": list(range(2, 18))},
                "action": {"mean": [0.1] * 12, "std": [0.2] * 12}}

    def test_state_order_endpoints_and_action_preservation(self):
        stats = self.stats()
        payload = v2.build_norm_payload(stats)["norm_stats"]
        state = payload["state"]
        low = np.array(stats["observation.state"]["q01"])[list(v2.STATE_ORDER)]
        high = np.array(stats["observation.state"]["q99"])[list(v2.STATE_ORDER)]
        np.testing.assert_allclose((low - state["mean"][:16]) / state["std"][:16], -1)
        np.testing.assert_allclose((high - state["mean"][:16]) / state["std"][:16], 1)
        self.assertEqual(payload["actions"]["mean"][:12], stats["action"]["mean"])
        self.assertEqual(payload["actions"]["std"][:12], stats["action"]["std"])
        self.assertEqual(state["mean"][16:], [0.0] * 16)
        self.assertEqual(payload["actions"]["std"][12:], [1.0] * 20)

    def test_constant_state_is_finite_and_bad_quantiles_rejected(self):
        stats = self.stats()
        stats["observation.state"]["q99"][0] = 0
        payload = v2.build_norm_payload(stats)["norm_stats"]
        self.assertEqual(payload["state"]["std"][0], 1)
        stats["observation.state"]["q99"][0] = -1
        with self.assertRaises(ValueError):
            v2.build_norm_payload(stats)

    def test_probe_gate_rejects_false_success_and_mismatched_stats(self):
        report = {"passed": True, "norm_fingerprint": "abc", "base_params_uri": "official",
                  "source_commit": v2.SOURCE_COMMIT, "probe_steps": 300, "batch_size": 8,
                  "selected_episode_indices": [0, 1],
                  "before": {"normalized_first5_arm_mse": 1., "noise_seeds": [101, 102, 103]},
                  "after": {"normalized_first5_arm_mse": .8, "noise_seeds": [101, 102, 103]}}
        with mock.patch.object(v2, "official_base_params_uri", return_value="official"):
            v2.require_probe(report, "abc")
            with self.assertRaises(ValueError):
                v2.require_probe(report, "different")
            report["after"]["normalized_first5_arm_mse"] = .95
            with self.assertRaises(ValueError):
                v2.require_probe(report, "abc")
            report["passed"] = False
            v2.require_probe(report, "abc", allow_no_gain=True)
            with self.assertRaises(ValueError):
                v2.require_probe(report, "different", allow_no_gain=True)
            report["after"]["normalized_first5_arm_mse"] = float("nan")
            with self.assertRaises(ValueError):
                v2.require_probe(report, "abc", allow_no_gain=True)

    def test_training_requires_manual_train_flag(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/finetune_pi05_base_faucet_v2.py")],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--train is required", result.stderr)

    def test_run_id_cannot_escape_output_directory(self):
        for name in ("../old", "/tmp/run", "a/b", "a b"):
            with self.assertRaises(ValueError):
                v2.run_root(name)

    def test_launcher_stops_on_execution_failure_and_orders_successful_stages(self):
        import tempfile
        import shutil
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "scripts").mkdir()
            launcher = folder / "scripts/run_pi05_base_faucet_v2_5090.sh"
            shutil.copyfile(ROOT / "scripts" / launcher.name, launcher)
            fake = folder / "openpi/.venv/bin/python"
            fake.parent.mkdir(parents=True)
            fake.write_text('#!/usr/bin/env bash\n'
                'printf "%s\\n" "$*" >> "$FAKE_CALLS"\n'
                'if [[ "$FAKE_FAIL" == 1 && "$*" == *--probe* ]]; then exit 17; fi\n')
            fake.chmod(0o755)
            calls = folder / "calls.txt"
            env = {**os.environ, "OPENPI_ROOT": str(folder / "openpi"),
                   "FAKE_CALLS": str(calls), "FAKE_FAIL": "1"}
            result = subprocess.run(["bash", str(launcher), "all"], env=env,
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 17, result.stderr)
            attempted = calls.read_text().splitlines()
            self.assertEqual(len(attempted), 2)
            self.assertIn("--probe", attempted[-1])
            self.assertIn("EXIT=17", (folder / "logs/pi05_base_faucet_v2" /
                v2.DEFAULT_RUN / "status.txt").read_text())
            calls.write_text("")
            result = subprocess.run(["bash", str(launcher), "all"],
                env={**env, "FAKE_FAIL": "0"}, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            attempted = calls.read_text().splitlines()
            self.assertEqual(len(attempted), 4)
            self.assertIn("config.py", attempted[0])
            self.assertIn("--probe", attempted[1])
            self.assertNotIn("--probe", attempted[2])
            self.assertIn("evaluate_pi05_base_faucet_v2.py", attempted[3])
            self.assertIn("/15000", attempted[3])

    @unittest.skipUnless(importlib.util.find_spec("flax"), "Pinned OpenPI env required")
    def test_real_transform_roundtrip_schedule_and_freeze(self):
        from flax import nnx
        from openpi import transforms
        from openpi.shared import normalize
        import pi05_base_faucet_config as old
        import tempfile
        payload = v2.build_norm_payload(self.stats())
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / "assets").mkdir()
            (base / "assets/norm_stats.json").write_text(json.dumps(payload))
            with mock.patch.object(old, "verify_base_checkpoint"), mock.patch.object(old, "BASE", base), \
                    mock.patch.object(old, "official_base_params_uri", return_value="gs://official/pi05_base/params"), \
                    mock.patch.object(v2, "norm_payload", return_value=payload):
                cfg = v2.make_config()
                data = cfg.data.create(cfg.assets_dirs, cfg.model)
                self.assertFalse(data.use_quantile_norm)
                self.assertEqual(cfg.weight_loader.params_path, str(base / "params"))
                self.assertEqual(cfg.batch_size, 8)
                self.assertEqual(cfg.num_train_steps, 15000)
                self.assertEqual(v2.SAVE_STEPS, (1000, 3000, 5000, 7000, 10000, 12000, 15000))
                schedule = cfg.lr_schedule.create()
                self.assertAlmostEqual(float(schedule(200)), 1e-5, places=10)
                self.assertGreater(float(schedule(5000)), 1e-6)
                self.assertGreater(float(schedule(7000)), 5e-6)
                self.assertGreater(float(schedule(12000)), 1e-6)
                self.assertAlmostEqual(float(schedule(15000)), 1e-6, places=10)
                param = nnx.Param(np.zeros(1))
                self.assertFalse(cfg.trainable_filter(("PaliGemma", "img", "kernel"), param))
                self.assertFalse(cfg.trainable_filter(("action_out_proj", "kernel"), param))
                self.assertTrue(cfg.trainable_filter(("PaliGemma", "llm", "lora_a"), param))
                self.assertFalse(cfg.freeze_filter(("rng", "count"), nnx.Variable(np.zeros(1))))
                norm = data.norm_stats
                sample = {"state": np.arange(32, dtype=float), "actions": np.full((50, 32), .3)}
                encoded = transforms.Normalize(norm, use_quantiles=False)(sample)
                decoded = transforms.Unnormalize(norm, use_quantiles=False)(encoded)
                np.testing.assert_allclose(decoded["actions"], sample["actions"], atol=1e-6)
                normalize.save(base / "saved/assets", norm)
                with mock.patch.object(v2, "norm_payload", return_value=payload):
                    v2.verify_saved_norm(base / "saved")


if __name__ == "__main__":
    unittest.main()
