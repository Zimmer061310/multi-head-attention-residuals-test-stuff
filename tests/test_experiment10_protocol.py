import unittest
from pathlib import Path

from src.experiments.experiment10_per_group_contribution import condition_manifest, load_spec


class Experiment10ProtocolTest(unittest.TestCase):
    def test_protocol_freezes_inputs_and_no_training(self):
        spec = load_spec()
        self.assertFalse(spec["training"])
        self.assertEqual(spec["seed"], 42)
        self.assertEqual(spec["milestone"], 2000)
        self.assertEqual(
            spec["checkpoint"]["sha256"],
            "74cff0ab19409dac9f6104e8986e4890c0837bc730d8ac233ae18011dbc58333",
        )
        self.assertEqual(
            spec["fixed_evaluation"]["sha256"],
            "29e545dd9399e9eaea6f5abf38f20ef76ba232466a01b6ddf13c3a6a287a3691",
        )

    def test_manifest_is_deterministic_and_self_hashed(self):
        first = condition_manifest()
        second = condition_manifest()
        self.assertEqual(first, second)
        self.assertEqual(len(first["phase_10abc"]), 25)
        self.assertEqual(len(first["phase_10d"]), 56)
        self.assertEqual(len(first["content_sha256"]), 64)

    def test_runbook_is_frozen_two_gpu_evaluation(self):
        root = Path(__file__).resolve().parents[1]
        runbook = (root / "docs/runbooks/experiment10-two-gpu.md").read_text()
        controller = (root / "scripts/evaluate/run_experiment10_controller.py").read_text()
        worker = (root / "scripts/evaluate/run_experiment10_worker.sh").read_text()
        self.assertIn("performs no training", runbook)
        self.assertIn("GPUs 0 and 1", runbook)
        self.assertIn("FAILED.json", controller)
        self.assertIn("refusing shutdown while GPU processes remain", controller)
        self.assertNotIn("train_scratch", worker)
        self.assertNotIn("optimizer", worker)


if __name__ == "__main__":
    unittest.main()
