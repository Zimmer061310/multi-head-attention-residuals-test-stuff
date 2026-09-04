import json
import unittest
from pathlib import Path

from src.experiments.experiment9_head_contribution import condition_manifest, load_spec


class Experiment9ProtocolTest(unittest.TestCase):
    def test_protocol_freezes_checkpoint_artifact_and_no_training(self):
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

    def test_gate_and_alignment_sample_are_frozen(self):
        spec = load_spec()
        gate = spec["phase_9a"]["local_contribution_gate"]
        self.assertEqual(gate["practical_margin_nll"], 0.001)
        self.assertEqual(spec["phase_9b"]["derangement_count"], 32)
        self.assertEqual(spec["phase_9b"]["derangement_seed"], 20260910)
        self.assertEqual(
            spec["phase_9b"]["alignment_evidence_rule"]["minimum_positive_fraction"],
            0.75,
        )

    def test_manifest_is_deterministic_and_self_hashed(self):
        first = condition_manifest()
        second = condition_manifest()
        self.assertEqual(first, second)
        self.assertEqual(len(first["phase_9a"]), 73)
        self.assertEqual(len(first["phase_9b"]), 32)
        self.assertEqual(len(first["content_sha256"]), 64)

    def test_runbook_uses_two_gpus_and_fail_closed_controller(self):
        root = Path(__file__).resolve().parents[1]
        runbook = (root / "docs/runbooks/experiment9-two-gpu.md").read_text()
        controller = (root / "scripts/evaluate/run_experiment9_controller.py").read_text()
        worker = (root / "scripts/evaluate/run_experiment9_worker.sh").read_text()
        self.assertIn("performs no training", runbook)
        self.assertIn("GPUs 0 and 1", runbook)
        self.assertIn("FAILED.json", controller)
        self.assertIn("refusing shutdown while GPU processes remain", controller)
        self.assertNotIn("train_scratch", worker)
        self.assertNotIn("optimizer", worker)


if __name__ == "__main__":
    unittest.main()
