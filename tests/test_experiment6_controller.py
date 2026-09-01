"""Fail-closed checks for the Experiment 6 completion controller."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate.run_experiment6_controller import checkpoint_complete


class Experiment6ControllerTest(unittest.TestCase):
    def make_checkpoint(self, root, *, variant="c8", step=2000, commit="locked"):
        checkpoint = Path(root) / "step-2000"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(b"weights")
        (checkpoint / "training_state.pt").write_bytes(b"state")
        groups = {"b": None, "c4": 4, "g4": 4, "c8": 8, "g8": 8}[variant]
        (checkpoint / "training_manifest.json").write_text(json.dumps({
            "global_step": step,
            "run_identity": {
                "seed": 42,
                "steps": 20000,
                "global_batch_size": 32,
                "experiment6_variant": variant,
                "experiment6_qkv_groups": groups,
                "source_commit": commit,
            },
        }))
        return checkpoint

    def test_complete_checkpoint_requires_frozen_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.make_checkpoint(directory)
            self.assertTrue(checkpoint_complete(checkpoint, "c8", "locked"))
            self.assertFalse(checkpoint_complete(checkpoint, "c8", "different"))

    def test_wrong_step_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.make_checkpoint(directory, step=1900)
            self.assertFalse(checkpoint_complete(checkpoint, "c8", "locked"))

    def test_missing_training_state_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.make_checkpoint(directory)
            (checkpoint / "training_state.pt").unlink()
            self.assertFalse(checkpoint_complete(checkpoint, "c8", "locked"))


if __name__ == "__main__":
    unittest.main()
