"""Tests for the Stage B milestone pause controller."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.setup.pause_stage_b_at_milestone import (
    checkpoint_step,
    evaluate_state,
    select_variants,
)


class StageBMilestonePauseTest(unittest.TestCase):
    def write_checkpoint(self, root, variant, milestone=2000, manifest_step=None):
        checkpoint = Path(root) / variant / f"step-{milestone}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "training_manifest.json").write_text(
            json.dumps(
                {"global_step": milestone if manifest_step is None else manifest_step}
            ),
            encoding="utf-8",
        )
        (checkpoint / "training_state.pt").write_bytes(b"state")
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"weights")
        return checkpoint

    def test_complete_atomic_checkpoint_is_ready(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_checkpoint(root, "h16")
            ready, running, failed = evaluate_state(
                ["h16"], 2000, root, {"mhar-stageb-h16"}
            )
        self.assertEqual(ready, {"h16"})
        self.assertFalse(running)
        self.assertFalse(failed)

    def test_missing_weights_is_not_complete(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = self.write_checkpoint(root, "h8")
            (checkpoint / "model.safetensors").unlink()
            self.assertIsNone(checkpoint_step(root, "h8", 2000))

    def test_live_run_without_checkpoint_is_running(self):
        ready, running, failed = evaluate_state(
            ["h4"], 2000, "/missing", {"mhar-stageb-h4"}
        )
        self.assertFalse(ready)
        self.assertEqual(running, {"h4"})
        self.assertFalse(failed)

    def test_dead_run_without_checkpoint_is_failure(self):
        ready, running, failed = evaluate_state(
            ["mixed-k4-best"], 2000, "/missing", set()
        )
        self.assertFalse(ready)
        self.assertFalse(running)
        self.assertEqual(failed, {"mixed-k4-best"})

    def test_manifest_step_must_match_directory(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_checkpoint(root, "mixed-k2", manifest_step=1999)
            self.assertEqual(checkpoint_step(root, "mixed-k2", 2000), 1999)
            ready, running, failed = evaluate_state(
                ["mixed-k2"], 2000, root, set()
            )
        self.assertFalse(ready)
        self.assertFalse(running)
        self.assertEqual(failed, {"mixed-k2"})

    def test_variant_subset_is_validated(self):
        available = ["h16", "h8", "mixed-k2"]
        self.assertEqual(
            select_variants("mixed-k2,h16", available), ["mixed-k2", "h16"])
        with self.assertRaises(ValueError):
            select_variants("h8,h8", available)
        with self.assertRaises(ValueError):
            select_variants("missing", available)


if __name__ == "__main__":
    unittest.main()
