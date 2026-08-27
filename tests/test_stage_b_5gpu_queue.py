"""Tests for the five-GPU cloned-server queue."""

import tempfile
import unittest
from pathlib import Path

from scripts.setup.run_stage_b_5gpu_queue import (
    MIXED_VARIANTS,
    UNIFORM_VARIANTS,
    latest_atomic_checkpoint,
)


class StageB5GpuQueueTest(unittest.TestCase):
    def write_checkpoint(self, root, variant, step):
        checkpoint = Path(root) / variant / f"step-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "training_manifest.json").write_text(
            f'{{"global_step": {step}}}', encoding="utf-8")
        (checkpoint / "training_state.pt").write_bytes(b"state")
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"weights")

    def test_queue_covers_eight_runs_on_five_then_three_gpus(self):
        self.assertEqual(len(MIXED_VARIANTS), 5)
        self.assertEqual(len(UNIFORM_VARIANTS), 3)
        self.assertEqual(len(set(MIXED_VARIANTS) | set(UNIFORM_VARIANTS)), 8)

    def test_latest_atomic_checkpoint_before_target(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_checkpoint(root, "h16", 1000)
            self.write_checkpoint(root, "h16", 1500)
            self.write_checkpoint(root, "h16", 2000)
            selected = latest_atomic_checkpoint(root, "h16", 2000)
        self.assertEqual(selected.name, "step-1500")

    def test_incomplete_checkpoint_is_ignored(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_checkpoint(root, "mixed-k2", 500)
            incomplete = Path(root) / "mixed-k2" / "step-1000"
            incomplete.mkdir()
            (incomplete / "training_manifest.json").write_text(
                '{"global_step": 1000}', encoding="utf-8")
            selected = latest_atomic_checkpoint(root, "mixed-k2", 2000)
        self.assertEqual(selected.name, "step-500")


if __name__ == "__main__":
    unittest.main()
