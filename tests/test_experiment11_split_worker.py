"""Scheduling invariants for the split-host Experiment 11 worker."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate.run_experiment11_split_worker import (
    ALL_RUNS,
    checkpoint_complete,
    current_step,
    next_target,
)


def write_checkpoint(root: Path, run_id: str, step: int, *, seed: int = 42) -> None:
    checkpoint = root / "training" / run_id / f"step-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"model")
    (checkpoint / "training_state.pt").write_bytes(b"state")
    (checkpoint / "training_manifest.json").write_text(json.dumps({
        "global_step": step,
        "chunks_consumed": step * 32,
        "run_identity": {
            "seed": seed,
            "mode": "full_mh",
            "attnres_heads": 8,
            "num_heads": 16,
            "num_kv_heads": 8,
            "experiment11_run_id": run_id,
            "experiment11_soft_q_groups": 8,
        },
    }))


class Experiment11SplitWorkerTest(unittest.TestCase):
    def test_targets_stop_at_probe_milestones(self):
        self.assertEqual(next_target(0, 200), 200)
        self.assertEqual(next_target(400, 200), 500)
        self.assertEqual(next_target(500, 200), 700)
        self.assertEqual(next_target(900, 200), 1000)
        self.assertEqual(next_target(1900, 200), 2000)

    def test_checkpoint_identity_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id = ALL_RUNS[0]
            write_checkpoint(root, run_id, 100)
            self.assertTrue(checkpoint_complete(root, run_id, 100))
            self.assertEqual(current_step(root, run_id), 100)
            manifest_path = root / "training" / run_id / "step-100/training_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["run_identity"]["seed"] = 43
            manifest_path.write_text(json.dumps(manifest))
            self.assertFalse(checkpoint_complete(root, run_id, 100))
            self.assertEqual(current_step(root, run_id), 0)

    def test_multiple_complete_checkpoints_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id = ALL_RUNS[0]
            write_checkpoint(root, run_id, 100)
            write_checkpoint(root, run_id, 200)
            with self.assertRaisesRegex(RuntimeError, "multiple checkpoints"):
                current_step(root, run_id)


if __name__ == "__main__":
    unittest.main()
