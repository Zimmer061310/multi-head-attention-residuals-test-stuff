"""Tests for the Stage B success-only shutdown watcher."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.setup.watch_stage_b_shutdown import evaluate_state


class StageBShutdownWatcherTest(unittest.TestCase):
    def write_final(self, root, variant, step=20000):
        destination = Path(root) / variant / "final"
        destination.mkdir(parents=True)
        (destination / "training_manifest.json").write_text(
            json.dumps({"global_step": step}), encoding="utf-8"
        )

    def test_completed_run_does_not_require_a_live_screen(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_final(root, "h16")
            completed, running, failed = evaluate_state(
                ["h16"], 20000, root, set()
            )
        self.assertEqual(completed, {"h16"})
        self.assertFalse(running)
        self.assertFalse(failed)

    def test_live_run_is_running(self):
        completed, running, failed = evaluate_state(
            ["h4"], 20000, "/missing", {"mhar-stageb-h4"}
        )
        self.assertFalse(completed)
        self.assertEqual(running, {"h4"})
        self.assertFalse(failed)

    def test_missing_screen_without_final_is_failure(self):
        completed, running, failed = evaluate_state(
            ["mixed-k4-best"], 20000, "/missing", set()
        )
        self.assertFalse(completed)
        self.assertFalse(running)
        self.assertEqual(failed, {"mixed-k4-best"})

    def test_wrong_final_step_is_failure_after_screen_exit(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_final(root, "h8", step=19999)
            completed, running, failed = evaluate_state(
                ["h8"], 20000, root, set()
            )
        self.assertFalse(completed)
        self.assertFalse(running)
        self.assertEqual(failed, {"h8"})


if __name__ == "__main__":
    unittest.main()
