"""State tests for the Stage B milestone evaluation workflow."""

import json
import inspect
import tempfile
import unittest
from pathlib import Path

from scripts.setup.run_stage_b_milestone_workflow import (
    eval_state,
    five_gpu_eval_assignments,
    five_gpu_eval_state,
    launch_five_gpu_queue,
    launch_resume,
    load_pause_status,
)


class StageBMilestoneWorkflowTest(unittest.TestCase):
    def test_evaluation_state_distinguishes_complete_running_and_failed(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "h16"
            destination.mkdir()
            (destination / "result.json").write_text("{}", encoding="utf-8")
            completed, running, failed = eval_state(
                ["h16", "h8", "h4"], root,
                {"mhar-stageb-eval-2000-h8"}, 2000)
        self.assertEqual(completed, {"h16"})
        self.assertEqual(running, {"h8"})
        self.assertEqual(failed, {"h4"})

    def test_missing_pause_status_returns_none(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(load_pause_status(root, 2000))

    def test_pause_status_is_loaded(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "milestone-2000-pause-status.json"
            path.write_text(json.dumps({"complete": True}), encoding="utf-8")
            self.assertTrue(load_pause_status(root, 2000)["complete"])

    def test_followup_workflow_uses_module_invocation(self):
        source = inspect.getsource(launch_resume)
        self.assertIn('"scripts.setup.run_stage_b_milestone_workflow"', source)
        self.assertNotIn(
            'str(ROOT / "scripts/setup/run_stage_b_milestone_workflow.py")', source)

    def test_five_gpu_followup_uses_queue(self):
        source = inspect.getsource(launch_five_gpu_queue)
        self.assertIn('"scripts.setup.run_stage_b_5gpu_queue"', source)
        self.assertIn('"--no-further-resume"', source)

    def test_five_gpu_evaluation_queues_all_eight_variants(self):
        variants = [f"v{index}" for index in range(8)]
        assignments = five_gpu_eval_assignments(variants)
        self.assertEqual(len(assignments), 5)
        self.assertEqual([len(values) for values in assignments], [2, 2, 2, 1, 1])
        self.assertEqual({value for values in assignments for value in values}, set(variants))

    def test_five_gpu_eval_state_treats_queued_variants_as_running(self):
        with tempfile.TemporaryDirectory() as root:
            result = Path(root) / "h16"
            result.mkdir()
            (result / "result.json").write_text("{}", encoding="utf-8")
            completed, running, failed = five_gpu_eval_state(
                ["h16", "h8", "h4"], root,
                {"mhar-stageb-eval-2000-gpu-0"}, 2000)
        self.assertEqual(completed, {"h16"})
        self.assertEqual(running, {"h8", "h4"})
        self.assertFalse(failed)


if __name__ == "__main__":
    unittest.main()
