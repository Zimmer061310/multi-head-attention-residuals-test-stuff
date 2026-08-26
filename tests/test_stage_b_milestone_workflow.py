"""State tests for the Stage B milestone evaluation workflow."""

import json
import inspect
import tempfile
import unittest
from pathlib import Path

from scripts.setup.run_stage_b_milestone_workflow import (
    eval_state,
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


if __name__ == "__main__":
    unittest.main()
