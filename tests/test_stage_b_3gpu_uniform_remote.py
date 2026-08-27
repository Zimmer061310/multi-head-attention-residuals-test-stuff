import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.setup.run_stage_b_3gpu_uniform_remote import (
    pause_status_path,
    primary_summary_path,
    read_primary_decision,
    ssh_base,
)


class StageB3GpuUniformRemoteTest(unittest.TestCase):
    def test_paths_are_milestone_scoped(self):
        self.assertEqual(
            pause_status_path("/tmp/output", 2000),
            Path("/tmp/output/milestone-2000-uniform-remote-pause-status.json"),
        )
        self.assertEqual(
            primary_summary_path("/tmp/output", 5000),
            Path("/tmp/output/milestones/step-5000/analysis/summary.json"),
        )

    def test_ssh_command_is_batch_only(self):
        command = ssh_base("example", 2222, "/tmp/key")
        self.assertIn("BatchMode=yes", command)
        self.assertEqual(command[-1], "root@example")

    @patch("scripts.setup.run_stage_b_3gpu_uniform_remote.subprocess.run")
    def test_reads_only_frozen_gate_decision(self, run):
        run.return_value.returncode = 0
        run.return_value.stdout = json.dumps({"decision": "resume_to_5000"})
        decision = read_primary_decision(
            2000, "/tmp/output", "example", 2222, "/tmp/key")
        self.assertEqual(decision, "resume_to_5000")

    @patch("scripts.setup.run_stage_b_3gpu_uniform_remote.subprocess.run")
    def test_missing_summary_returns_none(self, run):
        run.return_value.returncode = 1
        run.return_value.stdout = ""
        self.assertIsNone(read_primary_decision(
            2000, "/tmp/output", "example", 2222, "/tmp/key"))


if __name__ == "__main__":
    unittest.main()
