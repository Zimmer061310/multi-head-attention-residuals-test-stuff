"""Static tests for fresh-server reproducibility assets."""

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class StageBServerSetupTest(unittest.TestCase):
    def setUp(self):
        self.environment = json.loads(
            (ROOT / "stage_b_server_environment.json").read_text(encoding="utf-8"))

    def test_requirements_match_recorded_environment(self):
        text = (ROOT / "requirements-stage-b.txt").read_text(encoding="utf-8")
        pinned = dict(re.findall(r"^([A-Za-z0-9_-]+)==([^\s]+)$", text, re.MULTILINE))
        self.assertEqual(pinned, self.environment["packages"])

    def test_frozen_dataset_identity(self):
        files = self.environment["dataset"]["files"]
        self.assertEqual([row["bytes"] for row in files], [2152819114, 2152222432])
        self.assertEqual(
            [row["sha256"] for row in files],
            [
                "b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871",
                "3fcf2dc69cd52503986276d3d2d26a8c356d0f2ea28a0de4fdbda8cf87755693",
            ],
        )

    def test_seven_gpu_launcher_contains_every_frozen_run(self):
        screening = json.loads(
            (ROOT / "experiment2_stage_b_screening.json").read_text(encoding="utf-8"))
        launcher = (ROOT / "launch_experiment2_stage_b_7gpu.sh").read_text(encoding="utf-8")
        for row in screening["runs"]:
            self.assertIn(row["id"], launcher)
        self.assertIn("0,1,2,3,4,5,6", launcher)


if __name__ == "__main__":
    unittest.main()
