import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.setup import preflight_experiment3_server as preflight
from src.experiments.experiment1_partition_compatibility import save_fixed_eval_artifact


class Experiment3ServerSetupTests(unittest.TestCase):
    def test_probe_launcher_separates_controller_and_locked_training_worktrees(self):
        root = Path(__file__).resolve().parents[1]
        launcher = (root / "scripts/train/launch_experiment3_h16_probes_3gpu.sh").read_text()
        runner = (root / "scripts/train/run_experiment3_h16_probe_seed.sh").read_text()
        branch_runner = (
            root / "scripts/train/run_experiment3_actionability_branch.sh"
        ).read_text()
        self.assertIn("MHAR_CONTROLLER_REPO_DIR", launcher)
        self.assertIn("MHAR_TRAINING_REPO_DIR", launcher)
        self.assertIn("MHAR_RESUME_SEED${seed}", launcher)
        self.assertIn('cd "$MHAR_TRAINING_REPO_DIR"', runner)
        self.assertIn('cd "$MHAR_REPO_DIR"', branch_runner)
        self.assertNotIn("MHAR_TRAINING_REPO_DIR", branch_runner)

    def test_locked_shard_is_new_and_content_addressed(self):
        environment = preflight.ENVIRONMENT
        candidate = environment["experiment3_evaluation_dataset"]["files"][0]
        prior = {
            row["sha256"]
            for key in ("dataset", "evaluation_dataset")
            for row in environment[key]["files"]
        }
        self.assertEqual(candidate["name"], "003_00000.parquet")
        self.assertEqual(candidate["bytes"], 2152437524)
        self.assertEqual(
            candidate["sha256"],
            "22184e6eb25759ddd97783751ffc73e1705dfa2542e630dae1f2a8bac8ee6ddb",
        )
        self.assertNotIn(candidate["sha256"], prior)

    def test_artifact_preflight_accepts_locked_shape_and_source(self):
        spec = preflight.ENVIRONMENT["experiment3_evaluation_dataset"]["files"][0]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fixed_eval.pt"
            values = torch.arange(512 * 1024, dtype=torch.int64).reshape(512, 1024)
            save_fixed_eval_artifact(
                output,
                values,
                values.flip(0),
                {
                    "dataset": {"matched_files": [{
                        "path": "/locked/003_00000.parquet",
                        "bytes": spec["bytes"],
                        "sha256": spec["sha256"],
                    }]},
                    "sequence_length": 1024,
                },
            )
            observed = preflight.check_artifact(output)
            sidecar = json.loads(Path(str(output) + ".manifest.json").read_text())
            self.assertEqual(observed, sidecar["artifact_sha256"])


if __name__ == "__main__":
    unittest.main()
