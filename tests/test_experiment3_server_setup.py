import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from scripts.setup import preflight_experiment3_server as preflight
from scripts.setup.download_stage_b_data import resolve_source_url
from src.experiments.experiment1_partition_compatibility import save_fixed_eval_artifact


class Experiment3ServerSetupTests(unittest.TestCase):
    def test_artifact_only_preflight_does_not_require_local_source_shard(self):
        with mock.patch.object(
            sys,
            "argv",
            ["preflight", "--artifact", "/content-addressed/fixed_eval.pt", "--artifact-only"],
        ), mock.patch.object(preflight, "check_artifact", return_value="locked-digest"), \
                mock.patch.object(preflight, "check_source_shard") as source_check:
            preflight.main()
        source_check.assert_not_called()

    def test_huggingface_mirror_changes_transport_only(self):
        source = (
            "https://huggingface.co/datasets/org/repo/resolve/revision/file.parquet"
        )
        self.assertEqual(
            resolve_source_url(source, "https://hf-mirror.com"),
            "https://hf-mirror.com/datasets/org/repo/resolve/revision/file.parquet",
        )
        self.assertEqual(
            resolve_source_url("https://example.com/file", "https://hf-mirror.com"),
            "https://example.com/file",
        )

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
        self.assertIn("MHAR_MASTER_PORT_BASE", launcher)
        self.assertIn("watch_experiment3_h16_probe_stop.sh", launcher)
        self.assertIn('cd "$MHAR_TRAINING_REPO_DIR"', runner)
        self.assertIn('--master_port "$MHAR_MASTER_PORT"', runner)
        self.assertIn('STOP_ARGS=(--stop_after_step "$MHAR_TARGET_STEP")', runner)
        self.assertIn('cd "$MHAR_REPO_DIR"', branch_runner)
        self.assertNotIn("MHAR_TRAINING_REPO_DIR", branch_runner)

    def test_distributed_split_worker_is_atomic_and_content_checked(self):
        root = Path(__file__).resolve().parents[1]
        worker = (
            root / "scripts/evaluate/run_experiment3_distributed_split_worker.sh"
        ).read_text()
        self.assertIn("training_manifest.json", worker)
        self.assertIn("training_state.pt", worker)
        self.assertIn("model.safetensors", worker)
        self.assertIn("checkpoint_partial", worker)
        self.assertIn("checkpoint_complete \"$checkpoint_partial\"", worker)
        self.assertIn("mv \"$checkpoint_partial\" \"$MHAR_CHECKPOINT\"", worker)
        self.assertIn("--artifact-only", worker)

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
