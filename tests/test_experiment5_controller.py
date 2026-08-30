import argparse
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.experiments import experiment5_controller as controller


class ControllerTest(unittest.TestCase):
    def args(self, root):
        return argparse.Namespace(root=str(root), manifest=str(root / "manifest.json"),
                                  role="predicted-good", gpu=0, parent="/parent", artifact="/artifact")

    def test_worker_failure_does_not_evaluate_or_mark_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp))
            manifest = {"protocol": {"offsets": [0, 1, 2, 5, 10, 20, 50, 100]}}
            with patch.object(controller, "load_manifest", return_value=manifest), \
                 patch.object(controller, "verify_step0"), \
                 patch.object(controller, "training_command", return_value=["fake-training"]), \
                 patch.object(controller.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "fake-training")) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    controller.worker(args)
            self.assertEqual(run.call_count, 1)
            self.assertFalse((Path(tmp) / "COMPLETE-predicted-good.json").exists())

    def test_worker_trains_once_and_evaluates_all_seven_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp))
            manifest = {"protocol": {"offsets": [0, 1, 2, 5, 10, 20, 50, 100]}}
            with patch.object(controller, "load_manifest", return_value=manifest), \
                 patch.object(controller, "verify_step0"), \
                 patch.object(controller, "validate_checkpoint") as validate, \
                 patch.object(controller, "sha256_file", return_value="digest"), \
                 patch.object(controller, "training_command", return_value=["fake-training"]), \
                 patch.object(controller.subprocess, "run") as run:
                controller.worker(args)
            self.assertEqual(run.call_count, 8)
            self.assertEqual(validate.call_count, 8)
            self.assertEqual([int(c.args[0][-1]) for c in run.call_args_list[1:]], [1, 2, 5, 10, 20, 50, 100])
            self.assertTrue((Path(tmp) / "COMPLETE-predicted-good.json").is_file())

    def test_failed_step0_stops_before_training_analysis_and_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "new-experiment"
            args = self.args(root)
            with patch("torch.cuda.device_count", return_value=3), \
                 patch.object(controller.subprocess, "check_output", return_value=""), \
                 patch.object(controller.shutil, "disk_usage", return_value=argparse.Namespace(free=200 * 2**30)), \
                 patch.object(controller, "source_commit", return_value="test"), \
                 patch.object(controller, "prepare"), \
                 patch.object(controller, "parallel_phase") as phase, \
                 patch.object(controller, "step0_gate", side_effect=RuntimeError("parity failed")), \
                 patch.object(controller, "analyze") as analyze, \
                 patch.object(controller.subprocess, "run") as shell:
                with self.assertRaisesRegex(RuntimeError, "parity failed"):
                    controller.run(args)
            phase.assert_called_once_with(args, "step0")
            analyze.assert_not_called(); shell.assert_not_called()
            self.assertTrue((root / "FAILED.json").is_file())
            self.assertFalse((root / "READY_FOR_BACKUP.json").exists())

    def test_wrong_backup_digest_refuses_acknowledgment(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp))
            args.pushed_commit = "a" * 40
            args.summary_sha256 = "wrong-local-hash"
            with patch.object(controller, "verify_completion"), \
                 patch.object(controller, "sha256_file", return_value="actual-server-hash"):
                with self.assertRaisesRegex(RuntimeError, "differs"):
                    controller.acknowledge(args)
            self.assertFalse((Path(tmp) / "BACKUP_PUSH_ACK.json").exists())


if __name__ == "__main__":
    unittest.main()
