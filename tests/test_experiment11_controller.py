"""Static and checkpoint-gate tests for the Experiment 11 controller."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.evaluate.run_experiment11_controller import (
    MILESTONES,
    RUNS,
    checkpoint_complete,
    training_jobs,
)


class Experiment11ControllerTest(unittest.TestCase):
    def test_controller_matrix_and_milestones_are_frozen(self):
        self.assertEqual(len(RUNS), 9)
        self.assertEqual(MILESTONES, (500, 1000, 1500, 2000))

    def test_checkpoint_gate_checks_step_data_position_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "training_state.pt").write_bytes(b"state")
            (checkpoint / "model.safetensors").write_bytes(b"model")
            manifest = {
                "global_step": 500,
                "chunks_consumed": 16_000,
                "run_identity": {
                    "seed": 42,
                    "mode": "full_mh",
                    "attnres_heads": 8,
                    "num_heads": 16,
                    "num_kv_heads": 8,
                    "experiment11_run_id": "s2q8-l000",
                    "experiment11_soft_q_groups": 8,
                },
            }
            (checkpoint / "training_manifest.json").write_text(json.dumps(manifest))
            self.assertTrue(checkpoint_complete(checkpoint, "s2q8-l000", 500))
            manifest["chunks_consumed"] += 1
            (checkpoint / "training_manifest.json").write_text(json.dumps(manifest))
            self.assertFalse(checkpoint_complete(checkpoint, "s2q8-l000", 500))

    def test_fresh_jobs_capture_step_zero_and_resume_jobs_do_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            (repo / "scripts/train").mkdir(parents=True)
            args = SimpleNamespace(
                output_root=root / "output",
                repo=repo,
                python=Path("/python"),
                artifact=root / "artifact.pt",
                data_files="/data/*.parquet",
                master_port_base=31000,
            )
            fresh = training_jobs(args, 500)[0]
            self.assertEqual(len(fresh.command), 3)
            script = Path("scripts/train/run_experiment11_screen.sh").read_text()
            self.assertIn("--experiment11_probe_output", script)
            self.assertIn("--experiment11_probe_artifact", script)
            source = args.output_root / "training/s2q8-l000/step-500"
            source.mkdir(parents=True)
            (source / "training_state.pt").write_bytes(b"state")
            (source / "model.safetensors").write_bytes(b"model")
            (source / "training_manifest.json").write_text(json.dumps({
                "global_step": 500,
                "chunks_consumed": 16_000,
                "run_identity": {
                    "seed": 42, "mode": "full_mh", "attnres_heads": 8,
                    "num_heads": 16, "num_kv_heads": 8,
                    "experiment11_run_id": "s2q8-l000",
                    "experiment11_soft_q_groups": 8,
                },
            }))
            # Create valid sources for the other eight jobs as the pool validates all rows.
            for run_id in RUNS[1:]:
                candidate = args.output_root / "training" / run_id / "step-500"
                candidate.mkdir(parents=True)
                (candidate / "training_state.pt").write_bytes(b"state")
                (candidate / "model.safetensors").write_bytes(b"model")
                (candidate / "training_manifest.json").write_text(json.dumps({
                    "global_step": 500, "chunks_consumed": 16_000,
                    "run_identity": {
                        "seed": 42, "mode": "full_mh", "attnres_heads": 8,
                        "num_heads": 16, "num_kv_heads": 8,
                        "experiment11_run_id": run_id,
                        "experiment11_soft_q_groups": 8,
                    },
                }))
            resumed = training_jobs(args, 1000)[0]
            self.assertEqual(len(resumed.command), 4)
            self.assertEqual(Path(resumed.command[-1]), source)


if __name__ == "__main__":
    unittest.main()
