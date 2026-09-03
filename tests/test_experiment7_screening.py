"""Frozen protocol tests for Experiment 7."""

import unittest
from pathlib import Path

from src.experiments.experiment7_screening import bootstrap, load_spec, validate_training_manifest


class Experiment7ScreeningTest(unittest.TestCase):
    def test_run_matrix_and_reuse(self):
        spec, rows = load_spec()
        self.assertEqual(set(rows), {"b", "m4", "lq4", "blq4", "m8", "lq8", "blq8"})
        self.assertEqual({key for key, row in rows.items() if row["train"]},
                         {"lq4", "lq8", "blq4", "blq8"})
        self.assertEqual(rows["lq4"]["attnres_heads"], rows["lq4"]["local_q_groups"])
        self.assertEqual(spec["seed"], 42)

    def test_interaction_contrasts(self):
        spec, _ = load_spec(); contrasts = {row["id"]: row["terms"] for row in spec["contrasts"]}
        self.assertEqual(contrasts["h8-interaction"],
                         {"lq8": 1, "m8": -1, "blq8": -1, "b": 1})
        self.assertEqual(contrasts["lq8-minus-c8"], {"lq8": 1, "c8": -1})

    def test_bootstrap_constant_effect(self):
        result = lambda values: {"splits": {"confirmation": {"sequence_nlls": values}}}
        metrics = bootstrap({"a": result([1, 2, 3]), "b": result([2, 3, 4])},
                            {"a": 1, "b": -1}, "confirmation", samples=100)
        self.assertAlmostEqual(metrics["mean_delta_nll"], -1)
        self.assertAlmostEqual(metrics["ci95_low"], -1)

    def test_training_manifest_fails_closed(self):
        spec, rows = load_spec(); row = rows["lq8"]
        identity = {"mode": "full_mh", "attnres_heads": 8, "hidden_size": 1280,
                    "num_layers": 36, "num_heads": 16, "num_kv_heads": 8,
                    "intermediate_size": 5120, "seq_len": 1024, "steps": 20000,
                    "global_batch_size": 32, "lr": 5e-4, "lr_min": 5e-5,
                    "warmup": 1000, "seed": 42, "experiment7_variant": "lq8",
                    "experiment7_local_q_groups": 8}
        validate_training_manifest({"global_step": 2000, "run_identity": identity}, row, spec)
        identity["seed"] = 43
        with self.assertRaisesRegex(ValueError, "training manifest mismatch"):
            validate_training_manifest({"global_step": 2000, "run_identity": identity}, row, spec)

    def test_launcher_freezes_recipe_and_three_gpu_queue(self):
        root = Path(__file__).resolve().parents[1]
        run = (root / "scripts/train/run_experiment7_screen.sh").read_text()
        launch = (root / "scripts/train/launch_experiment7_3gpu.sh").read_text()
        for value in ("--steps 20000", "--stop_after_step 2000", "--save_every 100",
                      "--keep_last 1", "--keep_steps 2000",
                      "--reuse_step_checkpoint_as_final",
                      '--experiment7_variant "$MHAR_VARIANT"'):
            self.assertIn(value, run)
        self.assertIn("blq4;", launch); self.assertIn("blq8", launch)
        self.assertIn("MHAR_LQ_STAGGER_SECONDS", launch)
        self.assertLess(launch.index("mhar-exp7-blq-queue"), launch.index('MHAR_LQ_STAGGER_SECONDS" -gt'))
        self.assertIn('MHAR_HF_HOME="${MHAR_HF_HOME:-/root/hf-exp7}"', run)

    def test_wandb_summary_and_controller_resume_are_compatible(self):
        root = Path(__file__).resolve().parents[1]
        analysis = (root / "src/experiments/experiment7_screening.py").read_text()
        controller = (root / "scripts/evaluate/run_experiment7_controller.py").read_text()
        self.assertIn('run.summary.update({"decision": summary["decision"]', analysis)
        self.assertIn('if result.is_file() and not alive(name): return', controller)


if __name__ == "__main__": unittest.main()
