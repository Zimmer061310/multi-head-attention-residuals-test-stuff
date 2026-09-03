"""Frozen protocol tests for Experiment 8."""

import unittest
from pathlib import Path

from src.experiments.experiment8_screening import (
    bootstrap,
    classify_primary,
    load_spec,
    validate_training_manifest,
)


class Experiment8ScreeningTest(unittest.TestCase):
    def test_run_matrix_trains_only_midpoints(self):
        spec, rows = load_spec()
        self.assertEqual(set(rows), {"b", "bhq8", "blq8", "m8", "hq8", "lq8"})
        self.assertEqual(
            {key for key, row in rows.items() if row["train"]}, {"hq8", "bhq8"}
        )
        self.assertEqual(spec["seed"], 42)
        self.assertEqual(spec["head_assignment"]["local_query_head_position"], "even")
        self.assertEqual(spec["head_assignment"]["global_query_head_position"], "odd")

    def test_primary_and_interaction_contrasts(self):
        spec, _ = load_spec()
        contrasts = {row["id"]: row["terms"] for row in spec["contrasts"]}
        self.assertEqual(contrasts["hq8-minus-m8"], {"hq8": 1, "m8": -1})
        self.assertEqual(
            contrasts["h8-interaction"],
            {"hq8": 1, "m8": -1, "bhq8": -1, "b": 1},
        )
        self.assertEqual(contrasts["mhar-midpoint-curvature"], {
            "hq8": 2, "m8": -1, "lq8": -1,
        })

    def test_bootstrap_constant_effect(self):
        result = lambda values: {"splits": {"confirmation": {"sequence_nlls": values}}}
        metrics = bootstrap(
            {"a": result([1, 2, 3]), "b": result([2, 3, 4])},
            {"a": 1, "b": -1},
            "confirmation",
            samples=100,
        )
        self.assertAlmostEqual(metrics["mean_delta_nll"], -1)
        self.assertAlmostEqual(metrics["ci95_low"], -1)

    def test_practical_match_requires_whole_interval_inside_margin(self):
        row = {"splits": {"confirmation": {
            "aggregate_delta_nll": 0.001, "ci95_low": -0.004, "ci95_high": 0.003,
        }}}
        self.assertEqual(classify_primary(row, 0.005), "within_seed_practical_match")
        row["splits"]["confirmation"]["ci95_low"] = -0.006
        self.assertEqual(classify_primary(row, 0.005), "inconclusive_within_seed")

    def test_training_manifest_fails_closed(self):
        spec, rows = load_spec()
        row = rows["hq8"]
        identity = {
            "mode": "full_mh", "attnres_heads": 8, "hidden_size": 1280,
            "num_layers": 36, "num_heads": 16, "num_kv_heads": 8,
            "intermediate_size": 5120, "seq_len": 1024, "steps": 20000,
            "global_batch_size": 32, "lr": 5e-4, "lr_min": 5e-5,
            "warmup": 1000, "seed": 42, "experiment8_variant": "hq8",
            "experiment8_hybrid_q_groups": 8,
            "experiment8_local_head_position": "even",
            "experiment8_global_head_position": "odd",
        }
        validate_training_manifest(
            {"global_step": 2000, "run_identity": identity}, row, spec
        )
        identity["experiment8_local_head_position"] = "odd"
        with self.assertRaisesRegex(ValueError, "training manifest mismatch"):
            validate_training_manifest(
                {"global_step": 2000, "run_identity": identity}, row, spec
            )

    def test_launcher_freezes_recipe_and_two_gpu_parallelism(self):
        root = Path(__file__).resolve().parents[1]
        run = (root / "scripts/train/run_experiment8_screen.sh").read_text()
        launch = (root / "scripts/train/launch_experiment8_2gpu.sh").read_text()
        for value in (
            "--steps 20000", "--stop_after_step 2000", "--save_every 100",
            "--keep_last 1", "--keep_steps 2000", "--reuse_step_checkpoint_as_final",
            '--experiment8_variant "$MHAR_VARIANT"', "--hybrid_q_groups 8",
        ):
            self.assertIn(value, run)
        self.assertIn("mhar-exp8-hq8", launch)
        self.assertIn("mhar-exp8-bhq8", launch)
        self.assertNotIn("sleep", launch)
        self.assertIn('MHAR_GPU_IDS="${MHAR_GPU_IDS:-0,1}"', launch)

    def test_wandb_summary_uses_mapping_and_controller_accepts_finished_eval(self):
        root = Path(__file__).resolve().parents[1]
        analysis = (root / "src/experiments/experiment8_screening.py").read_text()
        controller = (root / "scripts/evaluate/run_experiment8_controller.py").read_text()
        self.assertIn('run.summary.update({', analysis)
        self.assertIn('if result.is_file() and not alive(name):', controller)


if __name__ == "__main__":
    unittest.main()
