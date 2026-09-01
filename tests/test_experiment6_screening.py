"""Protocol and analysis tests for Experiment 6 screening."""

import unittest
from pathlib import Path

from src.experiments.experiment6_screening import (
    bootstrap_contrast,
    load_spec,
    nonfinite_results,
    validate_training_manifest,
)


class Experiment6ScreeningTest(unittest.TestCase):
    def test_frozen_run_matrix_has_five_new_and_two_reused_runs(self):
        spec, rows = load_spec()
        self.assertEqual(set(rows), {"b", "m4", "c4", "g4", "m8", "c8", "g8"})
        self.assertEqual(
            {variant for variant, row in rows.items() if row["train"]},
            {"b", "c4", "g4", "c8", "g8"},
        )
        self.assertEqual(rows["c4"]["attnres_heads"], rows["c4"]["qkv_groups"])
        self.assertEqual(rows["c8"]["attnres_heads"], rows["c8"]["qkv_groups"])
        self.assertEqual(spec["seed"], 42)
        self.assertEqual(spec["screening_milestone"], 2000)
        self.assertEqual(spec["full_schedule_steps"], 20000)

    def test_primary_contrasts_include_both_interactions(self):
        spec, _ = load_spec()
        contrasts = {row["id"]: row["terms"] for row in spec["primary_contrasts"]}
        self.assertEqual(
            contrasts["h4-interaction"], {"c4": 1, "m4": -1, "g4": -1, "b": 1})
        self.assertEqual(
            contrasts["h8-interaction"], {"c8": 1, "m8": -1, "g8": -1, "b": 1})

    def test_bootstrap_interaction_preserves_constant_effect(self):
        def result(values):
            return {"splits": {"confirmation": {"sequence_nlls": values}}}

        results = {
            "c8": result([2.0, 3.0, 4.0]),
            "m8": result([2.2, 3.2, 4.2]),
            "g8": result([2.1, 3.1, 4.1]),
            "b": result([2.0, 3.0, 4.0]),
        }
        metrics = bootstrap_contrast(
            results, {"c8": 1, "m8": -1, "g8": -1, "b": 1},
            "confirmation", samples=100)
        self.assertAlmostEqual(metrics["mean_delta_nll"], -0.3)
        self.assertAlmostEqual(metrics["ci95_low"], -0.3)
        self.assertAlmostEqual(metrics["ci95_high"], -0.3)

    def test_launcher_preserves_full_schedule_but_stops_at_screen(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts/train/run_experiment6_screen.sh").read_text()
        self.assertIn("--steps 20000", script)
        self.assertIn("--stop_after_step 2000", script)
        self.assertIn("--save_every 100", script)
        self.assertIn("--keep_steps 500,1000,1500,2000", script)
        self.assertIn('--experiment6_variant "$MHAR_VARIANT"', script)
        self.assertIn("--num_heads 16", script)
        self.assertIn("--num_kv_heads 8", script)
        self.assertIn("--seed 42", script)

    def test_training_manifest_validation_fails_closed(self):
        spec, rows = load_spec()
        row = rows["c8"]
        identity = {
            "mode": "full_mh", "attnres_heads": 8,
            "hidden_size": 1280, "num_layers": 36,
            "num_heads": 16, "num_kv_heads": 8,
            "intermediate_size": 5120, "seq_len": 1024,
            "steps": 20000, "global_batch_size": 32,
            "lr": 5e-4, "lr_min": 5e-5, "warmup": 1000, "seed": 42,
            "experiment6_variant": "c8", "experiment6_qkv_groups": 8,
        }
        validate_training_manifest(
            {"global_step": 2000, "run_identity": identity}, row, spec)
        identity["seed"] = 43
        with self.assertRaisesRegex(ValueError, "training manifest mismatch"):
            validate_training_manifest(
                {"global_step": 2000, "run_identity": identity}, row, spec)

    def test_nonfinite_results_are_detected_before_bootstrap(self):
        results = {
            "b": {"splits": {
                "discovery": {"nll": 1.0, "sequence_nlls": [1.0, 2.0]},
                "confirmation": {"nll": float("nan"), "sequence_nlls": [1.0]},
            }}
        }
        self.assertEqual(nonfinite_results(results), ["b:confirmation"])


if __name__ == "__main__":
    unittest.main()
