"""Tests for Stage B fixed-data evaluation and the chaos gate."""

import tempfile
import unittest
from pathlib import Path

from figures.gen_fig_stage_b_milestone import plot_summary
from src.experiments.experiment2_stage_b_screening import (
    analyze_results,
    paired_bootstrap,
    spearman,
)


VARIANTS = (
    "h16", "h8", "h4", "mixed-k2", "mixed-k3", "mixed-k4-best",
    "mixed-k5", "mixed-k4-worst",
)


def mock_result(variant, discovery, confirmation, offset=0.0):
    sequence = [confirmation + offset * value for value in (-1.0, -0.5, 0.5, 1.0)]
    return {
        "variant": variant,
        "splits": {
            "discovery": {"nll": discovery, "sequence_nlls": [discovery] * 4},
            "confirmation": {"nll": confirmation, "sequence_nlls": sequence},
        },
    }


class StageBEvaluationTest(unittest.TestCase):
    def test_spearman_detects_same_and_reversed_order(self):
        self.assertAlmostEqual(spearman([1, 2, 3], [4, 5, 6]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [6, 5, 4]), -1.0)

    def test_paired_bootstrap_preserves_exact_constant_delta(self):
        result = paired_bootstrap([2.0, 3.0, 4.0], [1.5, 2.5, 3.5], samples=100)
        self.assertAlmostEqual(result["mean_delta_nll"], 0.5)
        self.assertAlmostEqual(result["ci95_low"], 0.5)
        self.assertAlmostEqual(result["ci95_high"], 0.5)

    def test_stable_decisive_result_stops_at_2000(self):
        values = {
            "mixed-k4-best": 2.00,
            "mixed-k3": 2.01,
            "mixed-k2": 2.02,
            "mixed-k5": 2.03,
            "h8": 2.04,
            "h16": 2.05,
            "h4": 2.06,
            "mixed-k4-worst": 2.08,
        }
        results = {
            variant: mock_result(variant, value, value, offset=0.0001)
            for variant, value in values.items()
        }
        summary = analyze_results(results)
        self.assertFalse(summary["chaos"])
        self.assertEqual(summary["decision"], "stop_at_2000")
        self.assertAlmostEqual(summary["rank_spearman"], 1.0)

    def test_unstable_indistinguishable_result_resumes(self):
        results = {}
        for index, variant in enumerate(VARIANTS):
            results[variant] = mock_result(
                variant, 2.0 + index * 1e-6, 2.0 - index * 1e-6, offset=0.01)
            results[variant]["splits"]["confirmation"]["sequence_nlls"] = [
                2.0, 2.01, 1.99, 2.0]
        summary = analyze_results(results)
        self.assertTrue(summary["chaos"])
        self.assertEqual(summary["decision"], "resume_to_5000")
        self.assertLess(summary["rank_spearman"], 0.5)

    def test_publication_figure_writes_png_and_pdf(self):
        values = {variant: 2.0 + index * 0.01 for index, variant in enumerate(VARIANTS)}
        results = {
            variant: mock_result(variant, value, value, offset=0.0001)
            for variant, value in values.items()
        }
        summary = {"milestone": 2000, **analyze_results(results)}
        with tempfile.TemporaryDirectory() as root:
            plot_summary(summary, root)
            self.assertTrue((Path(root) / "fig_stage_b_milestone.png").is_file())
            self.assertTrue((Path(root) / "fig_stage_b_milestone.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
