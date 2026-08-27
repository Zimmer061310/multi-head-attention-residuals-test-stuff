"""Synthetic smoothness and mechanism-gate tests for Experiment 3E."""

import unittest

from src.experiments.experiment3_common import h8_boundary_move_candidates
from src.experiments.experiment3_landscape import (
    landscape_analysis,
    normalized_roughness,
    protocol,
    quadratic_loocv_r2,
)


def landscape_rows(*, reverse=False):
    rows = {
        "native-h8": {
            "candidate_id": "native-h8",
            "nll": 5.0,
            "sequence_nlls": [5.0] * 32,
            "boundary_index": None,
            "offset": 0,
        }
    }
    for row in h8_boundary_move_candidates()[1:]:
        delta = (row["offset"] / 40) ** 2 * 0.04
        if reverse:
            delta = -delta
        rows[row["candidate_id"]] = {
            "candidate_id": row["candidate_id"],
            "nll": 5.0 + delta,
            "sequence_nlls": [5.0 + delta] * 32,
            "boundary_index": row["boundary_index"],
            "offset": row["offset"],
        }
    return rows


class Experiment3LandscapeTest(unittest.TestCase):
    def test_quadratic_diagnostic_identifies_smooth_curve(self):
        offsets = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
        values = [(offset / 40) ** 2 for offset in offsets]
        self.assertGreater(quadratic_loocv_r2(offsets, values), 0.99)
        self.assertLessEqual(normalized_roughness(values), 0.25)

    def test_smooth_replicating_landscape_selects_soft_family(self):
        spec = protocol()["landscape"]
        summary = landscape_analysis(
            landscape_rows(), landscape_rows(), spec=spec, samples=200, seed=2)
        self.assertTrue(summary["replicable_landscape"])
        self.assertTrue(summary["soft_learning_compatible"])
        self.assertEqual(summary["mechanism_classification"], "soft-learning-compatible")

    def test_reversed_confirmation_is_insufficient(self):
        spec = protocol()["landscape"]
        summary = landscape_analysis(
            landscape_rows(), landscape_rows(reverse=True),
            spec=spec, samples=200, seed=2)
        self.assertFalse(summary["replicable_landscape"])
        self.assertEqual(
            summary["mechanism_classification"],
            "insufficient-landscape-evidence")


if __name__ == "__main__":
    unittest.main()
