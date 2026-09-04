import unittest

import numpy as np

from src.experiments.experiment9_head_contribution import (
    bootstrap_delta,
    hierarchical_bootstrap,
)


class Experiment9AnalysisTest(unittest.TestCase):
    def test_paired_bootstrap_recovers_constant_effect(self):
        reference = np.asarray([1.0, 2.0, 3.0])
        candidate = reference + 0.25
        result = bootstrap_delta(candidate, reference, samples=200, seed=9)
        self.assertAlmostEqual(result["mean_delta_nll"], 0.25)
        self.assertAlmostEqual(result["ci95_low"], 0.25)
        self.assertAlmostEqual(result["ci95_high"], 0.25)

    def test_hierarchical_bootstrap_recovers_constant_effect(self):
        values = np.full((4, 8), 0.125)
        result = hierarchical_bootstrap(values, samples=200, seed=10)
        self.assertAlmostEqual(result["mean_delta_nll"], 0.125)
        self.assertAlmostEqual(result["ci95_low"], 0.125)
        self.assertEqual(result["bootstrap_unit"], "derangements_and_paired_sequences")

    def test_bootstraps_reject_wrong_shapes(self):
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            bootstrap_delta(np.ones((2, 2)), np.ones((2, 2)), samples=10, seed=1)
        with self.assertRaisesRegex(ValueError, "condition"):
            hierarchical_bootstrap(np.ones(5), samples=10, seed=1)


if __name__ == "__main__":
    unittest.main()
