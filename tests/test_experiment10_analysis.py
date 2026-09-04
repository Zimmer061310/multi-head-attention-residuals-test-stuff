import unittest

import numpy as np

from src.experiments.experiment10_per_group_contribution import (
    average_ranks,
    bootstrap_linear_combination,
    spearman,
)


class Experiment10AnalysisTest(unittest.TestCase):
    def test_spearman_handles_order_and_ties(self):
        self.assertAlmostEqual(spearman(np.arange(8.0), np.arange(8.0)), 1.0)
        self.assertAlmostEqual(spearman(np.arange(8.0), np.arange(7, -1, -1)), -1.0)
        np.testing.assert_allclose(average_ranks(np.asarray([1.0, 1.0, 3.0])), [0.5, 0.5, 2.0])

    def test_linear_combination_recovers_interaction(self):
        reference = np.asarray([1.0, 2.0, 3.0])
        local = reference + 0.1
        global_ = reference + 0.2
        whole = reference + 0.35
        result = bootstrap_linear_combination(
            [whole, local, global_, reference],
            [1.0, -1.0, -1.0, 1.0],
            samples=200,
            seed=10,
        )
        self.assertAlmostEqual(result["mean_delta_nll"], 0.05)
        self.assertAlmostEqual(result["ci95_low"], 0.05)
        self.assertAlmostEqual(result["ci95_high"], 0.05)

    def test_linear_combination_rejects_mismatched_shapes(self):
        with self.assertRaisesRegex(ValueError, "equal"):
            bootstrap_linear_combination(
                [np.ones(2), np.ones(3)], [1.0, -1.0], samples=10, seed=1
            )


if __name__ == "__main__":
    unittest.main()
