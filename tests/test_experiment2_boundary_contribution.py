"""CPU tests for the corrected Experiment 2 boundary contribution model."""

import unittest

import numpy as np

from src.experiments.experiment2_boundary_contribution import (
    additive_cross_validation,
    candidate_rankings,
    compatible_pairs,
    fit_additive,
    predict_additive,
    ridge_features,
    selection_manifest,
    stable_float,
    transfer_summary,
)
from src.attention_residuals.mhar_partition import (
    generate_adjacent_merge_partitions,
    merged_boundaries,
)


def complete_k4_design():
    partitions = generate_adjacent_merge_partitions(16, 4)
    indicators = np.zeros((len(partitions), 15), dtype=float)
    for row, partition in enumerate(partitions):
        indicators[row, list(merged_boundaries(partition))] = 1.0
    return indicators


class AdditiveModelTest(unittest.TestCase):
    def test_centered_constrained_fit_is_identified_and_exact(self):
        indicators = complete_k4_design()
        beta = np.linspace(-0.07, 0.07, 15)
        beta -= beta.mean()
        target = 0.8 + (indicators - indicators.mean(axis=0)) @ beta

        model = fit_additive(indicators, target)
        prediction = predict_additive(model, indicators)

        self.assertAlmostEqual(float(model["beta"].sum()), 0.0, places=12)
        np.testing.assert_allclose(model["beta"], beta, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(prediction, target, rtol=1e-10, atol=1e-10)
        result = additive_cross_validation(indicators, target, seed=20260826)
        self.assertGreater(result["metrics"]["r2"], 0.999999)
        # Many synthetic partitions tie exactly; tiny floating-point differences
        # inside folds can perturb average tie ranks without changing predictions.
        self.assertGreater(result["metrics"]["spearman"], 0.998)

    def test_pairwise_design_has_expected_structural_rank(self):
        indicators = complete_k4_design()
        pairs = compatible_pairs()
        features = ridge_features(indicators, pairs)
        augmented = np.column_stack([np.ones(len(indicators)), features])
        self.assertEqual(len(pairs), 91)
        self.assertEqual(augmented.shape[1], 107)
        self.assertEqual(np.linalg.matrix_rank(augmented), 91)


class TransferSelectionTest(unittest.TestCase):
    def test_frozen_outputs_ignore_sub_precision_blas_differences(self):
        left = stable_float(0.12345678901234)
        right = stable_float(0.12345678901235)
        self.assertEqual(left, right)

        beta_left = np.linspace(-0.07, 0.07, 15)
        beta_left -= beta_left.mean()
        beta_right = beta_left.copy()
        beta_right[0] += 1e-16
        self.assertEqual(
            candidate_rankings(beta_left, 3),
            candidate_rankings(beta_right, 3),
        )

    def test_rankings_and_frozen_roles_have_registered_counts(self):
        beta = np.linspace(-0.07, 0.07, 15)
        beta -= beta.mean()
        for num_merges, expected in ((3, 286), (5, 462)):
            rankings = candidate_rankings(beta, num_merges)
            self.assertEqual(len(rankings), expected)
            manifest = selection_manifest(
                rankings, num_merges=num_merges,
                source_hash="a" * 64, score_hash="b" * 64,
                seed=20260826, uniform_size=30)
            candidates = manifest["candidates"]
            self.assertEqual(
                sum("uniform_transfer" in row["roles"] for row in candidates), 30)
            self.assertEqual(sum("target_top" in row["roles"] for row in candidates), 10)
            self.assertEqual(sum("target_middle" in row["roles"] for row in candidates), 5)
            self.assertEqual(sum("target_bottom" in row["roles"] for row in candidates), 5)
            self.assertTrue(all(
                len(row["merged_boundaries"]) == num_merges for row in candidates))

    def test_transfer_summary_keeps_uniform_and_targeted_purposes_separate(self):
        rows = []
        for index in range(30):
            roles = ["uniform_transfer"]
            if index < 10:
                roles.append("target_top")
            elif index < 15:
                roles.append("target_middle")
            elif index < 20:
                roles.append("target_bottom")
            rows.append({
                "roles": roles,
                "predicted_score": index / 100,
                "actual_delta_nll": 0.5 + index / 50,
            })
        summary = transfer_summary(rows)
        self.assertEqual(summary["uniform_count"], 30)
        self.assertAlmostEqual(summary["uniform_spearman"], 1.0)
        self.assertTrue(summary["directional_conditions_met"])


if __name__ == "__main__":
    unittest.main()
