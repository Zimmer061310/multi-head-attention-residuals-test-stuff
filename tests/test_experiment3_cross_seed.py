"""Synthetic replication-gate tests for Experiment 3D."""

import unittest

from src.experiments.experiment3_cross_seed import cross_seed_analysis


def seed_record(seed, good_minus_random, *, signal=True, stability=True):
    good_value = 4.0
    random_value = good_value - good_minus_random
    bad_value = 4.2
    contrast = lambda candidate, reference, value: {
        "split": "confirmation",
        "candidate": candidate,
        "reference": reference,
        "mean_delta_nll": value,
        "ci95_low": value - 0.01,
        "ci95_high": value + 0.01,
    }
    return {
        "signal": {
            "signal_gate_passed": signal,
            "discovery_confirmation_spearman": 0.8 if signal else 0.1,
        },
        "temporal": {
            "stability_gate_passed": stability,
            "median_primary_discovery_spearman": 0.7 if stability else 0.0,
        },
        "actionability": {
            "actionability_gate_passed": good_minus_random < -0.01,
            "contrasts": [
                contrast("predicted-good", "random", good_minus_random),
                contrast("predicted-good", "predicted-bad", good_value - bad_value),
            ],
        },
        "selection": {
            "branches": {
                "predicted-good": {"boundary": seed % 15},
                "random": {"boundary": (seed + 3) % 15},
                "predicted-bad": {"boundary": (seed + 7) % 15},
            },
        },
        "good": {
            "splits": {"confirmation": {"sequence_nlls": [good_value] * 32}},
        },
        "random": {
            "splits": {"confirmation": {"sequence_nlls": [random_value] * 32}},
        },
    }


class Experiment3CrossSeedTest(unittest.TestCase):
    gate = {
        "minimum_signal_pass_seeds": 2,
        "minimum_stability_pass_seeds": 2,
        "minimum_good_beats_random_seeds": 2,
        "mean_good_minus_random_must_be_negative": True,
    }

    def test_replication_does_not_require_matching_boundary_ids(self):
        records = {
            42: seed_record(42, -0.05),
            43: seed_record(43, -0.03),
            44: seed_record(44, -0.02),
        }
        summary = cross_seed_analysis(
            records, samples=200, bootstrap_seed=3, gate_spec=self.gate)
        self.assertTrue(summary["replication_gate_passed"])
        self.assertEqual(summary["good_beats_random_seed_count"], 3)
        self.assertEqual(
            len({row["good_boundary"] for row in summary["seed_rows"]}), 3)

    def test_replication_rejects_one_successful_seed(self):
        records = {
            42: seed_record(42, -0.05),
            43: seed_record(43, 0.03, signal=False, stability=False),
            44: seed_record(44, 0.02, signal=False, stability=False),
        }
        summary = cross_seed_analysis(
            records, samples=200, bootstrap_seed=3, gate_spec=self.gate)
        self.assertFalse(summary["replication_gate_passed"])
        self.assertEqual(summary["signal_pass_seed_count"], 1)


if __name__ == "__main__":
    unittest.main()
