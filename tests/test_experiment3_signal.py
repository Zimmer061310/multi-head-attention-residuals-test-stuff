"""Synthetic gate tests for Experiments 3A and 3B."""

import json
import unittest
from pathlib import Path

from src.experiments.experiment3_signal import analyze_signal, protocol
from src.experiments.experiment3_temporal import analyze_temporal


def result_rows(values, *, seed=42, step=1500):
    rows = {
        "native-h16": {
            "candidate_id": "native-h16",
            "boundary": None,
            "nll": 5.0,
            "sequence_nlls": [5.0] * 32,
            "seed": seed,
            "step": step,
        }
    }
    for boundary, delta in enumerate(values):
        identifier = f"remove-{boundary:02d}"
        rows[identifier] = {
            "candidate_id": identifier,
            "boundary": boundary,
            "nll": 5.0 + delta,
            "sequence_nlls": [5.0 + delta] * 32,
            "seed": seed,
            "step": step,
        }
    return rows


class Experiment3SignalTest(unittest.TestCase):
    def test_protocol_has_locked_candidate_and_time_spaces(self):
        spec = protocol()
        self.assertEqual(spec["probe_steps"], [1000, 1500, 2000, 3000])
        self.assertEqual(spec["single_removal_boundaries"], list(range(15)))
        self.assertEqual(spec["seeds"], [42, 43, 44])

    def test_signal_gate_uses_frozen_best_and_worst(self):
        discovery = result_rows([index * 0.01 for index in range(15)])
        confirmation = result_rows([index * 0.008 for index in range(15)])
        selection = {
            "best": "remove-00",
            "worst": "remove-14",
            "top_three": ["remove-00", "remove-01", "remove-02"],
        }
        summary = analyze_signal(
            discovery, confirmation, selection, samples=200, seed=9)
        self.assertAlmostEqual(summary["discovery_confirmation_spearman"], 1.0)
        self.assertTrue(summary["signal_gate_passed"])
        self.assertLess(
            summary["best_minus_worst_confirmation"]["ci95_high"], 0)

    def test_signal_gate_rejects_reversed_confirmation(self):
        discovery = result_rows([index * 0.01 for index in range(15)])
        confirmation = result_rows([(14 - index) * 0.01 for index in range(15)])
        selection = {
            "best": "remove-00",
            "worst": "remove-14",
            "top_three": ["remove-00", "remove-01", "remove-02"],
        }
        summary = analyze_signal(
            discovery, confirmation, selection, samples=200, seed=9)
        self.assertFalse(summary["signal_gate_passed"])
        self.assertAlmostEqual(summary["discovery_confirmation_spearman"], -1.0)

    def test_temporal_gate_tracks_rank_not_absolute_nll(self):
        base = {f"remove-{index:02d}": float(index) for index in range(15)}
        discovery = {
            1000: base,
            1500: {key: value * 2 + 10 for key, value in base.items()},
            2000: {key: value * 3 - 4 for key, value in base.items()},
            3000: {key: value * 4 + 2 for key, value in base.items()},
        }
        confirmation = {
            step: {key: value + 0.1 for key, value in vector.items()}
            for step, vector in discovery.items()
        }
        summary = analyze_temporal(
            discovery,
            confirmation,
            adjacent_pairs=((1000, 1500), (1500, 2000), (2000, 3000)),
        )
        self.assertTrue(summary["stability_gate_passed"])
        self.assertEqual(summary["confirmation_same_sign_pairs"], 3)
        self.assertAlmostEqual(summary["median_primary_discovery_spearman"], 1.0)
        json.dumps(summary)


if __name__ == "__main__":
    unittest.main()
