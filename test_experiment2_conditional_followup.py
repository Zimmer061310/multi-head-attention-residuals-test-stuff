"""Tests for the pre-registered conditional Experiment 2 continuation."""

import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from experiment2_boundary_contribution import candidate_rankings
from experiment2_conditional_followup import (
    FOLLOWUP_SPACES,
    evaluation_command,
    exhaustive_manifest,
    gate_decisions,
    sampled_manifest,
)


class ConditionalFollowupTest(unittest.TestCase):
    def setUp(self):
        self.beta = np.linspace(-0.07, 0.07, 15)
        self.beta -= self.beta.mean()

    def test_directional_gates_control_the_correct_sides(self):
        decisions = gate_decisions({
            "k3": {"directional_conditions_met": True},
            "k5": {"directional_conditions_met": False},
        })
        self.assertEqual(decisions, {1: True, 2: True, 6: False, 7: False})

    def test_registered_candidate_space_sizes(self):
        for k, expected in FOLLOWUP_SPACES.items():
            self.assertEqual(len(candidate_rankings(self.beta, k)), expected)

    def test_exhaustive_and_sampled_manifests(self):
        common = {
            "source_hash": "a" * 64,
            "score_hash": "b" * 64,
            "gate_hash": "c" * 64,
            "source_commit": "d" * 40,
        }
        exhaustive = exhaustive_manifest(
            candidate_rankings(self.beta, 1), num_merges=1, **common)
        self.assertEqual(len(exhaustive["candidates"]), 15)
        self.assertTrue(all(
            row["roles"] == ["exhaustive_transfer"]
            for row in exhaustive["candidates"]))

        sampled = sampled_manifest(
            candidate_rankings(self.beta, 2), num_merges=2,
            seed=20260826, uniform_size=30, **common)
        self.assertEqual(
            sum("uniform_transfer" in row["roles"] for row in sampled["candidates"]),
            30)
        self.assertEqual(
            sum("target_top" in row["roles"] for row in sampled["candidates"]),
            10)

    def test_evaluation_command_preserves_frozen_runtime_settings(self):
        args = Namespace(
            checkpoint="checkpoint", artifact="fixed.pt", device="cuda",
            dtype="bf16", batch_size=1, wandb_mode="online",
            wandb_project="MHAR Stuff", wandb_group="group")
        command = evaluation_command(
            args, k=2, selection_path=Path("selection.json"),
            output_dir=Path("output"))
        self.assertIn("--split", command)
        self.assertEqual(command[command.index("--split") + 1], "confirmation")
        self.assertEqual(command[command.index("--dtype") + 1], "bf16")
        self.assertEqual(command[command.index("--batch-size") + 1], "1")
        self.assertEqual(
            command[command.index("--wandb-run-name") + 1],
            "exp2-step2000-k2-conditional-followup")


if __name__ == "__main__":
    unittest.main()
