"""Synthetic selection, branch-safety, and gate tests for Experiment 3C."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.experiment3_actionability import (
    actionability_analysis,
    build_branch_selection,
)
from src.experiments.experiment3_common import h16_boundary_candidates
from src.training.train_scratch import (
    BRANCH_SCIENTIFIC_IDENTITY_KEYS,
    sha256_tree,
    validate_branch_invocation,
    validate_branch_parent_identity,
)


def discovery_rows():
    rows = {
        "native-h16": {
            "candidate_id": "native-h16",
            "nll": 5.0,
            "partition_id": None,
            "boundary": None,
        }
    }
    for row in h16_boundary_candidates()[1:]:
        rows[row["candidate_id"]] = {
            **row,
            "nll": 5.0 + row["boundary"] * 0.01,
        }
    return rows


def branch_result(role, value):
    return {
        "role": role,
        "seed": 42,
        "artifact_sha256": "artifact",
        "splits": {
            split: {
                "nll": value,
                "sequence_nlls": [value] * 32,
            }
            for split in ("discovery", "confirmation")
        },
    }


class Experiment3ActionabilityTest(unittest.TestCase):
    def test_selection_is_within_seed_and_random_is_middle_ranked(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            (parent / "weights.bin").write_bytes(b"checkpoint")
            manifest = build_branch_selection(
                discovery_rows(),
                parent_checkpoint=parent,
                signal_summary={"signal_gate_passed": True, "step": 1500},
                temporal_summary={"stability_gate_passed": True},
                random_seed=123,
            )
        self.assertEqual(
            manifest["branches"]["predicted-good"]["candidate_id"], "remove-00")
        self.assertEqual(
            manifest["branches"]["predicted-bad"]["candidate_id"], "remove-14")
        self.assertIn(
            manifest["branches"]["random"]["candidate_id"],
            manifest["middle_rank_pool"])
        self.assertIsNone(manifest["branches"]["unchanged"]["partition_id"])

    def test_selection_refuses_failed_upstream_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "signal gate"):
                build_branch_selection(
                    discovery_rows(),
                    parent_checkpoint=Path(directory),
                    signal_summary={"signal_gate_passed": False},
                    temporal_summary={"stability_gate_passed": True},
                    random_seed=1,
                )

    def test_actionability_gate_uses_good_minus_random_confirmation(self):
        results = {
            "predicted-good": branch_result("predicted-good", 4.0),
            "random": branch_result("random", 4.1),
            "predicted-bad": branch_result("predicted-bad", 4.2),
            "unchanged": branch_result("unchanged", 4.15),
        }
        summary = actionability_analysis(results, samples=200, seed=5)
        self.assertTrue(summary["actionability_gate_passed"])
        self.assertTrue(summary["strong_actionability_result"])

    def test_branch_manifest_and_parent_identity_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent"
            parent.mkdir()
            (parent / "weights.bin").write_bytes(b"checkpoint")
            partition = h16_boundary_candidates()[1]["partition_id"]
            selection = root / "selection.json"
            selection.write_text(json.dumps({
                "parent_checkpoint_sha256": sha256_tree(parent),
                "branches": {
                    "predicted-good": {"partition_id": partition},
                },
            }), encoding="utf-8")
            args = argparse.Namespace(
                branch_from=str(parent),
                branch_manifest=str(selection),
                branch_role="predicted-good",
                mixed_partition=partition,
            )
            validate_branch_invocation(args, verify_parent_hash=True)
            args.mixed_partition = None
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_branch_invocation(args)

        parent_identity = {key: "same" for key in BRANCH_SCIENTIFIC_IDENTITY_KEYS}
        parent_identity["mixed_partition"] = None
        current_identity = dict(parent_identity)
        current_identity["mixed_partition"] = "allowed-change"
        validate_branch_parent_identity(parent_identity, current_identity)
        current_identity["seed"] = "different"
        with self.assertRaisesRegex(RuntimeError, "scientific identity"):
            validate_branch_parent_identity(parent_identity, current_identity)


if __name__ == "__main__":
    unittest.main()
