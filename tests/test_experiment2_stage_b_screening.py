"""Validation for the frozen Experiment 2 Stage B screening matrix."""

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from src.attention_residuals.mhar_partition import parse_mixed_partition_id


class StageBScreeningTest(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(
            (ROOT / "configs/experiment2/stage-b-screening.json").read_text(
                encoding="utf-8"))
        self.runs = {row["id"]: row for row in self.manifest["runs"]}

    def test_exact_seven_run_matrix(self):
        self.assertEqual(
            set(self.runs),
            {
                "h16", "h8", "mixed-k2", "mixed-k3", "mixed-k4-best",
                "mixed-k5", "mixed-k4-worst",
            },
        )
        self.assertEqual(len(self.runs), 7)
        self.assertEqual(self.manifest["protected_checkpoints"], [2000, 5000, 10000, 20000])

    def test_mixed_partitions_match_boundaries_and_group_counts(self):
        for row in self.manifest["runs"]:
            if not row["id"].startswith("mixed-"):
                continue
            partition = parse_mixed_partition_id(row["partition_id"], num_atomic_blocks=16)
            observed_boundaries = [group[0] for group in partition if len(group) == 2]
            self.assertEqual(observed_boundaries, row["merged_boundaries"])
            self.assertEqual(len(partition), row["routing_groups"])

    def test_best_and_worst_k4_have_identical_width_composition(self):
        best = parse_mixed_partition_id(self.runs["mixed-k4-best"]["partition_id"])
        worst = parse_mixed_partition_id(self.runs["mixed-k4-worst"]["partition_id"])
        self.assertEqual(sorted(map(len, best)), sorted(map(len, worst)))
        self.assertEqual(sorted(map(len, best)), [1] * 8 + [2] * 4)

    def test_selection_source_is_content_addressed(self):
        source = ROOT / self.manifest["selection_provenance"]["discovery_results"]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            self.manifest["selection_provenance"]["discovery_results_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
