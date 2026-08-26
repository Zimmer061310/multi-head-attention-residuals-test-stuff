"""CPU tests for Experiment 2 selection, analysis, and paired uncertainty."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.experiment2_mixed_width import (
    NATIVE_H16_ID,
    analyze_command,
    build_partition_choice_rows,
    paired_bootstrap,
)
from src.attention_residuals.mhar_partition import (
    generate_adjacent_merge_partitions,
    merged_boundaries,
    mixed_partition_id,
    mixed_segment_widths,
)


def write_synthetic_results(path: Path):
    rows = [{
        "partition_id": NATIVE_H16_ID,
        "nll": 2.8,
        "ppl": 16.44,
        "elapsed_seconds": 1.0,
        "tokens_per_second": 100.0,
        "merged_boundaries": [],
        "segment_widths": [80] * 16,
    }]
    for index, partition in enumerate(generate_adjacent_merge_partitions(16, 4)):
        nll = 2.7 + index / 100000
        rows.append({
            "partition_id": mixed_partition_id(partition),
            "nll": nll,
            "ppl": 16.0,
            "elapsed_seconds": 1.0,
            "tokens_per_second": 100.0,
            "merged_boundaries": list(merged_boundaries(partition)),
            "segment_widths": list(mixed_segment_widths(partition)),
        })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class PairedBootstrapTest(unittest.TestCase):
    def test_constant_paired_delta_has_degenerate_interval(self):
        result = paired_bootstrap([1.1, 1.2, 1.3], [1.0, 1.1, 1.2], samples=1000)
        self.assertAlmostEqual(result["mean_delta_nll"], 0.1)
        self.assertAlmostEqual(result["ci95_low"], 0.1)
        self.assertAlmostEqual(result["ci95_high"], 0.1)


class AnalysisTest(unittest.TestCase):
    def test_partition_choice_rows_include_native_and_all_495_mixed_choices(self):
        native = {
            "partition_id": NATIVE_H16_ID,
            "nll": 2.8,
            "merged_boundaries": [],
            "segment_widths": [80] * 16,
            "routing_groups": 16,
        }
        ranked = []
        for rank, partition in enumerate(generate_adjacent_merge_partitions(16, 4), 1):
            ranked.append({
                "rank": rank,
                "partition_id": mixed_partition_id(partition),
                "nll": 2.8 + rank / 1000,
                "delta_nll_vs_native_h16": rank / 1000,
                "merged_boundaries": list(merged_boundaries(partition)),
                "segment_widths": list(mixed_segment_widths(partition)),
                "routing_groups": 12,
            })

        rows = build_partition_choice_rows(native, ranked)

        self.assertEqual(len(rows), 496)
        self.assertEqual(rows[0]["choice_rank"], 1)
        self.assertEqual(rows[0]["partition_id"], NATIVE_H16_ID)
        self.assertEqual(rows[0]["delta_nll_vs_native_h16"], 0.0)
        self.assertEqual(rows[-1]["choice_rank"], 496)
        self.assertTrue(all(len(row["merged_boundaries"]) == 4 for row in rows[1:]))

    def test_analysis_freezes_top5_median_worst_and_native(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            discovery = root / "discovery.jsonl"
            output = root / "analysis"
            write_synthetic_results(discovery)
            args = argparse.Namespace(
                discovery_results=str(discovery),
                confirmation_results=None,
                output_dir=str(output),
                allow_incomplete=False,
                top_count=50,
                bootstrap_samples=1000,
            )
            analyze_command(args)
            selection = json.loads((output / "confirmation_selection.json").read_text())
            roles = {role for row in selection["candidates"] for role in row["roles"]}
            self.assertEqual(
                roles,
                {
                    "native_h16", "discovery_top_1", "discovery_top_2",
                    "discovery_top_3", "discovery_top_4", "discovery_top_5",
                    "discovery_median", "discovery_worst",
                },
            )
            self.assertEqual(len(selection["candidates"]), 8)
            self.assertTrue((output / "fig_partition_ranking.png").is_file())
            self.assertTrue((output / "fig_partition_choice_map.png").is_file())
            self.assertTrue((output / "fig_partition_choice_map.pdf").is_file())
            with (output / "partition_choice_map.csv").open(encoding="utf-8") as handle:
                self.assertEqual(sum(1 for _ in handle), 497)
            self.assertTrue((output / "fig_boundary_associations.pdf").is_file())
            # The immutable selection can be reproduced without changing it.
            first = (output / "confirmation_selection.json").read_bytes()
            analyze_command(args)
            self.assertEqual(first, (output / "confirmation_selection.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
