import unittest

import torch

from src.experiments.experiment10_per_group_contribution import (
    Experiment10Intervention,
    phase_10abc_conditions,
    phase_10d_conditions,
)
from src.experiments.experiment8_hybrid_q import HybridQueryLinear


class Experiment10InterventionTest(unittest.TestCase):
    def test_primary_conditions_are_complete_and_matched(self):
        conditions = phase_10abc_conditions()
        self.assertEqual(len(conditions), 25)
        self.assertEqual(sum(row["kind"] == "reference" for row in conditions), 1)
        self.assertEqual(sum(row["kind"] == "single-local" for row in conditions), 8)
        self.assertEqual(sum(row["kind"] == "single-global" for row in conditions), 8)
        self.assertEqual(sum(row["kind"] == "whole-group" for row in conditions), 8)
        for group in range(8):
            by_id = {row["id"]: row for row in conditions}
            self.assertEqual(by_id[f"remove-local-g{group}"]["removed_heads"], [2 * group])
            self.assertEqual(by_id[f"remove-global-g{group}"]["removed_heads"], [2 * group + 1])
            self.assertEqual(by_id[f"remove-group-g{group}"]["removed_heads"], [2 * group, 2 * group + 1])

    def test_alignment_conditions_exhaust_one_group_substitutions(self):
        conditions = phase_10d_conditions()
        self.assertEqual(len(conditions), 56)
        self.assertEqual(len({row["id"] for row in conditions}), 56)
        for row in conditions:
            sources = row["local_chunk_sources"]
            changed = [index for index, source in enumerate(sources) if index != source]
            self.assertEqual(changed, [row["target_group"]])
            self.assertEqual(sources[row["target_group"]], row["source_chunk"])

    def test_single_group_source_map_changes_only_selected_local_output(self):
        projection = HybridQueryLinear(16, 16)
        with torch.no_grad():
            projection.local_weight.fill_(1)
            projection.global_weight.fill_(1)
        inputs = torch.arange(16.0).reshape(1, 16)
        aligned = projection(inputs).reshape(1, 8, 2)
        projection.local_group_permutation = (1, 1, 2, 3, 4, 5, 6, 7)
        changed = projection(inputs).reshape(1, 8, 2)
        torch.testing.assert_close(aligned[..., 1], changed[..., 1])
        self.assertFalse(torch.equal(aligned[..., 0], changed[..., 0]))
        self.assertFalse(torch.equal(aligned[..., 0, 0], changed[..., 0, 0]))
        torch.testing.assert_close(aligned[..., 1:, 0], changed[..., 1:, 0])

    def test_invalid_source_maps_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            Experiment10Intervention(None, local_chunk_sources=range(8))
        with self.assertRaisesRegex(ValueError, "eight"):
            Experiment10Intervention(None, local_chunk_sources=[1, 0])
        with self.assertRaisesRegex(ValueError, "outside"):
            Experiment10Intervention(None, local_chunk_sources=[8, 1, 2, 3, 4, 5, 6, 7])


if __name__ == "__main__":
    unittest.main()
