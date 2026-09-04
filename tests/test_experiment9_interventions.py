import unittest

import torch
from torch import nn

from src.experiments.experiment8_hybrid_q import HybridQueryLinear
from src.experiments.experiment9_head_contribution import (
    GLOBAL_HEADS,
    LOCAL_HEADS,
    mask_head_outputs,
    phase_9a_conditions,
    phase_9b_conditions,
)


class Experiment9InterventionTest(unittest.TestCase):
    def test_local_and_global_masks_zero_exact_head_slices(self):
        values = torch.arange(32.0).reshape(1, 16, 2)
        flat = values.reshape(1, 32)
        zero_local = mask_head_outputs(flat, LOCAL_HEADS, num_heads=16, head_dim=2)
        zero_global = mask_head_outputs(flat, GLOBAL_HEADS, num_heads=16, head_dim=2)
        local_view = zero_local.reshape(16, 2)
        global_view = zero_global.reshape(16, 2)
        torch.testing.assert_close(local_view[list(LOCAL_HEADS)], torch.zeros(8, 2))
        torch.testing.assert_close(global_view[list(GLOBAL_HEADS)], torch.zeros(8, 2))
        torch.testing.assert_close(local_view[list(GLOBAL_HEADS)], values[0, list(GLOBAL_HEADS)])
        torch.testing.assert_close(global_view[list(LOCAL_HEADS)], values[0, list(LOCAL_HEADS)])

    def test_balanced_family_is_exhaustive_and_matched(self):
        conditions = phase_9a_conditions()
        balanced = [row for row in conditions if row["kind"] == "balanced-random-8"]
        self.assertEqual(len(conditions), 73)
        self.assertEqual(len(balanced), 70)
        self.assertEqual(len({tuple(row["removed_heads"]) for row in balanced}), 70)
        for row in balanced:
            removed = row["removed_heads"]
            self.assertEqual(len(removed), 8)
            self.assertEqual(sum(head % 2 == 0 for head in removed), 4)
            self.assertEqual(sum(head % 2 == 1 for head in removed), 4)
            for group in range(8):
                self.assertEqual(sum(head in removed for head in (2 * group, 2 * group + 1)), 1)

    def test_frozen_derangements_are_unique_and_complete(self):
        conditions = phase_9b_conditions()
        permutations = [tuple(row["local_chunk_permutation"]) for row in conditions]
        self.assertEqual(len(permutations), 32)
        self.assertEqual(len(set(permutations)), 32)
        for permutation in permutations:
            self.assertEqual(sorted(permutation), list(range(8)))
            self.assertTrue(all(source != target for target, source in enumerate(permutation)))

    def test_local_projection_permutation_changes_only_local_rows(self):
        projection = HybridQueryLinear(16, 16)
        with torch.no_grad():
            projection.local_weight.fill_(1)
            projection.global_weight.fill_(1)
        inputs = torch.arange(16.0).reshape(1, 16)
        aligned = projection(inputs).reshape(1, 8, 2)
        projection.local_group_permutation = (1, 2, 3, 4, 5, 6, 7, 0)
        permuted = projection(inputs).reshape(1, 8, 2)
        self.assertTrue(torch.equal(aligned[..., 1], permuted[..., 1]))
        self.assertFalse(torch.equal(aligned[..., 0], permuted[..., 0]))

    def test_default_hybrid_projection_has_no_runtime_intervention_state(self):
        projection = HybridQueryLinear(16, 16)
        self.assertIsNone(projection.local_group_permutation)
        self.assertNotIn("local_group_permutation", projection.state_dict())

    def test_bad_masks_fail_closed(self):
        values = torch.ones(1, 32)
        with self.assertRaisesRegex(ValueError, "unique"):
            mask_head_outputs(values, [0, 0], num_heads=16, head_dim=2)
        with self.assertRaisesRegex(ValueError, "range"):
            mask_head_outputs(values, [16], num_heads=16, head_dim=2)


if __name__ == "__main__":
    unittest.main()
