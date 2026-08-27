"""CPU correctness tests for Experiment 3 routing primitives."""

import unittest

import torch

from src.attention_residuals.mhar_partition import (
    canonicalize_contiguous_partition,
    contiguous_mhar_eager,
    contiguous_partition_from_boundaries,
    mixed_partition_from_merges,
)
from src.attention_residuals.modeling_qwen3_attnres import (
    Qwen3AttnResConfig,
    Qwen3AttnResForCausalLM,
)
from src.experiments.experiment3_common import (
    h16_boundary_candidates,
    h8_boundary_move_candidates,
)


class IdentityNorm(torch.nn.Module):
    def forward(self, value):
        return value


class Experiment3RouterTest(unittest.TestCase):
    def test_contiguous_partition_validation(self):
        partition = contiguous_partition_from_boundaries(
            (4, 9), hidden_size=12, num_groups=3, min_width=3)
        self.assertEqual(partition, ((0, 4), (4, 9), (9, 12)))
        with self.assertRaisesRegex(ValueError, "no gaps"):
            canonicalize_contiguous_partition(
                ((0, 4), (5, 12)), hidden_size=12)
        with self.assertRaisesRegex(ValueError, "narrower"):
            contiguous_partition_from_boundaries(
                (2, 9), hidden_size=12, min_width=3)

    def test_uniform_contiguous_router_matches_equal_width_equation(self):
        torch.manual_seed(7)
        V = torch.randn(5, 2, 3, 12, dtype=torch.float64)
        query = torch.randn(12, dtype=torch.float64)
        partition = contiguous_partition_from_boundaries(
            (3, 6, 9), hidden_size=12, num_groups=4)
        actual = contiguous_mhar_eager(
            V, query, IdentityNorm(), partition, num_groups=4)
        K = V.view(5, 2, 3, 4, 3)
        logits = torch.einsum("h k, n b t h k -> n b t h", query.view(4, 3), K)
        weights = logits.softmax(dim=0)
        expected = torch.einsum(
            "n b t h, n b t h k -> b t h k", weights, K).reshape(2, 3, 12)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_irregular_contiguous_router_has_gradients(self):
        torch.manual_seed(8)
        V = torch.randn(4, 1, 2, 12, requires_grad=True)
        query = torch.randn(12, requires_grad=True)
        partition = ((0, 2), (2, 7), (7, 12))
        output = contiguous_mhar_eager(
            V, query, IdentityNorm(), partition, num_groups=3)
        output.square().mean().backward()
        self.assertEqual(output.shape, (1, 2, 12))
        self.assertTrue(torch.isfinite(V.grad).all())
        self.assertTrue(torch.isfinite(query.grad).all())

    def test_candidate_spaces_are_frozen(self):
        h16 = h16_boundary_candidates()
        self.assertEqual(len(h16), 16)
        self.assertEqual([row["boundary"] for row in h16[1:]], list(range(15)))
        self.assertTrue(all(len(mixed_partition_from_merges((i,))) == 15 for i in range(15)))
        h8 = h8_boundary_move_candidates()
        self.assertEqual(len(h8), 57)
        self.assertEqual(sum(row["candidate_id"] == "native-h8" for row in h8), 1)

    def test_model_setter_propagates_and_is_mutually_exclusive(self):
        config = Qwen3AttnResConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=32,
            max_position_embeddings=16,
            head_dim=4,
            attnres_mode="full_mh",
            attnres_num_heads=4,
            tie_word_embeddings=True,
        )
        model = Qwen3AttnResForCausalLM(config)
        partition = ((0, 3), (3, 8), (8, 12), (12, 16))
        model.set_mhar_contiguous_partition(partition)
        self.assertEqual(model.mhar_contiguous_partition, partition)
        self.assertTrue(all(
            layer._mhar_contiguous_partition == partition
            for layer in model.model.layers))
        model.set_mhar_mixed_partition(
            ((0,), (1,), (2,), (3,)))
        self.assertIsNone(model.mhar_contiguous_partition)


if __name__ == "__main__":
    unittest.main()
