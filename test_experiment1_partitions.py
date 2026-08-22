"""CPU correctness tests for Experiment 1 partition routing."""

import os
import sys
import unittest
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Attention-Residuals"))

import torch
import torch.nn as nn

from mhar_partition import (
    REFERENCE_PARTITION_H4,
    arbitrary_group_mhar_eager,
    canonicalize_partition,
    coordinate_distance,
    generate_pair_partitions,
    original_pair_retention,
    parse_partition_id,
    partition_id,
)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.square().mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def ordinary_mhar_eager(V, query, norm, num_heads):
    K = norm(V)
    n, b, t, d = V.shape
    q = query.view(num_heads, -1)
    dh = d // num_heads
    logits = torch.einsum("h k, n b t h k -> n b t h", q, K.view(n, b, t, num_heads, dh))
    weights = logits.softmax(dim=0)
    output = torch.einsum(
        "n b t h, n b t h k -> b t h k", weights, V.view(n, b, t, num_heads, dh))
    return output.reshape(b, t, d)


class PartitionEnumerationTest(unittest.TestCase):
    def test_all_105_partitions_and_preregistered_counts(self):
        partitions = generate_pair_partitions(8)
        self.assertEqual(len(partitions), 105)
        self.assertEqual(len(set(partitions)), 105)

        retention_counts = Counter(original_pair_retention(p)[0] for p in partitions)
        self.assertEqual(retention_counts, {4: 1, 2: 12, 1: 32, 0: 60})

        distance_counts = Counter(coordinate_distance(p)[0] for p in partitions)
        self.assertEqual(distance_counts, {4: 1, 6: 6, 8: 12, 10: 20,
                                           12: 24, 14: 18, 16: 24})

    def test_partition_id_round_trip(self):
        for partition in generate_pair_partitions(8):
            self.assertEqual(parse_partition_id(partition_id(partition)), partition)

    def test_invalid_partitions_are_rejected(self):
        with self.assertRaises(ValueError):
            canonicalize_partition(((0, 1), (1, 2)), num_primitive_blocks=4)
        with self.assertRaises(ValueError):
            canonicalize_partition(((0, 0), (1, 2)), num_primitive_blocks=4)


class ArbitraryRoutingKernelTest(unittest.TestCase):
    def _modules_and_inputs(self, *, seed=0):
        torch.manual_seed(seed)
        d, sources, batch, tokens = 64, 5, 2, 7
        blocks = [torch.randn(batch, tokens, d) for _ in range(sources - 1)]
        partial = torch.randn(batch, tokens, d)
        proj = nn.Linear(d, 1, bias=False)
        norm = RMSNorm(d, eps=1e-6)
        with torch.no_grad():
            proj.weight.normal_(0, 0.25)
            norm.weight.uniform_(0.8, 1.2)
        return blocks, partial, proj, norm

    def test_reference_partition_matches_ordinary_h4(self):
        blocks, partial, proj, norm = self._modules_and_inputs()
        V = torch.stack(blocks + [partial], dim=0)
        ordinary = ordinary_mhar_eager(V, proj.weight.view(-1), norm, 4)
        arbitrary = arbitrary_group_mhar_eager(
            V, proj.weight.view(-1), norm, REFERENCE_PARTITION_H4, 4)
        torch.testing.assert_close(arbitrary, ordinary, rtol=1e-6, atol=1e-6)

    def test_partition_representation_order_does_not_matter(self):
        blocks, partial, proj, norm = self._modules_and_inputs()
        partition = ((0, 5), (1, 7), (2, 4), (3, 6))
        reordered = ((6, 3), (4, 2), (7, 1), (5, 0))
        V = torch.stack(blocks + [partial], dim=0)
        first = arbitrary_group_mhar_eager(V, proj.weight.view(-1), norm, partition, 4)
        second = arbitrary_group_mhar_eager(V, proj.weight.view(-1), norm, reordered, 4)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_reference_partition_gradient_parity(self):
        blocks_a, partial_a, proj_a, norm_a = self._modules_and_inputs(seed=7)
        blocks_b = [value.detach().clone().requires_grad_(True) for value in blocks_a]
        partial_b = partial_a.detach().clone().requires_grad_(True)
        blocks_a = [value.requires_grad_(True) for value in blocks_a]
        partial_a.requires_grad_(True)
        proj_b = nn.Linear(64, 1, bias=False)
        norm_b = RMSNorm(64, eps=1e-6)
        proj_b.load_state_dict(proj_a.state_dict())
        norm_b.load_state_dict(norm_a.state_dict())

        V_a = torch.stack(blocks_a + [partial_a], dim=0)
        V_b = torch.stack(blocks_b + [partial_b], dim=0)
        ordinary = ordinary_mhar_eager(V_a, proj_a.weight.view(-1), norm_a, 4)
        arbitrary = arbitrary_group_mhar_eager(
            V_b, proj_b.weight.view(-1), norm_b, REFERENCE_PARTITION_H4, 4)
        ordinary.square().mean().backward()
        arbitrary.square().mean().backward()

        torch.testing.assert_close(proj_b.weight.grad, proj_a.weight.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(norm_b.weight.grad, norm_a.weight.grad, rtol=1e-5, atol=1e-6)
        for got, expected in zip(blocks_b + [partial_b], blocks_a + [partial_a]):
            torch.testing.assert_close(got.grad, expected.grad, rtol=1e-5, atol=1e-6)


class ModelIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MHAR_RUN_MODEL_INTEGRATION") == "1",
        "set MHAR_RUN_MODEL_INTEGRATION=1 in the pinned Transformers runtime",
    )
    def test_reference_partition_matches_complete_model(self):
        from modeling_qwen3_attnres import Qwen3AttnResConfig, Qwen3AttnResForCausalLM

        torch.manual_seed(11)
        config = Qwen3AttnResConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=32,
            head_dim=16,
            attnres_mode="full_mh",
            attnres_num_heads=4,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,
        )
        model = Qwen3AttnResForCausalLM(config).eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "res_proj.weight" in name:
                    parameter.normal_(0, 0.2)

        input_ids = torch.randint(0, config.vocab_size, (2, 11))
        with torch.no_grad():
            ordinary = model(input_ids=input_ids, use_cache=False).logits
            canonical = model.set_mhar_partition(REFERENCE_PARTITION_H4)
            arbitrary = model(input_ids=input_ids, use_cache=False).logits

        self.assertEqual(canonical, REFERENCE_PARTITION_H4)
        self.assertEqual(model.mhar_partition, REFERENCE_PARTITION_H4)
        self.assertTrue(all(
            layer._mhar_partition == REFERENCE_PARTITION_H4
            for layer in model.model.layers))
        torch.testing.assert_close(arbitrary, ordinary, rtol=1e-5, atol=1e-6)

        alternative_partition = ((0, 7), (1, 6), (2, 5), (3, 4))
        model.set_mhar_partition(alternative_partition)
        with torch.no_grad():
            alternative = model(input_ids=input_ids, use_cache=False).logits
        self.assertGreater((alternative - ordinary).abs().max().item(), 1e-7)

        model.set_mhar_partition(None)
        self.assertIsNone(model.mhar_partition)
        self.assertTrue(all(layer._mhar_partition is None for layer in model.model.layers))


if __name__ == "__main__":
    unittest.main()
