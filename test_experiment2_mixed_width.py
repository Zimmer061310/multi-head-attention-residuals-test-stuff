"""Correctness tests for Experiment 2 mixed-width MHAR routing."""

import math
import os
import sys
import unittest

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "Attention-Residuals"))

from mhar_partition import (  # noqa: E402
    canonicalize_mixed_partition,
    generate_adjacent_merge_partitions,
    merged_boundaries,
    mixed_partition_from_merges,
    mixed_partition_id,
    mixed_segment_widths,
    mixed_width_mhar_eager,
    parse_mixed_partition_id,
)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.square().mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)


def ordinary_mhar_eager(V, query, norm, num_heads):
    K = norm(V)
    n, b, t, d = V.shape
    width = d // num_heads
    logits = torch.einsum(
        "h k, n b t h k -> n b t h",
        query.view(num_heads, width),
        K.view(n, b, t, num_heads, width),
    )
    weights = logits.softmax(dim=0)
    return torch.einsum(
        "n b t h, n b t h k -> b t h k",
        weights,
        V.view(n, b, t, num_heads, width),
    ).reshape(b, t, d)


class MixedPartitionEnumerationTest(unittest.TestCase):
    def test_primary_space_contains_exactly_495_partitions(self):
        partitions = generate_adjacent_merge_partitions(16, 4)
        self.assertEqual(len(partitions), 495)
        self.assertEqual(len(set(partitions)), 495)
        for partition in partitions:
            self.assertEqual(len(partition), 12)
            self.assertEqual(len(merged_boundaries(partition)), 4)
            self.assertEqual(mixed_segment_widths(partition).count(160), 4)
            self.assertEqual(mixed_segment_widths(partition).count(80), 8)

    def test_extended_space_counts(self):
        counts = [len(generate_adjacent_merge_partitions(16, k)) for k in range(9)]
        expected = [math.comb(16 - k, k) for k in range(9)]
        self.assertEqual(counts, expected)
        self.assertEqual(sum(counts), 1597)

    def test_id_round_trip(self):
        for partition in generate_adjacent_merge_partitions(16, 4):
            encoded = mixed_partition_id(partition)
            self.assertEqual(parse_mixed_partition_id(encoded), partition)

    def test_invalid_or_overlapping_groups_are_rejected(self):
        with self.assertRaises(ValueError):
            mixed_partition_from_merges((0, 1), num_atomic_blocks=16)
        with self.assertRaises(ValueError):
            canonicalize_mixed_partition(((0, 2),), num_atomic_blocks=2)
        with self.assertRaises(ValueError):
            canonicalize_mixed_partition(((1,), (0,)), num_atomic_blocks=2)


class MixedWidthKernelTest(unittest.TestCase):
    @staticmethod
    def _inputs(seed=0):
        torch.manual_seed(seed)
        V = torch.randn(5, 2, 7, 64)
        query = torch.randn(64)
        norm = RMSNorm(64)
        with torch.no_grad():
            norm.weight.uniform_(0.8, 1.2)
        return V, query, norm

    def test_pure_h16_forward_parity(self):
        V, query, norm = self._inputs()
        partition = mixed_partition_from_merges((), num_atomic_blocks=16)
        expected = ordinary_mhar_eager(V, query, norm, 16)
        observed = mixed_width_mhar_eager(
            V, query, norm, partition, num_atomic_blocks=16)
        torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)

    def test_pure_h8_forward_parity(self):
        V, query, norm = self._inputs()
        partition = mixed_partition_from_merges(range(0, 16, 2), num_atomic_blocks=16)
        expected = ordinary_mhar_eager(V, query, norm, 8)
        observed = mixed_width_mhar_eager(
            V, query, norm, partition, num_atomic_blocks=16)
        torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)

    def test_mixed_case_matches_explicit_segment_reference(self):
        V, query, norm = self._inputs(seed=3)
        partition = mixed_partition_from_merges((0, 4, 8, 12), num_atomic_blocks=16)
        K = norm(V)
        pieces = []
        width = 4
        for group in partition:
            start, end = group[0] * width, (group[-1] + 1) * width
            logits = (K[..., start:end] * query[start:end]).sum(-1)
            weights = logits.softmax(dim=0)
            pieces.append((weights[..., None] * V[..., start:end]).sum(0))
        expected = torch.cat(pieces, dim=-1)
        observed = mixed_width_mhar_eager(
            V, query, norm, partition, num_atomic_blocks=16)
        torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)

    def test_h16_gradient_parity(self):
        V, query, norm = self._inputs(seed=7)
        V_a = V.detach().clone().requires_grad_(True)
        V_b = V.detach().clone().requires_grad_(True)
        q_a = query.detach().clone().requires_grad_(True)
        q_b = query.detach().clone().requires_grad_(True)
        norm_b = RMSNorm(64)
        norm_b.load_state_dict(norm.state_dict())
        partition = mixed_partition_from_merges((), num_atomic_blocks=16)

        ordinary_mhar_eager(V_a, q_a, norm, 16).square().mean().backward()
        mixed_width_mhar_eager(
            V_b, q_b, norm_b, partition, num_atomic_blocks=16,
        ).square().mean().backward()

        torch.testing.assert_close(V_b.grad, V_a.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(q_b.grad, q_a.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            norm_b.weight.grad, norm.weight.grad, rtol=1e-5, atol=1e-6)


class MixedWidthModelIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MHAR_RUN_MODEL_INTEGRATION") == "1",
        "set MHAR_RUN_MODEL_INTEGRATION=1 in the pinned Transformers runtime",
    )
    def test_native_h16_complete_model_parity_and_global_application(self):
        from modeling_qwen3_attnres import Qwen3AttnResConfig, Qwen3AttnResForCausalLM

        torch.manual_seed(13)
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
            attnres_num_heads=16,
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
            native = model(input_ids=input_ids, use_cache=False).logits
            h16 = mixed_partition_from_merges((), num_atomic_blocks=16)
            model.set_mhar_mixed_partition(h16)
            parity = model(input_ids=input_ids, use_cache=False).logits
        torch.testing.assert_close(parity, native, rtol=1e-5, atol=1e-6)
        self.assertTrue(all(layer._mhar_mixed_partition == h16 for layer in model.model.layers))

        mixed = mixed_partition_from_merges((0, 4, 8, 12), num_atomic_blocks=16)
        model.set_mhar_mixed_partition(mixed)
        with torch.no_grad():
            changed = model(input_ids=input_ids, use_cache=False).logits
        self.assertGreater((changed - native).abs().max().item(), 1e-7)

        model.set_mhar_mixed_partition(None)
        self.assertIsNone(model.mhar_mixed_partition)


if __name__ == "__main__":
    unittest.main()
