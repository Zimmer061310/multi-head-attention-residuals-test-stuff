"""Correctness and isolation tests for Experiment 6 grouped Q/K/V."""

import tempfile
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from src.attention_residuals.modeling_qwen3_attnres import Qwen3AttnResConfig
from src.experiments.experiment6_coupled_qkv import (
    Experiment6BaselineForCausalLM,
    Experiment6MHARForCausalLM,
    GroupedLinear,
    experiment6_parameter_report,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


def common_config(*, groups=4, layers=1):
    heads = 8 if groups == 4 else 16
    kv_heads = 4 if groups == 4 else 8
    return dict(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        intermediate_size=64,
        max_position_embeddings=32,
        head_dim=32 // heads,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
    )


class Experiment6GroupedQKVTest(unittest.TestCase):
    def test_grouped_linear_equals_explicit_block_mask(self):
        torch.manual_seed(7)
        for groups in (4, 8):
            dense = nn.Linear(32, 32, bias=True)
            grouped = GroupedLinear.from_dense(dense, groups)
            inputs = torch.randn(2, 3, 32)
            expected = F.linear(
                inputs,
                grouped.dense_weight(),
                grouped.bias.reshape(-1),
            )
            torch.testing.assert_close(grouped(inputs), expected)

    def test_disallowed_cross_group_input_has_no_effect(self):
        dense = nn.Linear(32, 16, bias=False)
        grouped = GroupedLinear.from_dense(dense, 4)
        inputs = torch.zeros(1, 32)
        inputs[0, 8] = 1.0  # input group 1
        outputs = grouped(inputs).reshape(1, 4, 4)
        self.assertEqual(torch.count_nonzero(outputs[:, 0]).item(), 0)
        self.assertEqual(torch.count_nonzero(outputs[:, 2:]).item(), 0)

    def test_g4_copies_matched_dense_initialization_and_preserves_wo(self):
        kwargs = common_config(groups=4)
        torch.manual_seed(42)
        dense = Qwen3ForCausalLM(Qwen3Config(**kwargs))
        grouped_config = Qwen3Config(**kwargs)
        grouped_config.experiment6_qkv_groups = 4
        torch.manual_seed(42)
        grouped = Experiment6BaselineForCausalLM(grouped_config)

        dense_attn = dense.model.layers[0].self_attn
        grouped_attn = grouped.model.layers[0].self_attn
        torch.testing.assert_close(dense_attn.o_proj.weight, grouped_attn.o_proj.weight)
        for name in ("q_proj", "k_proj", "v_proj"):
            expected = GroupedLinear.from_dense(getattr(dense_attn, name), 4)
            torch.testing.assert_close(getattr(grouped_attn, name).weight, expected.weight)
        torch.testing.assert_close(dense.model.embed_tokens.weight, grouped.model.embed_tokens.weight)

    def test_c8_copies_matched_m8_initialization_and_preserves_routing(self):
        kwargs = common_config(groups=8)
        dense_config = Qwen3AttnResConfig(
            attnres_mode="full_mh", attnres_num_heads=8, **kwargs)
        coupled_config = Qwen3AttnResConfig(
            attnres_mode="full_mh", attnres_num_heads=8, **kwargs)
        coupled_config.experiment6_qkv_groups = 8

        from src.attention_residuals.modeling_qwen3_attnres import Qwen3AttnResForCausalLM
        torch.manual_seed(123)
        dense = Qwen3AttnResForCausalLM(dense_config)
        torch.manual_seed(123)
        coupled = Experiment6MHARForCausalLM(coupled_config)
        dense_layer = dense.model.layers[0]
        coupled_layer = coupled.model.layers[0]

        torch.testing.assert_close(
            dense_layer.attn_res_proj.weight, coupled_layer.attn_res_proj.weight)
        torch.testing.assert_close(
            dense_layer.mlp_res_proj.weight, coupled_layer.mlp_res_proj.weight)
        torch.testing.assert_close(
            dense_layer.self_attn.o_proj.weight, coupled_layer.self_attn.o_proj.weight)
        for name in ("q_proj", "k_proj", "v_proj"):
            expected = GroupedLinear.from_dense(getattr(dense_layer.self_attn, name), 8)
            torch.testing.assert_close(
                getattr(coupled_layer.self_attn, name).weight, expected.weight)

    def test_parameter_report_exposes_capacity_change_and_dense_wo(self):
        config = Qwen3Config(**common_config(groups=4, layers=2))
        config.experiment6_qkv_groups = 4
        model = Experiment6BaselineForCausalLM(config)
        report = experiment6_parameter_report(model)
        self.assertEqual(report["qkv_groups"], 4)
        dense_qkv_per_layer = 32 * 32 + 2 * (16 * 32)
        self.assertEqual(report["qkv_parameters"], 2 * dense_qkv_per_layer // 4)
        self.assertEqual(report["attention_output_parameters"], 2 * 32 * 32)

    def test_real_1b_qkv_parameter_counts_are_explicit(self):
        dense_per_layer = 1280 * 1280 + 2 * (640 * 1280)
        self.assertEqual(dense_per_layer, 3_276_800)
        for groups, expected in ((4, 819_200), (8, 409_600)):
            projections = [
                GroupedLinear(1280, 1280, groups, bias=False),
                GroupedLinear(1280, 640, groups, bias=False),
                GroupedLinear(1280, 640, groups, bias=False),
            ]
            self.assertEqual(
                sum(projection.weight.numel() for projection in projections),
                expected,
            )

    def test_small_grouped_baseline_forward(self):
        config = Qwen3Config(**common_config(groups=4))
        config.experiment6_qkv_groups = 4
        model = Experiment6BaselineForCausalLM(config).eval()
        tokens = torch.randint(0, config.vocab_size, (2, 8))
        with torch.no_grad():
            logits = model(input_ids=tokens, use_cache=False).logits
        self.assertEqual(tuple(logits.shape), (2, 8, config.vocab_size))

    def test_small_coupled_mhar_forward(self):
        config = Qwen3AttnResConfig(
            attnres_mode="full_mh", attnres_num_heads=4,
            **common_config(groups=4))
        config.experiment6_qkv_groups = 4
        model = Experiment6MHARForCausalLM(config).eval()
        tokens = torch.randint(0, config.vocab_size, (2, 8))
        with torch.no_grad():
            logits = model(input_ids=tokens, use_cache=False).logits
        self.assertEqual(tuple(logits.shape), (2, 8, config.vocab_size))

    def test_grouped_checkpoint_round_trip(self):
        config = Qwen3Config(**common_config(groups=4))
        config.experiment6_qkv_groups = 4
        original = Experiment6BaselineForCausalLM(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            original.save_pretrained(directory)
            restored = Experiment6BaselineForCausalLM.from_pretrained(directory).eval()
        self.assertEqual(restored.config.experiment6_qkv_groups, 4)
        for expected, observed in zip(original.parameters(), restored.parameters()):
            torch.testing.assert_close(expected, observed)

    def test_coupled_mhar_rejects_mismatched_group_counts(self):
        config = Qwen3AttnResConfig(
            attnres_mode="full_mh", attnres_num_heads=4,
            **common_config(groups=4))
        config.experiment6_qkv_groups = 8
        with self.assertRaisesRegex(ValueError, "qkv_groups == attnres_num_heads"):
            Experiment6MHARForCausalLM(config)


if __name__ == "__main__":
    unittest.main()
