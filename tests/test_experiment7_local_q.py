"""Correctness and isolation tests for Experiment 7 Local-Q / Global-KV."""

import tempfile
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from src.attention_residuals.modeling_qwen3_attnres import (
    Qwen3AttnResConfig, Qwen3AttnResForCausalLM,
)
from src.experiments.experiment6_coupled_qkv import GroupedLinear
from src.experiments.experiment7_local_q import (
    Experiment7BaselineForCausalLM, Experiment7MHARForCausalLM,
    experiment7_parameter_report,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


def common(*, groups=4, layers=1):
    heads = 8 if groups == 4 else 16
    kv_heads = 4 if groups == 4 else 8
    return dict(vocab_size=64, hidden_size=32, num_hidden_layers=layers,
                num_attention_heads=heads, num_key_value_heads=kv_heads,
                intermediate_size=64, max_position_embeddings=32,
                head_dim=32 // heads, tie_word_embeddings=True, rms_norm_eps=1e-6)


class Experiment7LocalQTest(unittest.TestCase):
    def test_only_q_is_grouped_and_matches_dense_diagonal_blocks(self):
        kwargs = common(groups=4)
        torch.manual_seed(42); dense = Qwen3ForCausalLM(Qwen3Config(**kwargs))
        config = Qwen3Config(**kwargs); config.experiment7_local_q_groups = 4
        torch.manual_seed(42); local = Experiment7BaselineForCausalLM(config)
        dense_attention = dense.model.layers[0].self_attn
        attention = local.model.layers[0].self_attn
        self.assertIsInstance(attention.q_proj, GroupedLinear)
        for name in ("k_proj", "v_proj", "o_proj"):
            self.assertIsInstance(getattr(attention, name), nn.Linear)
            torch.testing.assert_close(getattr(attention, name).weight,
                                       getattr(dense_attention, name).weight)
        expected = GroupedLinear.from_dense(dense_attention.q_proj, 4)
        torch.testing.assert_close(attention.q_proj.weight, expected.weight)

    def test_grouped_q_equals_explicit_mask(self):
        config = Qwen3Config(**common(groups=8)); config.experiment7_local_q_groups = 8
        model = Experiment7BaselineForCausalLM(config)
        query = model.model.layers[0].self_attn.q_proj
        inputs = torch.randn(2, 3, 32)
        torch.testing.assert_close(query(inputs), F.linear(inputs, query.dense_weight()))

    def test_lq8_preserves_mhar_and_global_kv_initialization(self):
        kwargs = common(groups=8)
        dense_config = Qwen3AttnResConfig(attnres_mode="full_mh", attnres_num_heads=8, **kwargs)
        local_config = Qwen3AttnResConfig(attnres_mode="full_mh", attnres_num_heads=8, **kwargs)
        local_config.experiment7_local_q_groups = 8
        torch.manual_seed(7); dense = Qwen3AttnResForCausalLM(dense_config)
        torch.manual_seed(7); local = Experiment7MHARForCausalLM(local_config)
        dense_layer, local_layer = dense.model.layers[0], local.model.layers[0]
        torch.testing.assert_close(dense_layer.attn_res_proj.weight, local_layer.attn_res_proj.weight)
        for name in ("k_proj", "v_proj", "o_proj"):
            torch.testing.assert_close(getattr(dense_layer.self_attn, name).weight,
                                       getattr(local_layer.self_attn, name).weight)

    def test_parameter_report_counts_local_q_but_dense_kv(self):
        config = Qwen3Config(**common(groups=4, layers=2)); config.experiment7_local_q_groups = 4
        report = experiment7_parameter_report(Experiment7BaselineForCausalLM(config))
        self.assertEqual(report["local_q_groups"], 4)
        self.assertEqual(report["q_parameters"], 2 * 32 * 32 // 4)
        self.assertEqual(report["k_parameters"], 2 * 16 * 32)
        self.assertEqual(report["v_parameters"], 2 * 16 * 32)
        self.assertEqual(report["o_parameters"], 2 * 32 * 32)

    def test_real_1b_projection_counts(self):
        layers = 36
        for groups, expected_qkv in ((4, 2_048_000), (8, 1_843_200)):
            q = 1280 * 1280 // groups
            kv = 2 * 640 * 1280
            self.assertEqual(q + kv, expected_qkv)
            self.assertEqual(layers * (q + kv), layers * expected_qkv)

    def test_small_forward_and_checkpoint_round_trip(self):
        config = Qwen3Config(**common(groups=4)); config.experiment7_local_q_groups = 4
        original = Experiment7BaselineForCausalLM(config).eval()
        tokens = torch.randint(0, 64, (2, 8))
        with torch.no_grad(): self.assertEqual(original(input_ids=tokens).logits.shape, (2, 8, 64))
        with tempfile.TemporaryDirectory() as directory:
            original.save_pretrained(directory)
            restored = Experiment7BaselineForCausalLM.from_pretrained(directory)
        self.assertEqual(restored.config.experiment7_local_q_groups, 4)
        for expected, observed in zip(original.parameters(), restored.parameters()):
            torch.testing.assert_close(expected, observed)

    def test_mhar_rejects_mismatched_groups(self):
        config = Qwen3AttnResConfig(attnres_mode="full_mh", attnres_num_heads=4,
                                    **common(groups=4))
        config.experiment7_local_q_groups = 8
        with self.assertRaisesRegex(ValueError, "local-Q groups == attnres_num_heads"):
            Experiment7MHARForCausalLM(config)


if __name__ == "__main__": unittest.main()
