"""Correctness and isolation tests for Experiment 8 hybrid Q."""

import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from src.attention_residuals.modeling_qwen3_attnres import (
    Qwen3AttnResConfig, Qwen3AttnResForCausalLM,
)
from src.experiments.experiment8_hybrid_q import (
    Experiment8BaselineForCausalLM, Experiment8MHARForCausalLM,
    HybridQueryLinear, experiment8_parameter_report,
)
from src.training.train_scratch import build_model
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


def common(*, layers=1):
    return dict(
        vocab_size=64, hidden_size=32, num_hidden_layers=layers,
        num_attention_heads=16, num_key_value_heads=8,
        intermediate_size=64, max_position_embeddings=32,
        head_dim=2, tie_word_embeddings=True, rms_norm_eps=1e-6,
    )


class Experiment8HybridQTest(unittest.TestCase):
    @staticmethod
    def training_args(variant="hq8"):
        return SimpleNamespace(
            qkv_groups=None, experiment6_variant=None,
            local_q_groups=None, experiment7_variant=None,
            hybrid_q_groups=8, experiment8_variant=variant,
            mode="full_mh" if variant == "hq8" else "baseline",
            attnres_heads=8, hidden_size=32, num_layers=1, num_heads=16,
            num_kv_heads=8, intermediate_size=64, seq_len=16,
            resume_from=None, branch_from=None, fsdp=False,
            mixed_partition=None, num_blocks=4, gate_type="scalar",
            null_source=False, hyper_n=1,
        )

    def test_only_q_changes_and_retained_weights_match_dense(self):
        kwargs = common()
        torch.manual_seed(42); dense = Qwen3ForCausalLM(Qwen3Config(**kwargs))
        config = Qwen3Config(**kwargs); config.experiment8_hybrid_q_groups = 8
        torch.manual_seed(42); hybrid = Experiment8BaselineForCausalLM(config)
        dense_attention = dense.model.layers[0].self_attn
        attention = hybrid.model.layers[0].self_attn
        self.assertIsInstance(attention.q_proj, HybridQueryLinear)
        for name in ("k_proj", "v_proj", "o_proj"):
            self.assertIsInstance(getattr(attention, name), nn.Linear)
            torch.testing.assert_close(
                getattr(attention, name).weight, getattr(dense_attention, name).weight)
        expected = HybridQueryLinear.from_dense(dense_attention.q_proj)
        torch.testing.assert_close(attention.q_proj.local_weight, expected.local_weight)
        torch.testing.assert_close(attention.q_proj.global_weight, expected.global_weight)

    def test_projection_equals_explicit_masked_dense_matrix(self):
        dense = nn.Linear(32, 32, bias=True)
        hybrid = HybridQueryLinear.from_dense(dense)
        inputs = torch.randn(2, 3, 32)
        torch.testing.assert_close(
            hybrid(inputs), F.linear(inputs, hybrid.dense_weight(), hybrid.dense_bias()))

    def test_even_heads_are_local_and_odd_heads_are_global(self):
        hybrid = HybridQueryLinear(32, 32)
        with torch.no_grad():
            hybrid.local_weight.fill_(1)
            hybrid.global_weight.fill_(1)
        inputs = torch.zeros(1, 32)
        inputs[0, 7] = 1  # group 1, outside group 0's local coordinates
        heads = hybrid(inputs).view(8, 2, 2)
        torch.testing.assert_close(heads[0, 0], torch.zeros(2))
        torch.testing.assert_close(heads[0, 1], torch.ones(2))
        torch.testing.assert_close(heads[1, 0], torch.ones(2))
        torch.testing.assert_close(heads[1, 1], torch.ones(2))

    def test_hq8_preserves_mhar_and_dense_kvo_initialization(self):
        kwargs = common()
        dense_config = Qwen3AttnResConfig(
            attnres_mode="full_mh", attnres_num_heads=8, **kwargs)
        hybrid_config = Qwen3AttnResConfig(
            attnres_mode="full_mh", attnres_num_heads=8, **kwargs)
        hybrid_config.experiment8_hybrid_q_groups = 8
        torch.manual_seed(7); dense = Qwen3AttnResForCausalLM(dense_config)
        torch.manual_seed(7); hybrid = Experiment8MHARForCausalLM(hybrid_config)
        dense_layer, hybrid_layer = dense.model.layers[0], hybrid.model.layers[0]
        torch.testing.assert_close(
            dense_layer.attn_res_proj.weight, hybrid_layer.attn_res_proj.weight)
        for name in ("k_proj", "v_proj", "o_proj"):
            torch.testing.assert_close(
                getattr(dense_layer.self_attn, name).weight,
                getattr(hybrid_layer.self_attn, name).weight)

    def test_parameter_report_matches_real_1b_projection_counts(self):
        config = Qwen3Config(**common(layers=2)); config.experiment8_hybrid_q_groups = 8
        report = experiment8_parameter_report(Experiment8BaselineForCausalLM(config))
        self.assertEqual(report["q_local_parameters"], 2 * 8 * 2 * 4)
        self.assertEqual(report["q_global_parameters"], 2 * 8 * 2 * 32)
        self.assertEqual(report["q_parameters"], 2 * (64 + 512))
        q = 8 * 160 * 80 + 8 * 1280 * 80
        kv = 2 * 1280 * 640
        self.assertEqual(q, 921_600)
        self.assertEqual(q + kv, 2_560_000)

    def test_small_forward_and_checkpoint_round_trip(self):
        config = Qwen3Config(**common()); config.experiment8_hybrid_q_groups = 8
        original = Experiment8BaselineForCausalLM(config).eval()
        tokens = torch.randint(0, 64, (2, 8))
        with torch.no_grad():
            self.assertEqual(original(input_ids=tokens).logits.shape, (2, 8, 64))
        with tempfile.TemporaryDirectory() as directory:
            original.save_pretrained(directory)
            restored = Experiment8BaselineForCausalLM.from_pretrained(directory)
        self.assertEqual(restored.config.experiment8_hybrid_q_groups, 8)
        for expected, observed in zip(original.parameters(), restored.parameters()):
            torch.testing.assert_close(expected, observed)

    def test_rejects_non_matching_gqa_layout(self):
        kwargs = common(); kwargs["num_key_value_heads"] = 4
        config = Qwen3Config(**kwargs); config.experiment8_hybrid_q_groups = 8
        with self.assertRaisesRegex(ValueError, "number of KV heads"):
            Experiment8BaselineForCausalLM(config)

    def test_mhar_rejects_non_h8_routing(self):
        config = Qwen3AttnResConfig(
            attnres_mode="full_mh", attnres_num_heads=4, **common())
        config.experiment8_hybrid_q_groups = 8
        with self.assertRaisesRegex(ValueError, "groups == attnres_num_heads"):
            Experiment8MHARForCausalLM(config)

    def test_training_entrypoint_builds_frozen_hq8_and_bhq8(self):
        hq8 = build_model(self.training_args("hq8"), torch.device("cpu"))
        bhq8 = build_model(self.training_args("bhq8"), torch.device("cpu"))
        self.assertIsInstance(hq8, Experiment8MHARForCausalLM)
        self.assertIsInstance(bhq8, Experiment8BaselineForCausalLM)
        self.assertEqual(hq8.config.experiment8_local_head_position, "even")
        self.assertEqual(bhq8.config.experiment8_global_head_position, "odd")

    def test_training_entrypoint_rejects_wrong_frozen_variant_layout(self):
        args = self.training_args("hq8")
        args.attnres_heads = 4
        with self.assertRaisesRegex(ValueError, "requires mode/heads/hybrid_q_groups"):
            build_model(args, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
