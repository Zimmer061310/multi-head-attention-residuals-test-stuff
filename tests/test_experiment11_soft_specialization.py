"""Correctness and isolation tests for Experiment 11 soft query masks."""

import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from src.attention_residuals.modeling_qwen3_attnres import (
    Qwen3AttnResConfig,
    Qwen3AttnResForCausalLM,
)
from src.experiments.experiment11_soft_specialization import (
    Experiment11MHARForCausalLM,
    RUNS,
    SoftSpecializedQueryLinear,
    experiment11_parameter_report,
    weight_specialization_metrics,
)
from src.training.train_scratch import build_model


def common(*, layers=1):
    return dict(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=layers,
        num_attention_heads=16,
        num_key_value_heads=8,
        intermediate_size=64,
        max_position_embeddings=32,
        head_dim=2,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
    )


def config(run_id, *, layers=1):
    result = Qwen3AttnResConfig(
        attnres_mode="full_mh", attnres_num_heads=8, **common(layers=layers)
    )
    result.experiment11_run_id = run_id
    return result


class Experiment11SoftSpecializationTest(unittest.TestCase):
    @staticmethod
    def training_args(run_id="s2q8-l025"):
        return SimpleNamespace(
            qkv_groups=None,
            experiment6_variant=None,
            local_q_groups=None,
            experiment7_variant=None,
            hybrid_q_groups=None,
            experiment8_variant=None,
            experiment11_run=run_id,
            mode="full_mh",
            attnres_heads=8,
            hidden_size=32,
            num_layers=1,
            num_heads=16,
            num_kv_heads=8,
            intermediate_size=64,
            seq_len=16,
            resume_from=None,
            branch_from=None,
            fsdp=False,
            mixed_partition=None,
            num_blocks=4,
            gate_type="scalar",
            null_source=False,
            hyper_n=1,
        )

    def test_frozen_matrix_contains_nine_unique_runs(self):
        self.assertEqual(len(RUNS), 9)
        self.assertEqual(len({row.run_id for row in RUNS}), 9)
        self.assertEqual(
            {row.lambda_value for row in RUNS if row.family != "m8"},
            {0.0, 0.1, 0.25, 0.5},
        )

    def test_s2q_mask_and_gslq_head_order(self):
        s2 = SoftSpecializedQueryLinear(32, 32, family="s2q8", lambda_value=.25)
        gsl = SoftSpecializedQueryLinear(32, 32, family="gslq8", lambda_value=.25)
        s2_mask = s2._experiment11_mask
        gsl_mask = gsl._experiment11_mask
        for group in range(8):
            local = slice(4 * group, 4 * (group + 1))
            for head in range(2):
                self.assertTrue(torch.all(s2_mask[group, head, :, local] == 1))
                outside = torch.cat(
                    (s2_mask[group, head, :, :local.start],
                     s2_mask[group, head, :, local.stop:]), dim=-1)
                self.assertTrue(torch.all(outside == .25))
            self.assertTrue(torch.all(gsl_mask[group, 0] == s2_mask[group, 0]))
            self.assertTrue(torch.all(gsl_mask[group, 1] == 1))

    def test_projection_equals_explicit_masked_dense_matrix(self):
        dense = nn.Linear(32, 32, bias=False)
        soft = SoftSpecializedQueryLinear.from_dense(
            dense, family="s2q8", lambda_value=.1
        )
        inputs = torch.randn(2, 3, 32)
        torch.testing.assert_close(
            soft(inputs), F.linear(inputs, soft.masked_weight())
        )

    def test_lambda_one_is_bitwise_dense_forward(self):
        dense = nn.Linear(32, 32, bias=False)
        endpoint = SoftSpecializedQueryLinear.from_dense(
            dense, family="m8", lambda_value=1.0
        )
        inputs = torch.randn(2, 4, 32)
        self.assertTrue(torch.equal(dense(inputs), endpoint(inputs)))

    def test_lambda_zero_blocks_cross_input_and_gradient(self):
        layer = SoftSpecializedQueryLinear(
            32, 32, family="s2q8", lambda_value=0.0, bias=False
        )
        inputs = torch.randn(2, 3, 32, requires_grad=True)
        layer(inputs).sum().backward()
        grad = layer.weight.grad.view(8, 2, 2, 32)
        for group in range(8):
            outside = torch.cat((grad[group, :, :, :4 * group],
                                 grad[group, :, :, 4 * (group + 1):]), dim=-1)
            self.assertTrue(torch.equal(outside, torch.zeros_like(outside)))

    def test_component_decomposition_reconstructs_projection_without_bias(self):
        layer = SoftSpecializedQueryLinear(
            32, 32, family="gslq8", lambda_value=.25, bias=False
        )
        inputs = torch.randn(2, 3, 32)
        local, cross = layer.local_cross_components(inputs)
        self.assertEqual(local.shape, (2, 3, 8, 2, 2))
        torch.testing.assert_close((local + cross).reshape(2, 3, 32), layer(inputs))

    def test_all_runs_share_identical_physical_initialization(self):
        parameters = []
        for row in RUNS:
            torch.manual_seed(42)
            model = Experiment11MHARForCausalLM(config(row.run_id))
            parameters.append(dict(model.named_parameters()))
        keys = parameters[0].keys()
        for state in parameters[1:]:
            self.assertEqual(keys, state.keys())
            for key in keys:
                self.assertTrue(torch.equal(parameters[0][key], state[key]), key)

    def test_only_q_forward_is_masked_and_kvo_weights_match_dense_mhar(self):
        torch.manual_seed(7)
        dense = Qwen3AttnResForCausalLM(
            Qwen3AttnResConfig(
                attnres_mode="full_mh", attnres_num_heads=8, **common()
            )
        )
        torch.manual_seed(7)
        soft = Experiment11MHARForCausalLM(config("s2q8-l025"))
        dense_layer, soft_layer = dense.model.layers[0], soft.model.layers[0]
        torch.testing.assert_close(
            dense_layer.attn_res_proj.weight, soft_layer.attn_res_proj.weight
        )
        torch.testing.assert_close(
            dense_layer.self_attn.q_proj.weight, soft_layer.self_attn.q_proj.weight
        )
        for name in ("k_proj", "v_proj", "o_proj"):
            self.assertIsInstance(getattr(soft_layer.self_attn, name), nn.Linear)
            torch.testing.assert_close(
                getattr(dense_layer.self_attn, name).weight,
                getattr(soft_layer.self_attn, name).weight,
            )

    def test_small_forward_and_checkpoint_round_trip(self):
        original = Experiment11MHARForCausalLM(config("gslq8-l010")).eval()
        tokens = torch.randint(0, 64, (2, 8))
        with torch.no_grad():
            self.assertEqual(original(input_ids=tokens).logits.shape, (2, 8, 64))
        with tempfile.TemporaryDirectory() as directory:
            original.save_pretrained(directory)
            restored = Experiment11MHARForCausalLM.from_pretrained(directory).eval()
        self.assertEqual(restored.config.experiment11_run_id, "gslq8-l010")
        for key, expected in original.state_dict().items():
            torch.testing.assert_close(expected, restored.state_dict()[key])
        torch.testing.assert_close(
            original.model.layers[0].self_attn.q_proj._experiment11_mask,
            restored.model.layers[0].self_attn.q_proj._experiment11_mask,
        )

    def test_parameter_and_weight_reports_cover_real_modules(self):
        model = Experiment11MHARForCausalLM(config("s2q8-l025", layers=2))
        report = experiment11_parameter_report(model)
        self.assertEqual(report["q_parameters"], 2 * 32 * 32)
        self.assertEqual(report["mask_parameters"], 0)
        metrics = weight_specialization_metrics(model)
        self.assertEqual(len(metrics["rows"]), 2 * 8 * 2)
        self.assertTrue(all(row["cross_scale"] == .25 for row in metrics["rows"]))

    def test_training_entrypoint_is_explicit_and_rejects_projection_overlap(self):
        model = build_model(self.training_args(), torch.device("cpu"))
        self.assertIsInstance(model, Experiment11MHARForCausalLM)
        self.assertEqual(model.config.experiment11_run_id, "s2q8-l025")
        args = self.training_args()
        args.hybrid_q_groups = 8
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            build_model(args, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
