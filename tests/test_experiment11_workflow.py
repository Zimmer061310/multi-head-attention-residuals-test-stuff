"""Protocol, probe, and selection tests for Experiment 11."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from figures.gen_fig_experiment11_soft_specialization import (
    nll_curves,
    selected_contrasts,
    selected_heatmap,
    softness_curves,
    style,
    trajectories,
)

from src.attention_residuals.modeling_qwen3_attnres import Qwen3AttnResConfig
from src.experiments.experiment11_soft_specialization import (
    Experiment11MHARForCausalLM,
    RUNS,
)
from src.experiments.experiment11_workflow import (
    canonical_hash,
    load_spec,
    paired_bootstrap,
    metric_contrast,
    probe_model,
    select_command,
    validate_selection_manifest,
)


def tiny_model(run_id):
    config = Qwen3AttnResConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=8,
        intermediate_size=64,
        max_position_embeddings=32,
        head_dim=2,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
        attnres_mode="full_mh",
        attnres_num_heads=8,
    )
    config.experiment11_run_id = run_id
    return Experiment11MHARForCausalLM(config).eval()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


class Experiment11WorkflowTest(unittest.TestCase):
    def test_protocol_is_frozen_and_complete(self):
        spec = load_spec()
        self.assertEqual(len(spec["runs"]), 9)
        self.assertEqual(spec["fixed_evaluation"]["discovery_probe_sequences"], 32)
        self.assertEqual(spec["inference"]["bootstrap_samples"], 10_000)

    def test_probe_uses_qnorm_and_returns_complete_maps(self):
        model = tiny_model("gslq8-l025")
        result = probe_model(
            model, torch.randint(0, 64, (2, 8)), device=torch.device("cpu")
        )
        self.assertEqual(result["diagnostic_head_positions"], [0])
        self.assertEqual(
            np.asarray(result["metrics"]["r_act"]["layer_group_head"]).shape,
            (1, 8, 2),
        )
        self.assertEqual(len(result["metrics"]["theta_radians"]["per_sequence"]), 2)
        self.assertTrue(np.isfinite(result["metrics"]["r_act"]["mean"]))

    def test_lambda_zero_probe_has_zero_effective_cross_path(self):
        result = probe_model(
            tiny_model("s2q8-l000"),
            torch.randint(0, 64, (1, 8)),
            device=torch.device("cpu"),
        )
        self.assertEqual(result["metrics"]["r_act"]["mean"], 0.0)
        self.assertEqual(
            result["metrics"]["component_cosine_valid_fraction"]["mean"], 0.0
        )

    def test_paired_bootstrap_is_deterministic(self):
        values = np.asarray([-.2, -.1, .0, .1])
        first = paired_bootstrap(values, samples=200, seed=9)
        second = paired_bootstrap(values, samples=200, seed=9)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean"], -.05)

    def test_gsl_softness_uses_corresponding_even_m8_slot(self):
        first = {
            "activation_metrics": {
                "diagnostic_head_positions": [0],
                "metrics": {"r_act": {"per_sequence_head": [[1.0, 99.0], [2.0, 99.0]]}},
            }
        }
        second = {
            "activation_metrics": {
                "diagnostic_head_positions": [0, 1],
                "metrics": {"r_act": {"per_sequence_head": [[3.0, -99.0], [4.0, -99.0]]}},
            }
        }
        result = metric_contrast(first, second, "r_act", samples=100, seed=1)
        self.assertEqual(result["mean"], -2.0)

    def test_discovery_selection_is_frozen_before_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nll = {row.run_id: 5.0 + index / 100 for index, row in enumerate(RUNS)}
            nll.update({"s2q8-l025": 4.0, "gslq8-l050": 4.1})
            for row in RUNS:
                payload = {
                    "format_version": 1,
                    "run_id": row.run_id,
                    "split": "discovery",
                    "artifact_sha256": load_spec()["fixed_evaluation"]["sha256"],
                    "protocol_sha256": __import__(
                        "src.experiments.experiment11_workflow", fromlist=["protocol_sha256"]
                    ).protocol_sha256(),
                    "checkpoint_sha256": f"checkpoint-{row.run_id}",
                    "metrics": {
                        "nll": nll[row.run_id],
                        "sequence_nlls": [nll[row.run_id], nll[row.run_id] + .1],
                    },
                }
                payload["content_sha256"] = canonical_hash(payload)
                write_json(root / "evaluations" / row.run_id / "discovery.json", payload)
            output = root / "selection.json"
            select_command(type("Args", (), {
                "results_root": str(root), "output": str(output)
            })())
            selected = validate_selection_manifest(output, root)
            self.assertEqual(selected["selected"]["s2q8"]["run_id"], "s2q8-l025")
            self.assertEqual(selected["selected"]["gslq8"]["run_id"], "gslq8-l050")
            # A changed discovery input invalidates the frozen manifest.
            path = root / "evaluations/s2q8-l025/discovery.json"
            changed = json.loads(path.read_text())
            changed["metrics"]["nll"] = 9.0
            changed["content_sha256"] = canonical_hash(changed)
            write_json(path, changed)
            with self.assertRaisesRegex(ValueError, "selection input changed"):
                validate_selection_manifest(output, root)

    def test_figure_generator_writes_vector_and_raster_outputs(self):
        curves = []
        trajectory_rows = []
        for split in ("discovery", "confirmation"):
            for index, row in enumerate(RUNS):
                curves.append({
                    "split": split, "run_id": row.run_id, "family": row.family,
                    "lambda": row.lambda_value, "nll": 4.0 + index / 100,
                })
        for row in RUNS:
            for step in (0, 500, 1000, 1500, 2000):
                trajectory_rows.append({
                    "run_id": row.run_id, "family": row.family,
                    "lambda": row.lambda_value, "milestone": step,
                    "r_weight": row.lambda_value, "r_act": row.lambda_value / 2,
                    "theta_radians": row.lambda_value / 3,
                })
        comparison = {
            "id": "selected-minus-m8", "aggregate_delta_nll": -.01,
            "ci95_low": -.02, "ci95_high": -.005,
        }
        summary = {
            "curves": curves,
            "trajectories": trajectory_rows,
            "families": {
                "s2q8": {"selected_run_id": "s2q8-l025", "confirmation_nll": [comparison]},
                "gslq8": {"selected_run_id": "gslq8-l025", "confirmation_nll": [comparison]},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run_id in ("s2q8-l025", "gslq8-l025"):
                write_json(root / "probes" / run_id / "step-2000-confirmation.json", {
                    "activation_metrics": {
                        "diagnostic_head_positions": [0, 1] if run_id.startswith("s2") else [0],
                        "metrics": {
                            "r_act": {"layer_group_head": np.ones((1, 8, 2)).tolist()},
                            "theta_radians": {"layer_group_head": (np.ones((1, 8, 2)) / 2).tolist()},
                        },
                    },
                })
            style()
            nll_curves(summary, root)
            selected_contrasts(summary, root)
            softness_curves(summary, root)
            trajectories(summary, root)
            selected_heatmap(summary, root, root, "s2q8")
            selected_heatmap(summary, root, root, "gslq8")
            for stem in ("nll_curves", "selected_contrasts", "softness_curves",
                         "softness_trajectories", "s2q8_heatmap", "gslq8_heatmap"):
                self.assertGreater((root / f"fig_{stem}.png").stat().st_size, 1000)
                self.assertGreater((root / f"fig_{stem}.pdf").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
