"""CPU tests for the fixed-data Experiment 1 evaluation workflow."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from src.experiments.experiment1_partition_compatibility import (
    analyze_discovery_rows,
    load_fixed_eval_artifact,
    materialize_document_disjoint_sequences,
    save_fixed_eval_artifact,
    select_confirmation_partitions,
)

ROOT = Path(__file__).resolve().parents[1]

from src.attention_residuals.mhar_partition import (
    REFERENCE_PARTITION_H4,
    coordinate_distance,
    generate_pair_partitions,
    original_pair_retention,
    partition_id,
)


class FixedArtifactTest(unittest.TestCase):
    def test_round_trip_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed.pt"
            discovery = torch.arange(48, dtype=torch.int64).view(3, 16)
            confirmation = torch.arange(32, dtype=torch.int64).view(2, 16)
            manifest = save_fixed_eval_artifact(
                path, discovery, confirmation, {"source": "unit-test"})
            payload, observed_hash = load_fixed_eval_artifact(path)

            self.assertEqual(observed_hash, manifest["artifact_sha256"])
            self.assertEqual(payload["metadata"], {"source": "unit-test"})
            torch.testing.assert_close(
                payload["discovery_input_ids"], discovery.to(torch.int32))
            torch.testing.assert_close(
                payload["confirmation_input_ids"], confirmation.to(torch.int32))

    def test_materialization_discards_boundary_document_tail(self):
        class CharacterTokenizer:
            eos_token_id = 99

            @staticmethod
            def encode(text, add_special_tokens=False):
                del add_special_tokens
                return [ord(value) - ord("a") + 1 for value in text]

        documents = [
            {"text": "aaaaaaaaaa"},
            {"text": "bbbbbbbbbb"},
        ]
        discovery, confirmation, counts = materialize_document_disjoint_sequences(
            documents,
            CharacterTokenizer(),
            seq_len=4,
            discovery_count=2,
            confirmation_count=2,
        )

        self.assertEqual(counts, {"discovery": 1, "confirmation": 1, "total": 2})
        self.assertTrue(torch.all(discovery == 1))
        self.assertTrue(torch.all(confirmation == 2))


def synthetic_discovery_rows():
    rows = []
    for partition in generate_pair_partitions(8):
        retained, retention = original_pair_retention(partition)
        total_distance, mean_distance = coordinate_distance(partition)
        nll = 2.5 + 0.01 * mean_distance
        rows.append({
            "partition_id": partition_id(partition),
            "partition": [list(pair) for pair in partition],
            "original_pairs_retained": retained,
            "retention": retention,
            "total_coordinate_distance": total_distance,
            "mean_coordinate_distance": mean_distance,
            "total_nll": nll * 100,
            "valid_tokens": 100,
            "nll": nll,
            "ppl": float(torch.exp(torch.tensor(nll))),
            "elapsed_seconds": 0.1,
        })
    return rows


class AnalysisTest(unittest.TestCase):
    def test_exhaustive_analysis_and_confirmation_selection(self):
        rows = synthetic_discovery_rows()
        ranked, summary = analyze_discovery_rows(rows)

        self.assertEqual(len(ranked), 105)
        self.assertTrue(summary["complete_exhaustive_run"])
        self.assertEqual(
            {row["count"] for row in summary["retention_summary"]},
            {1, 12, 32, 60},
        )
        self.assertEqual(
            [row["count"] for row in summary["distance_summary"]],
            [1, 6, 12, 20, 24, 18, 24],
        )
        self.assertGreater(summary["spearman_distance_vs_nll"], 0.99)

        selected, roles = select_confirmation_partitions(rows)
        self.assertGreaterEqual(len(selected), 2)
        all_roles = {role for values in roles.values() for role in values}
        self.assertEqual(
            all_roles, {"reference", "discovery_best", "discovery_worst"})


class EvaluationCliSmokeTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MHAR_RUN_MODEL_INTEGRATION") == "1",
        "set MHAR_RUN_MODEL_INTEGRATION=1 to run the tiny model CLI smoke test",
    )
    def test_tiny_checkpoint_fixed_data_and_resume(self):
        from src.attention_residuals.modeling_qwen3_attnres import (
            Qwen3AttnResConfig,
            Qwen3AttnResForCausalLM,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            artifact = root / "fixed.pt"
            output = root / "run"

            torch.manual_seed(19)
            config = Qwen3AttnResConfig(
                vocab_size=64,
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                intermediate_size=64,
                max_position_embeddings=16,
                head_dim=8,
                attnres_mode="full_mh",
                attnres_num_heads=4,
                rms_norm_eps=1e-6,
                tie_word_embeddings=True,
            )
            model = Qwen3AttnResForCausalLM(config)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if "res_proj.weight" in name:
                        parameter.normal_(0, 0.2)
            model.save_pretrained(checkpoint)

            discovery = torch.randint(0, config.vocab_size, (2, 8))
            confirmation = torch.randint(0, config.vocab_size, (2, 8))
            save_fixed_eval_artifact(
                artifact, discovery, confirmation, {"source": "cli-smoke"})

            command = [
                sys.executable,
                "-m", "src.experiments.experiment1_partition_compatibility",
                "evaluate",
                "--checkpoint", str(checkpoint),
                "--artifact", str(artifact),
                "--output-dir", str(output),
                "--split", "discovery",
                "--device", "cpu",
                "--dtype", "fp32",
                "--batch-size", "1",
                "--allow-nonstandard-model",
                "--smoke-limit", "3",
                "--wandb-mode", "offline",
                "--wandb-project", "MHAR Stuff",
                "--wandb-group", "unit-test",
            ]
            subprocess_env = dict(
                os.environ,
                WANDB_DIR=str(root),
                WANDB_CACHE_DIR=str(root / "wandb-cache"),
                WANDB_CONFIG_DIR=str(root / "wandb-config"),
                WANDB_DATA_DIR=str(root / "wandb-data"),
            )
            first_run = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True, env=subprocess_env)
            self.assertEqual(first_run.returncode, 0, first_run.stderr)
            # Resume must skip the same completed rows without duplicating them.
            resumed_run = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True, env=subprocess_env)
            self.assertEqual(resumed_run.returncode, 0, resumed_run.stderr)

            rows = [
                json.loads(line)
                for line in (output / "discovery_results.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 3)
            self.assertEqual(len({row["partition_id"] for row in rows}), 3)
            manifest = json.loads((output / "discovery_run_manifest.json").read_text())
            self.assertEqual(manifest["run_identity"]["partition_count"], 3)
            self.assertEqual(
                manifest["run_identity"]["artifact_sha256"],
                load_fixed_eval_artifact(artifact)[1],
            )
            self.assertEqual(manifest["wandb"]["project"], "MHAR Stuff")
            self.assertTrue(manifest["wandb"]["run_id"])

            analysis_dir = output / "analysis"
            subprocess.run([
                sys.executable,
                "-m", "src.experiments.experiment1_partition_compatibility",
                "analyze",
                "--discovery-results", str(output / "discovery_results.jsonl"),
                "--output-dir", str(analysis_dir),
                "--allow-incomplete",
                "--wandb-mode", "offline",
                "--wandb-project", "MHAR Stuff",
                "--wandb-group", "unit-test",
            ], check=True, cwd=ROOT, capture_output=True, text=True,
               env=subprocess_env)
            self.assertTrue((analysis_dir / "analysis.json").is_file())
            self.assertTrue((analysis_dir / "analysis.md").is_file())
            for stem in (
                "fig_nll_vs_distance", "fig_nll_by_retention", "fig_partition_ranking"
            ):
                self.assertTrue((analysis_dir / f"{stem}.png").is_file())
                self.assertTrue((analysis_dir / f"{stem}.pdf").is_file())
            self.assertEqual(
                len((analysis_dir / "ranked_partitions.csv").read_text().splitlines()),
                4,
            )


if __name__ == "__main__":
    unittest.main()
