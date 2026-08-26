"""Tests for resumable from-scratch training checkpoints."""

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

from src.training import train_scratch
from src.attention_residuals.modeling_qwen3_attnres import (
    Qwen3AttnResConfig,
    Qwen3AttnResForCausalLM,
)
from src.training.train_scratch import (
    data_files_identity,
    load_training_state,
    parse_keep_steps,
    save_training_checkpoint,
)


class FakeTokenizer:
    def save_pretrained(self, path):
        path = Path(path)
        (path / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer": "fake"}), encoding="utf-8")


class ResumeCheckpointTest(unittest.TestCase):
    def test_training_entrypoint_imports_sys_for_run_manifest(self):
        self.assertIs(train_scratch.sys, sys)

    def make_model_and_optimizer(self):
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
            tie_word_embeddings=True,
        )
        model = Qwen3AttnResForCausalLM(config)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: min(1.0, step / 10))
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        model(input_ids=input_ids, labels=input_ids).loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        return model, optimizer, scheduler

    def test_full_state_round_trip_and_rotation(self):
        model, optimizer, scheduler = self.make_model_and_optimizer()
        identity = {"source_commit": "unit-test", "global_batch_size": 32}
        tokenizer = FakeTokenizer()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for step in (1, 2, 3):
                checkpoint = save_training_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=step,
                    chunks_consumed=step * 8,
                    run_identity=identity,
                    out_dir=output,
                    keep_last=2,
                    wandb_run_id="wandb-unit-test",
                    elapsed_training_seconds=float(step),
                )
                self.assertTrue((checkpoint / "model.safetensors").is_file())
                self.assertTrue((checkpoint / "training_state.pt").is_file())

            self.assertFalse((output / "step-1").exists())
            self.assertTrue((output / "step-2").is_dir())
            self.assertTrue((output / "step-3").is_dir())

            state = load_training_state(output / "step-3")
            self.assertEqual(state["global_step"], 3)
            self.assertEqual(state["chunks_consumed"], 24)
            self.assertEqual(state["run_identity"], identity)
            self.assertEqual(state["wandb_run_id"], "wandb-unit-test")
            self.assertTrue(state["optimizer"]["state"])
            self.assertEqual(
                state["scheduler"]["last_epoch"], scheduler.state_dict()["last_epoch"])

    def test_data_file_identity_is_content_addressed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.parquet").write_bytes(b"second")
            (root / "a.parquet").write_bytes(b"first")
            identity = data_files_identity(str(root / "*.parquet"))
            self.assertEqual(
                [Path(row["path"]).name for row in identity["files"]],
                ["a.parquet", "b.parquet"],
            )
            self.assertEqual(identity["files"][0]["bytes"], 5)
            self.assertEqual(
                identity["files"][0]["sha256"],
                "a7937b64b8caa58f03721bb6bacf5c78cb235febe0e70b1"
                "b84cd99541461a08e",
            )

    def test_protected_milestones_survive_checkpoint_rotation(self):
        model, optimizer, scheduler = self.make_model_and_optimizer()
        identity = {"source_commit": "unit-test", "global_batch_size": 32}
        tokenizer = FakeTokenizer()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for step in (1, 2, 3, 4, 5):
                save_training_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=step,
                    chunks_consumed=step * 8,
                    run_identity=identity,
                    out_dir=output,
                    keep_last=2,
                    keep_steps=parse_keep_steps("2,4"),
                    wandb_run_id="wandb-unit-test",
                    elapsed_training_seconds=float(step),
                )

            self.assertEqual(
                sorted(path.name for path in output.glob("step-*")),
                ["step-2", "step-4", "step-5"],
            )

    def test_keep_steps_validation(self):
        self.assertEqual(parse_keep_steps("2000, 5000,2000"), {2000, 5000})
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_keep_steps("0,2000")
        with self.assertRaisesRegex(ValueError, "comma-separated"):
            parse_keep_steps("two-thousand")

    def test_h8_launcher_matches_h16_scientific_recipe(self):
        def launcher_flags(name):
            text = (ROOT / name).read_text(encoding="utf-8")
            return dict(re.findall(r"^  --([a-z_]+)\s+([^\\\n]+)", text, re.MULTILINE))

        h8 = launcher_flags("scripts/train/run_experiment2_train_1b_h8.sh")
        h16 = launcher_flags("scripts/train/run_experiment2_train_1b_h16.sh")
        non_scientific = {"attnres_heads", "keep_steps", "wandb_group", "run_name"}
        self.assertEqual(
            {key: value for key, value in h8.items() if key not in non_scientific},
            {key: value for key, value in h16.items() if key not in non_scientific},
        )
        self.assertEqual(h8["attnres_heads"], "8 ")
        self.assertEqual(h16["attnres_heads"], "16 ")
        self.assertEqual(h8["keep_steps"], "2000,5000,10000,20000 ")


if __name__ == "__main__":
    unittest.main()
