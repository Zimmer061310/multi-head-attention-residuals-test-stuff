import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.experiments import experiment5_washout as washout
from src.experiments.experiment5_controller import training_command, verify_ack
from src.training.train_scratch import checkpoint_due, parse_keep_steps


class WashoutTest(unittest.TestCase):
    def test_cli_string_paths_hash_identically_to_path_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text('{"test": true}', encoding="utf-8")
            self.assertEqual(washout.sha256_file(str(path)), washout.sha256_file(path))

    def small_rows(self, gaps=None):
        protocol = washout.spec().copy()
        protocol.update(sequences_per_split=4, sequence_length=5, bootstrap_samples=100)
        rows = {}
        for offset in protocol["offsets"]:
            for role, delta in zip(washout.ROLES, (0, (gaps or {}).get(offset, 0.01), -0.02)):
                seq = np.arange(4) / 10 + 3 + delta
                metrics = {"sequence_nlls": seq.tolist(), "nll": float(seq.mean()),
                           "total_nll": float(seq.mean()) * 16, "valid_tokens": 16}
                rows[role, offset] = {"splits": {split: copy.deepcopy(metrics) for split in protocol["splits"]}}
        return protocol, rows

    def test_protocol_offsets_and_unchanged_schedule(self):
        protocol = washout.spec()
        self.assertEqual(protocol["offsets"], [0, 1, 2, 5, 10, 20, 50, 100])
        manifest = {"protocol": protocol, "parent_checkpoint": "/parent", "parent_training_manifest": {
            "run_identity": {"data_files": {"pattern": "/data/*.parquet"}}}}
        for gpu, role in enumerate(washout.ROLES):
            cmd = training_command(manifest, "/manifest", "/output", role, gpu)
            for flag, value in (("--steps", "20000"), ("--stop_after_step", "1600"),
                                ("--batch_size", "4"), ("--grad_accum", "8"), ("--eval_every", "0")):
                self.assertEqual(cmd[cmd.index(flag) + 1], value)
            self.assertNotIn("--resume_from", cmd)
            self.assertNotIn("--fused", cmd)
            self.assertEqual(cmd[cmd.index("--save_steps") + 1], "1501,1502,1505,1510,1520,1550,1600")

    def test_sparse_save_and_periodic_default(self):
        steps = parse_keep_steps("1501,1502,1505,1510,1520,1550,1600")
        self.assertEqual([s for s in range(1501, 1601) if checkpoint_due(s, 0, steps)], sorted(steps))
        self.assertTrue(checkpoint_due(2000, 2000))
        self.assertFalse(checkpoint_due(1999, 2000))
        self.assertFalse(checkpoint_due(1515, 0, steps))

    def test_step0_reproduces_real_archived_results(self):
        reference = [json.loads(l) for l in washout.REFERENCE.read_text().splitlines()]
        rows = {(role, 0): {"splits": {split: next(r for r in reference if r["candidate_id"] == washout.CANDIDATES[role] and r["split"] == split)
                                      for split in washout.spec()["splits"]}} for role in washout.ROLES}
        self.assertEqual(len(washout.check_step0(rows, reference, washout.spec())), 6)
        changed = copy.deepcopy(rows)
        metrics = changed["predicted-good", 0]["splits"]["confirmation"]
        metrics["sequence_nlls"][0] += 0.001
        with self.assertRaisesRegex(RuntimeError, "step0 failed"):
            washout.check_step0(changed, reference, washout.spec())

    def test_invalid_measurement_is_rejected(self):
        protocol, rows = self.small_rows()
        metrics = rows["unchanged", 0]["splits"]["confirmation"]
        metrics["valid_tokens"] -= 1
        with self.assertRaisesRegex(RuntimeError, "token count"):
            washout.validate_metrics(metrics, protocol)
        metrics["sequence_nlls"][0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            washout.validate_metrics(metrics, protocol)

    def test_washout_vs_persistence_and_reversal(self):
        protocol, rows = self.small_rows({t: (0.011 if t < 5 else 0) for t in washout.spec()["offsets"]})
        table, result = washout.analyze_rows(rows, protocol)
        self.assertEqual(len(table), 16)
        self.assertEqual(result["first_sampled_sustained_practical_equivalence"], 5)
        self.assertEqual(result["interpretation"], "rapid_practical_washout")
        protocol, rows = self.small_rows()
        self.assertEqual(washout.analyze_rows(rows, protocol)[1]["interpretation"], "negative_gap_persists_at_50_and_100")
        protocol, rows = self.small_rows({t: -0.005 if t >= 50 else 0.01 for t in washout.spec()["offsets"]})
        self.assertEqual(washout.analyze_rows(rows, protocol)[1]["interpretation"], "inconclusive_or_nonmonotonic")

    def test_crossing_zero_is_not_equivalence(self):
        protocol, rows = self.small_rows()
        with patch.object(washout, "paired_bootstrap", return_value={"ci95_low": -0.02, "ci95_high": 0.02}):
            _, result = washout.analyze_rows(rows, protocol)
        self.assertIsNone(result["first_sampled_sustained_practical_equivalence"])
        self.assertEqual(result["interpretation"], "inconclusive_or_nonmonotonic")

    def test_write_never_overwrites_and_gate_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "result.json"
            washout.write_new(target, {"accepted": True})
            with self.assertRaises(FileExistsError):
                washout.write_new(target, {"accepted": False})
            with self.assertRaises(FileNotFoundError):
                washout.verify_step0(tmp, target)
            with self.assertRaises(FileNotFoundError):
                verify_ack(tmp, target)

    def test_bad_checkpoint_role_or_step_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "selection.json"
            washout.write_new(manifest_path, {})
            for name in ("training_state.pt", "model.safetensors"):
                (root / name).touch()
            training = {"global_step": 1501, "chunks_consumed": 1501 * 32, "run_identity": {
                "seed": 43, "branch": {"role": "predicted-good", "selection_manifest_sha256": washout.sha256_file(manifest_path),
                                        "parent_checkpoint_sha256": washout.spec()["parent_checkpoint_sha256"]},
                "mixed_partition": washout.PARTITIONS["predicted-good"], "source_commit": "test"}}
            washout.write_new(root / "training_manifest.json", training)
            manifest = {"protocol": washout.spec(), "source_commit": "test"}
            washout.validate_checkpoint(manifest, manifest_path, root, "predicted-good", 1)
            with self.assertRaisesRegex(RuntimeError, "branch manifest"):
                washout.validate_checkpoint(manifest, manifest_path, root, "predicted-bad", 1)
            with self.assertRaisesRegex(RuntimeError, "step or seed"):
                washout.validate_checkpoint(manifest, manifest_path, root, "predicted-good", 2)

    def test_figure_outputs_from_synthetic_data(self):
        from figures.gen_fig_experiment5_washout import plot
        protocol, rows = self.small_rows()
        table, _ = washout.analyze_rows(rows, protocol)
        with tempfile.TemporaryDirectory() as tmp:
            plot(table, Path(tmp))
            self.assertGreater((Path(tmp) / "fig_washout.pdf").stat().st_size, 1000)
            self.assertGreater((Path(tmp) / "fig_washout.png").stat().st_size, 1000)

    def test_snapshot_saving_does_not_change_training_rng_or_weights(self):
        from tests.test_train_resume import FakeTokenizer, ResumeCheckpointTest
        from src.training.train_scratch import save_training_checkpoint
        model, optimizer, scheduler = ResumeCheckpointTest().make_model_and_optimizer()
        model_initial = copy.deepcopy(model.state_dict())
        optimizer_initial = copy.deepcopy(optimizer.state_dict())
        scheduler_initial = copy.deepcopy(scheduler.state_dict())
        rng_initial = torch.get_rng_state().clone()
        finals = []
        for save in (False, True):
            model.load_state_dict(model_initial)
            optimizer.load_state_dict(copy.deepcopy(optimizer_initial))
            scheduler.load_state_dict(scheduler_initial)
            torch.set_rng_state(rng_initial)
            with tempfile.TemporaryDirectory() as tmp:
                for step in range(1, 6):
                    inputs = torch.randint(0, 64, (2, 8))
                    model(input_ids=inputs, labels=inputs).loss.backward()
                    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                    if save and checkpoint_due(step, 0, (1, 2, 5)):
                        save_training_checkpoint(model=model, tokenizer=FakeTokenizer(), optimizer=optimizer,
                                                 scheduler=scheduler, global_step=step, chunks_consumed=step * 2,
                                                 run_identity={}, out_dir=tmp, keep_last=1, keep_steps=(1, 2, 5),
                                                 wandb_run_id=None, elapsed_training_seconds=0)
                finals.append((copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()))
        self.assertTrue(torch.equal(finals[0][1], finals[1][1]))
        self.assertTrue(all(torch.equal(finals[0][0][key], finals[1][0][key]) for key in finals[0][0]))


if __name__ == "__main__":
    unittest.main()
