import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate.run_experiment8_controller import checkpoint_complete


class Experiment8ControllerTest(unittest.TestCase):
    def make(self, root, variant="hq8", step=2000, commit="locked"):
        checkpoint = Path(root) / variant / "step-2000"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_bytes(b"weights")
        (checkpoint / "training_state.pt").write_bytes(b"state")
        mode = "full_mh" if variant == "hq8" else "baseline"
        (checkpoint / "training_manifest.json").write_text(json.dumps({
            "global_step": step,
            "run_identity": {
                "seed": 42, "steps": 20000, "global_batch_size": 32,
                "mode": mode, "attnres_heads": 8, "num_heads": 16,
                "num_kv_heads": 8, "experiment8_variant": variant,
                "experiment8_hybrid_q_groups": 8,
                "experiment8_local_head_position": "even",
                "experiment8_global_head_position": "odd",
                "source_commit": commit,
            },
        }))

    def test_complete_hq8_and_bhq8_identities(self):
        with tempfile.TemporaryDirectory() as root:
            self.make(root, "hq8")
            self.make(root, "bhq8")
            self.assertTrue(checkpoint_complete(Path(root), "hq8", "locked"))
            self.assertTrue(checkpoint_complete(Path(root), "bhq8", "locked"))

    def test_wrong_step_or_source_is_incomplete(self):
        with tempfile.TemporaryDirectory() as root:
            self.make(root, step=1900)
            self.assertFalse(checkpoint_complete(Path(root), "hq8", "locked"))
            self.assertFalse(checkpoint_complete(Path(root), "hq8", "wrong"))

    def test_swapped_local_global_order_is_incomplete(self):
        with tempfile.TemporaryDirectory() as root:
            self.make(root)
            path = Path(root) / "hq8/step-2000/training_manifest.json"
            payload = json.loads(path.read_text())
            payload["run_identity"]["experiment8_local_head_position"] = "odd"
            path.write_text(json.dumps(payload))
            self.assertFalse(checkpoint_complete(Path(root), "hq8", "locked"))


if __name__ == "__main__":
    unittest.main()
