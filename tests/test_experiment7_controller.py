import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate.run_experiment7_controller import checkpoint_complete


class Experiment7ControllerTest(unittest.TestCase):
    def make(self, root, variant="lq8", step=2000, commit="locked"):
        checkpoint = Path(root) / variant / "step-2000"; checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_bytes(b"weights")
        (checkpoint / "training_state.pt").write_bytes(b"state")
        (checkpoint / "training_manifest.json").write_text(json.dumps({
            "global_step": step, "run_identity": {"seed": 42, "steps": 20000,
            "global_batch_size": 32, "experiment7_variant": variant,
            "experiment7_local_q_groups": 4 if variant.endswith("4") else 8,
            "source_commit": commit}}))

    def test_complete_identity(self):
        with tempfile.TemporaryDirectory() as root:
            self.make(root)
            self.assertTrue(checkpoint_complete(Path(root), "lq8", "locked"))
            self.assertFalse(checkpoint_complete(Path(root), "lq8", "wrong"))

    def test_wrong_step(self):
        with tempfile.TemporaryDirectory() as root:
            self.make(root, step=1900)
            self.assertFalse(checkpoint_complete(Path(root), "lq8", "locked"))


if __name__ == "__main__": unittest.main()
