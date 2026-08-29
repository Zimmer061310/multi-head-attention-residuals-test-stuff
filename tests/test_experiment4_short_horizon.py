import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.experiment4_short_horizon import moving_block_ci


class Experiment4ShortHorizonTest(unittest.TestCase):
    def test_moving_block_ci_preserves_constant_difference(self):
        self.assertEqual(moving_block_ci([-0.25] * 10, samples=100, seed=1), [-0.25, -0.25])

    def test_moving_block_ci_rejects_empty_input(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            moving_block_ci([], samples=10)


if __name__ == "__main__":
    unittest.main()
