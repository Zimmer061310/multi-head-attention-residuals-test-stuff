"""Check README figure data against enumeration and accepted experiment records."""

import tempfile
import unittest
from pathlib import Path

from figures.gen_fig_readme_overview import (
    partition_space, plot_partition_space, plot_seed_diagnostics, seed_diagnostics,
)


class ReadmeFigureTests(unittest.TestCase):
    def test_partition_design_counts_not_loss(self):
        data = partition_space()
        self.assertEqual(data["kind"], "exact_design_counts_not_measured_nll")
        self.assertEqual(data["partitions"], 105)
        self.assertEqual(data["retention_counts"], [1, 12, 32, 60])
        self.assertEqual(data["distance_counts"], [1, 6, 12, 20, 24, 18, 24])
        self.assertEqual(sum(data["distance_counts"]), 105)

    def test_all_seeds_and_failed_temporal_gate_preserved(self):
        rows, hashes = seed_diagnostics()
        self.assertEqual([r["seed"] for r in rows], [42, 43, 44])
        self.assertEqual(len(hashes), 6)
        self.assertTrue(all(r["signal_passed"] for r in rows))
        self.assertTrue(all(not r["temporal_passed"] for r in rows))
        self.assertGreater(rows[1]["temporal_median"], 0.5)
        self.assertLess(rows[1]["temporal_spearman"][-1], 0)
        for row in rows:
            self.assertEqual(row["adjacent_pairs"], [[1000, 1500], [1500, 2000], [2000, 3000]])
            self.assertLess(row["best_worst_ci95"][1], 0)

    def test_both_figures_export_png_and_vector_pdf(self):
        import matplotlib
        matplotlib.use("Agg")
        with tempfile.TemporaryDirectory() as directory:
            plot_partition_space(partition_space(), directory)
            plot_seed_diagnostics(seed_diagnostics()[0], directory)
            for name in ("fig_experiment1_partition_space", "fig_experiment3_gate_summary"):
                png = (Path(directory) / f"{name}.png").read_bytes()
                pdf = (Path(directory) / f"{name}.pdf").read_bytes()
                self.assertTrue(png.startswith(b"\x89PNG"))
                self.assertTrue(pdf.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
