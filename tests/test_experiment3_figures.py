import csv
import json
import tempfile
import unittest
from pathlib import Path

from figures.gen_fig_experiment3 import (
    plot_actionability,
    plot_cross_seed,
    plot_landscape,
    plot_score_trajectories,
    plot_signal,
    plot_temporal,
    plot_training_curves,
)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


class Experiment3FigureTests(unittest.TestCase):
    def test_all_planned_figures_render_png_pdf_and_caption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            signal = root / "signal.csv"
            write_csv(signal, [{
                "candidate_id": f"remove-{i}", "boundary": i,
                "discovery_delta_nll": (i - 7) / 10000,
                "confirmation_delta_nll": (i - 6) / 10000,
            } for i in range(15)])

            temporal = root / "temporal.csv"
            write_csv(temporal, [{
                "split": split, "step_t": a, "step_u": b, "step_gap": b - a,
                "spearman": 0.6 + index * 0.03, "pearson": 0.7,
                "top_three_overlap": 2, "top_three_jaccard": 0.5,
                "median_side_agreement": 0.8, "selected_at_t": "remove-1",
                "selected_future_delta_nll": -0.001, "future_best_delta_nll": -0.002,
                "future_regret": 0.001,
            } for split in ("discovery", "confirmation")
              for index, (a, b) in enumerate(((1000, 1500), (1500, 2000), (2000, 3000)))])

            action = root / "action.csv"
            write_csv(action, [{
                "split": "confirmation", "candidate": "predicted-good",
                "reference": reference, "mean_delta_nll": -0.002 + index * 0.0005,
                "ci95_low": -0.003 + index * 0.0005,
                "ci95_high": -0.001 + index * 0.0005,
                "bootstrap_samples": 100, "bootstrap_seed": 1,
            } for index, reference in enumerate(("random", "predicted-bad", "unchanged"))])

            cross_seed = root / "cross_seed.csv"
            write_csv(cross_seed, [{
                "seed": seed, "good_minus_random": -0.002 + index * 0.0002,
                "good_minus_random_ci95_low": -0.003 + index * 0.0002,
                "good_minus_random_ci95_high": -0.001 + index * 0.0002,
                "good_boundary": index + 1,
                "random_boundary": index + 6,
                "bad_boundary": index + 11,
            } for index, seed in enumerate((42, 43, 44))])

            landscape = root / "landscape.csv"
            write_csv(landscape, [{
                "split": split, "boundary_index": boundary, "offset": offset,
                "candidate_id": f"b{boundary}-{offset}",
                "delta_nll": (offset / 40) ** 2 / 1000,
                "nll": 3.0 + (offset / 40) ** 2 / 1000,
            } for split in ("discovery", "confirmation") for boundary in range(1, 8)
              for offset in range(-40, 41, 10)])
            metrics = root / "landscape_metrics.csv"
            write_csv(metrics, [{
                "boundary_index": boundary,
                "discovery_normalized_roughness": 0.1 + boundary / 100,
                "confirmation_normalized_roughness": 0.12 + boundary / 100,
            } for boundary in range(1, 8)])
            training = root / "training.jsonl"
            training.write_text("".join(
                json.dumps({
                    "branch_role": role, "step": step,
                    "loss": 3.0 - step / 100000 + index / 1000,
                }) + "\n"
                for index, role in enumerate(
                    ("predicted-good", "predicted-bad", "random", "unchanged"))
                for step in (1600, 1700, 1800, 1900, 2000)
            ), encoding="utf-8")
            trajectories = root / "trajectories.csv"
            write_csv(trajectories, [{
                "split": split, "step": step, "candidate_id": f"remove-{boundary:02d}",
                "boundary": boundary, "delta_nll": (boundary - 7 + step / 10000) / 1000,
            } for split in ("discovery", "confirmation")
              for step in (1000, 1500, 2000, 3000) for boundary in range(15)])

            outputs = []
            outputs += plot_signal(signal, root)
            outputs += plot_temporal(temporal, root)
            outputs += plot_score_trajectories(trajectories, root)
            outputs += plot_actionability(action, root)
            outputs += plot_training_curves(training, root)
            outputs += plot_cross_seed(cross_seed, root)
            outputs += plot_landscape(landscape, root, metrics)
            self.assertEqual(len(outputs), 27)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in outputs))


if __name__ == "__main__":
    unittest.main()
