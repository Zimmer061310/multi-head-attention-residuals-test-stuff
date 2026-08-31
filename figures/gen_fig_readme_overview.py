"""README figures from exact combinatorics and accepted Experiment 3 summaries.

Run from the repository root: python3 -m figures.gen_fig_readme_overview
This does not evaluate models, change gates, or write into accepted results.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
COLORS = ("#0072B2", "#D55E00", "#009E73")


def partition_space():
    from src.attention_residuals.mhar_partition import (
        coordinate_distance, generate_pair_partitions, original_pair_retention,
    )
    partitions = generate_pair_partitions(8)
    retention = Counter(original_pair_retention(p)[0] for p in partitions)
    distances = Counter(coordinate_distance(p)[0] for p in partitions)
    return {
        "kind": "exact_design_counts_not_measured_nll",
        "partitions": len(partitions),
        "retention_percent": [100, 50, 25, 0],
        "retention_counts": [retention[k] for k in (4, 2, 1, 0)],
        "mean_pair_distance": [k / 4 for k in sorted(distances)],
        "distance_counts": [distances[k] for k in sorted(distances)],
    }


def seed_diagnostics(root=ROOT):
    rows, hashes = [], {}
    for seed in (42, 43, 44):
        paths = [Path(root) / f"results/experiment3/seed-{seed}/{stage}/{stage}_summary.json"
                 for stage in ("signal", "temporal")]
        signal, temporal = [json.loads(p.read_text()) for p in paths]
        for p in paths:
            hashes[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
        if signal["seed"] != seed or temporal["seed"] != seed or signal["step"] != 1500:
            raise ValueError("Unexpected seed or signal checkpoint")
        gate = temporal["gate_definition"]
        pairs = gate["adjacent_pairs"]
        primary = temporal["primary_discovery_spearman"]
        if len(primary) != len(pairs) or len(pairs) != 3:
            raise ValueError("Expected three preregistered adjacent checkpoint pairs")
        if abs(median(primary) - temporal["median_primary_discovery_spearman"]) > 1e-12:
            raise ValueError("Inconsistent recorded temporal median")
        expected = (all(v > 0 for v in primary)
                    and median(primary) >= gate["minimum_median_discovery_spearman"]
                    and temporal["confirmation_same_sign_pairs"] >= gate["minimum_confirmation_same_sign_pairs"])
        if expected != temporal["stability_gate_passed"]:
            raise ValueError("Recorded temporal gate disagrees with its frozen rules")
        rows.append({
            "seed": seed,
            "signal_spearman": signal["discovery_confirmation_spearman"],
            "signal_passed": signal["signal_gate_passed"],
            "signal_gate": signal["gate_definition"],
            "best_worst_ci95": [signal["best_minus_worst_confirmation"][k]
                                for k in ("ci95_low", "ci95_high")],
            "adjacent_pairs": pairs,
            "temporal_spearman": primary,
            "temporal_median": temporal["median_primary_discovery_spearman"],
            "temporal_passed": temporal["stability_gate_passed"],
            "temporal_gate": gate,
        })
    return rows, hashes


def plot_partition_space(data, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    panels = (
        (axes[0], data["retention_percent"], data["retention_counts"],
         "Original-pair retention", "Reference pairs retained (%)"),
        (axes[1], data["mean_pair_distance"], data["distance_counts"],
         "Coordinate distance", "Mean distance between paired atoms"),
    )
    for ax, x, counts, title, xlabel in panels:
        bars = ax.bar(range(len(x)), counts, color=COLORS[0], width=0.65)
        ax.bar_label(bars, padding=3, fontsize=10)
        ax.set_xticks(range(len(x)), [str(v).removesuffix(".0") for v in x])
        ax.set(xlabel=xlabel, ylabel="Number of partitions", title=title,
               ylim=(0, max(counts) * 1.22))
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle("Experiment 1: exact counts across all 105 pairings", fontsize=15)
    fig.text(0.5, 0.02, "Design-space counts only — not measured NLL. Each panel totals 105 partitions.",
             ha="center", fontsize=10, color="#4B5563")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    save(fig, output, "fig_experiment1_partition_space")


def plot_seed_diagnostics(rows, output):
    import matplotlib.pyplot as plt
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.55]})
    values = [r["signal_spearman"] for r in rows]
    bars = left.bar(range(3), values, color=COLORS, width=0.55)
    left.bar_label(bars, labels=[f"{r['signal_spearman']:.4f}\n{'PASS' if r['signal_passed'] else 'FAIL'}"
                                for r in rows], padding=5, fontsize=9)
    left.set_xticks(range(3), [f"Seed {r['seed']}" for r in rows])
    left.set(ylim=(0, 1.15), ylabel="Discovery / confirmation Spearman",
             title=f"Signal at step 1,500: {sum(r['signal_passed'] for r in rows)}/{len(rows)} pass")
    left.axhline(0.5, color="#666666", linestyle="--", linewidth=1)
    left.text(0.02, 0.52, "0.5 rank-agreement requirement", transform=left.get_yaxis_transform(), fontsize=8)
    for r, color, marker in zip(rows, COLORS, ("o", "s", "^")):
        right.plot(range(3), r["temporal_spearman"], color=color, marker=marker,
                   label=f"Seed {r['seed']} · median {r['temporal_median']:.3f} · {'PASS' if r['temporal_passed'] else 'FAIL'}")
    pairs = rows[0]["adjacent_pairs"]
    right.set_xticks(range(3), [f"{a:,}→{b:,}" for a, b in pairs])
    right.set(ylim=(-0.3, 1.0), xlim=(-0.15, 2.15),
              ylabel="Adjacent-checkpoint discovery Spearman",
              xlabel="Checkpoint pair (last interval spans 1,000 updates)",
              title=f"Temporal stability: {sum(r['temporal_passed'] for r in rows)}/{len(rows)} pass")
    right.axhline(0, color="#333333", linewidth=1)
    right.axhline(0.5, color="#666666", linestyle="--", linewidth=1)
    right.text(0.02, 0.51, "0.5 required median (not a per-pair cutoff)",
               transform=right.get_yaxis_transform(), fontsize=8)
    right.legend(loc="upper right", fontsize=8)
    for ax in (left, right):
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle("Experiment 3: reliable snapshot rankings do not guarantee temporal stability", fontsize=14)
    fig.text(0.5, 0.055, "Signal also requires a negative best−worst confirmation CI. Temporal gate requires median ≥0.5,",
             ha="center", fontsize=9, color="#4B5563")
    fig.text(0.5, 0.015, "all adjacent discovery correlations >0, and ≥2 confirmation sign agreements. No 3C branches were launched.",
             ha="center", fontsize=9, color="#4B5563")
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    save(fig, output, "fig_experiment3_gate_summary")


def save(fig, output, name):
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", dpi=300)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures/readme")
    args = parser.parse_args()
    design = partition_space()
    seeds, hashes = seed_diagnostics()
    with plt.rc_context({"font.family": "serif", "font.size": 10,
                         "axes.titlesize": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "lines.linewidth": 1.8, "lines.markersize": 6}):
        plot_partition_space(design, args.output_dir)
        plot_seed_diagnostics(seeds, args.output_dir)
    sources = {"experiment1": design, "experiment3": seeds, "source_sha256": hashes}
    partition_source = ROOT / "src/attention_residuals/mhar_partition.py"
    sources["source_sha256"][str(partition_source.relative_to(ROOT))] = hashlib.sha256(partition_source.read_bytes()).hexdigest()
    (args.output_dir / "figure_data.json").write_text(json.dumps(sources, indent=2) + "\n")


if __name__ == "__main__":
    main()
