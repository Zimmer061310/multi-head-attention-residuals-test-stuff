#!/usr/bin/env python3
"""Generate Experiment 9 contribution and conditional alignment figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
GRAY = "#8C8C8C"


def configure() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
    })


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf")
    fig.savefig(output / f"{stem}.png", dpi=300)
    plt.close(fig)


def contribution_figure(summary: dict, output: Path) -> None:
    balanced = [
        row for row in summary["conditions"]
        if row["condition"]["kind"] == "balanced-random-8"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, split in zip(axes, ("discovery", "confirmation")):
        values = np.asarray([
            row["splits"][split]["aggregate_delta_nll"] for row in balanced
        ])
        ax.hist(values, bins=14, color=GRAY, edgecolor="white", alpha=0.85)
        local = summary["structured"]["zero-local"]["splits"][split]["aggregate_delta_nll"]
        global_ = summary["structured"]["zero-global"]["splits"][split]["aggregate_delta_nll"]
        ax.axvline(local, color=ORANGE, linewidth=2.2, label="Zero local")
        ax.axvline(global_, color=BLUE, linewidth=2.2, linestyle="--", label="Zero global")
        ax.set_title(split.capitalize())
        ax.set_xlabel(r"Ablated $-$ unchanged NLL")
        ax.set_ylabel("Balanced mask count")
    axes[1].legend(loc="best")
    fig.suptitle("Experiment 9A: structured head removal vs 70 matched masks", y=1.02)
    fig.tight_layout()
    save(fig, output, "fig_head_contribution")


def alignment_figure(summary: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, split in zip(axes, ("discovery", "confirmation")):
        values = np.asarray([
            row["splits"][split]["aggregate_delta_nll"] for row in summary["conditions"]
        ])
        ax.axvline(0, color="#222222", linewidth=1)
        ax.scatter(values, np.arange(len(values)), color=GREEN, s=18, alpha=0.85)
        ax.axvline(values.mean(), color=ORANGE, linewidth=2, label="Mean")
        ax.set_title(split.capitalize())
        ax.set_xlabel(r"Permuted $-$ aligned NLL")
        ax.set_ylabel("Frozen derangement index")
    axes[1].legend(loc="best")
    fig.suptitle("Experiment 9B: local-chunk alignment derangements", y=1.02)
    fig.tight_layout()
    save(fig, output, "fig_local_chunk_alignment")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure()
    phase_9a = json.loads(
        (args.analysis_root / "phase-9a/summary.json").read_text(encoding="utf-8")
    )
    contribution_figure(phase_9a, args.output_dir)
    phase_9b_path = args.analysis_root / "phase-9b/summary.json"
    if phase_9b_path.is_file():
        alignment_figure(json.loads(phase_9b_path.read_text(encoding="utf-8")), args.output_dir)


if __name__ == "__main__":
    main()
