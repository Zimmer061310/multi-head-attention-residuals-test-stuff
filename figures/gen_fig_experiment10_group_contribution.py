#!/usr/bin/env python3
"""Generate publication figures for Experiment 10."""

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
    groups = np.arange(8)
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, split in zip(axes, ("discovery", "confirmation")):
        vectors = summary["vectors"][split]
        ax.axhline(0, color="#222222", linewidth=0.8)
        ax.bar(groups - width, vectors["D_L"], width, color=ORANGE, label=r"$D_L$")
        ax.bar(groups, vectors["D_G"], width, color=BLUE, label=r"$D_G$")
        ax.bar(groups + width, vectors["D_GL"], width, color=GREEN, label=r"$D_{GL}$")
        ax.set_title(split.capitalize())
        ax.set_xlabel("GQA group")
        ax.set_xticks(groups)
        ax.set_ylabel(r"Ablated $-$ HQ8 NLL")
    axes[1].legend(loc="best")
    fig.suptitle("Experiment 10: per-group query-head contribution", y=1.02)
    fig.tight_layout()
    save(fig, output, "fig_group_contribution")


def distribution_figure(summary: dict, output: Path) -> None:
    groups = np.arange(8)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for split, color in (("discovery", BLUE), ("confirmation", ORANGE)):
        axes[0].plot(groups, summary["vectors"][split]["D_L"], marker="o", color=color, label=split)
        axes[1].plot(groups, summary["vectors"][split]["interaction"], marker="o", color=color, label=split)
    for ax in axes:
        ax.axhline(0, color="#222222", linewidth=0.8)
        ax.set_xlabel("GQA group")
        ax.set_xticks(groups)
    axes[0].set_title("Local-head damage")
    axes[0].set_ylabel(r"$D_L$ NLL")
    axes[1].set_title("Local/global interaction")
    axes[1].set_ylabel(r"$D_{GL}-D_L-D_G$ NLL")
    axes[1].legend(loc="best")
    fig.tight_layout()
    save(fig, output, "fig_local_distribution")


def alignment_figure(summary: dict, output: Path) -> None:
    matrix = np.full((8, 8), np.nan)
    for row in summary["conditions"]:
        condition = row["condition"]
        matrix[condition["target_group"], condition["source_chunk"]] = row["splits"]["confirmation"]["aggregate_delta_nll"]
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xlabel("Incorrect source chunk")
    ax.set_ylabel("Target local-Q group")
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_title("Experiment 10D: one-group misalignment damage")
    fig.colorbar(image, ax=ax, label=r"Misaligned $-$ HQ8 NLL")
    fig.tight_layout()
    save(fig, output, "fig_group_alignment")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure()
    primary = json.loads((args.analysis_root / "phase-10abc/summary.json").read_text())
    contribution_figure(primary, args.output_dir)
    distribution_figure(primary, args.output_dir)
    alignment = args.analysis_root / "phase-10d/summary.json"
    if alignment.is_file():
        alignment_figure(json.loads(alignment.read_text()), args.output_dir)


if __name__ == "__main__":
    main()
