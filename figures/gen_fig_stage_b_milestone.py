#!/usr/bin/env python3
"""Plot paired Stage B milestone deltas from an analysis summary."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "h16": "H16",
    "h8": "H8",
    "h4": "H4",
    "mixed-k2": "Mixed k=2",
    "mixed-k3": "Mixed k=3",
    "mixed-k4-best": "Mixed k=4 best",
    "mixed-k5": "Mixed k=5",
    "mixed-k4-worst": "Mixed k=4 worst",
}


def plot_summary(summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    confirmation = summary["confirmation_ranking"]
    order = [row["variant"] for row in confirmation]
    discovery = {row["variant"]: row for row in summary["discovery_ranking"]}
    confirmation_by_variant = {row["variant"]: row for row in confirmation}
    contrasts = {
        row["candidate"]: row
        for row in summary["contrasts"]
        if row["reference"] == "h8"
    }
    scale = 1000.0
    y = np.arange(len(order))
    discovery_delta = [discovery[v]["delta_nll_vs_h8"] * scale for v in order]
    confirmation_delta = [
        confirmation_by_variant[v]["delta_nll_vs_h8"] * scale for v in order
    ]
    low = []
    high = []
    for variant, value in zip(order, confirmation_delta):
        if variant == "h8":
            low.append(0.0)
            high.append(0.0)
            continue
        contrast = contrasts[variant]
        low.append(value - contrast["ci95_low"] * scale)
        high.append(contrast["ci95_high"] * scale - value)

    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "savefig.bbox": "tight",
    }):
        fig, axis = plt.subplots(figsize=(6.75, 3.7))
        height = 0.34
        axis.barh(
            y - height / 2, discovery_delta, height,
            color="#56B4E9", label="Discovery")
        axis.barh(
            y + height / 2, confirmation_delta, height,
            xerr=np.array([low, high]), capsize=2.5,
            color="#D55E00", label="Untouched confirmation (95% paired CI)")
        axis.axvline(0.0, color="#2E3440", linewidth=0.9)
        axis.set_yticks(y, [LABELS[value] for value in order])
        axis.invert_yaxis()
        axis.set_xlabel(r"$\Delta$ token-weighted NLL vs H8 (millinats/token; lower is better)")
        axis.set_title(
            f"Stage B architecture screen at step {summary['milestone']}", loc="left")
        axis.text(
            0.0, 1.02,
            f"Discovery/confirmation Spearman: {summary['rank_spearman']:+.3f}",
            transform=axis.transAxes, fontsize=8.3, color="#4B5563")
        axis.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_dir / "fig_stage_b_milestone.png", dpi=300)
        fig.savefig(output_dir / "fig_stage_b_milestone.pdf")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    plot_summary(summary, args.output_dir)


if __name__ == "__main__":
    main()
