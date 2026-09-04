#!/usr/bin/env python3
"""Generate publication figures for Experiment 8 from committed result files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "experiment8" / "step-2000" / "analysis"

RESULT_FILES = {
    "M8": ROOT
    / "results/experiment2/stage-b-screening/step-2000/h8/result.json",
    "HQ8": ROOT / "results/experiment8/step-2000/hq8/evaluation/result.json",
    "LQ8": ROOT / "results/experiment7/step-2000/lq8/evaluation/result.json",
    "B": ROOT / "results/experiment6/step-2000/b/evaluation/result.json",
    "BHQ8": ROOT / "results/experiment8/step-2000/bhq8/evaluation/result.json",
    "BLQ8": ROOT / "results/experiment7/step-2000/blq8/evaluation/result.json",
}

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
GRAY = "#6B7280"


def load_nlls() -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for model, path in RESULT_FILES.items():
        with path.open() as handle:
            data = json.load(handle)
        values[model] = {
            split: float(data["splits"][split]["nll"])
            for split in ("discovery", "confirmation")
        }
    return values


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_nll_progression(values: dict[str, dict[str, float]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    x = [0, 50, 100]
    panels = [
        ("MHAR residual", ["M8", "HQ8", "LQ8"]),
        ("Baseline residual", ["B", "BHQ8", "BLQ8"]),
    ]
    styles = {
        "discovery": (BLUE, "o", "Discovery"),
        "confirmation": (ORANGE, "s", "Confirmation"),
    }

    for ax, (title, models) in zip(axes, panels):
        for split, (color, marker, label) in styles.items():
            y = [values[model][split] for model in models]
            ax.plot(x, y, color=color, marker=marker, linewidth=2, label=label)
            for xx, yy, model in zip(x, y, models):
                ax.annotate(
                    model,
                    (xx, yy),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=color,
                )
        ax.set_title(title)
        ax.set_xlabel("Local query heads within each GQA group (%)")
        ax.set_xticks(x)
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.8)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axes[0].set_ylabel("Token-weighted held-out NLL (lower is better)")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Experiment 8: hybrid query access is the midpoint between global and local Q", y=1.03)
    fig.tight_layout()
    save(fig, "fig_hybrid_q_nll")


def plot_contrasts() -> None:
    wanted = [
        ("hq8-minus-m8", "HQ8 − M8", BLUE),
        ("bhq8-minus-b", "BHQ8 − B", GRAY),
        ("h8-interaction", "H8 interaction", GREEN),
        ("hq8-minus-lq8", "HQ8 − LQ8", ORANGE),
        ("bhq8-minus-blq8", "BHQ8 − BLQ8", GRAY),
    ]
    with (OUT / "contrasts.csv").open(newline="") as handle:
        rows = {
            row["contrast"]: row
            for row in csv.DictReader(handle)
            if row["split"] == "confirmation"
        }

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for y, (key, label, color) in enumerate(reversed(wanted)):
        row = rows[key]
        delta = float(row["delta_nll"])
        low = float(row["ci95_low"])
        high = float(row["ci95_high"])
        ax.errorbar(
            delta,
            y,
            xerr=[[delta - low], [high - delta]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2,
            capsize=3,
            markersize=6,
        )
        ax.annotate(f"{delta:+.4f}", (high, y), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8)

    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_yticks(range(len(wanted)), [item[1] for item in reversed(wanted)])
    ax.set_xlabel("Confirmation delta NLL (negative favors first named model)")
    ax.set_title("Experiment 8 fixed-validation contrasts (paired 95% CI)")
    ax.grid(axis="x", color="#D1D5DB", linewidth=0.7, alpha=0.8)
    fig.tight_layout()
    save(fig, "fig_hybrid_q_contrasts")


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    values = load_nlls()
    plot_nll_progression(values)
    plot_contrasts()


if __name__ == "__main__":
    main()
