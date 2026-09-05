#!/usr/bin/env python3
"""Generate reproducible Experiment 11 publication figures from frozen outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"s2q8": "#0072B2", "gslq8": "#D55E00", "m8": "#4D4D4D"}
MARKERS = {"s2q8": "o", "gslq8": "s"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": .18,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save(fig, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf")
    fig.savefig(output / f"{stem}.png", dpi=300)
    plt.close(fig)


def nll_curves(summary: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.55), sharey=True)
    rows = summary["curves"]
    for axis, split in zip(axes, ("discovery", "confirmation")):
        endpoint = next(row for row in rows if row["split"] == split and row["run_id"] == "m8-l100")
        for family in ("s2q8", "gslq8"):
            family_rows = sorted(
                [row for row in rows if row["split"] == split and row["family"] == family],
                key=lambda row: row["lambda"],
            ) + [endpoint]
            axis.plot(
                [row["lambda"] for row in family_rows],
                [row["nll"] for row in family_rows],
                color=COLORS[family], marker=MARKERS[family],
                label=family.upper(),
            )
        axis.set_title(split.capitalize())
        axis.set_xlabel(r"Training bias $\lambda$")
        axis.set_xticks([0, .1, .25, .5, 1])
    axes[0].set_ylabel("Token-weighted NLL")
    axes[1].legend()
    save(fig, output, "fig_nll_curves")


def selected_contrasts(summary: dict, output: Path) -> None:
    rows = []
    for family in ("s2q8", "gslq8"):
        for row in summary["families"][family]["confirmation_nll"]:
            rows.append((family, row))
    values = np.asarray([row[1]["aggregate_delta_nll"] for row in rows])
    low = values - np.asarray([row[1]["ci95_low"] for row in rows])
    high = np.asarray([row[1]["ci95_high"] for row in rows]) - values
    labels = [f"{family.upper()}\n{row['id'].replace('selected-minus-', 'vs ')}" for family, row in rows]
    fig, ax = plt.subplots(figsize=(5.8, 2.65))
    x = np.arange(len(rows))
    ax.errorbar(
        x, values, yerr=np.vstack([low, high]), fmt="none",
        ecolor="#333333", capsize=4, linewidth=1.2,
    )
    ax.scatter(x, values, c=[COLORS[family] for family, _ in rows], s=38, zorder=3)
    ax.axhline(0, color="#222222", linewidth=.9)
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"Confirmation $\Delta$NLL")
    save(fig, output, "fig_selected_contrasts")


def softness_curves(summary: dict, output: Path) -> None:
    rows = [row for row in summary["trajectories"] if row["milestone"] == 2000]
    endpoint = next(row for row in rows if row["run_id"] == "m8-l100")
    metrics = (("r_weight", r"$R_{weight}$"),
               ("r_act", r"$R_{act}$"),
               ("theta_radians", r"$\theta$ (radians)"))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for axis, (metric, label) in zip(axes, metrics):
        for family in ("s2q8", "gslq8"):
            family_rows = sorted(
                [row for row in rows if row["family"] == family],
                key=lambda row: row["lambda"],
            ) + [endpoint]
            axis.plot(
                [row["lambda"] for row in family_rows],
                [row[metric] for row in family_rows],
                color=COLORS[family], marker=MARKERS[family], label=family.upper(),
            )
        axis.set_xlabel(r"$\lambda$")
        axis.set_ylabel(label)
        axis.set_xticks([0, .25, .5, 1])
    axes[-1].legend()
    save(fig, output, "fig_softness_curves")


def trajectories(summary: dict, output: Path) -> None:
    metrics = (("r_weight", r"$R_{weight}$"),
               ("r_act", r"$R_{act}$"),
               ("theta_radians", r"$\theta$"))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    selected = {
        summary["families"][family]["selected_run_id"]: family
        for family in ("s2q8", "gslq8")
    }
    selected["m8-l100"] = "m8"
    for run_id, family in selected.items():
        rows = sorted(
            [row for row in summary["trajectories"] if row["run_id"] == run_id],
            key=lambda row: row["milestone"],
        )
        for axis, (metric, label) in zip(axes, metrics):
            metric_rows = rows
            if metric == "r_weight":
                metric_rows = sorted(
                    [row for row in summary.get("weight_trajectories", rows)
                     if row["run_id"] == run_id],
                    key=lambda row: row["milestone"],
                )
            axis.plot(
                [row["milestone"] for row in metric_rows],
                [row[metric] for row in metric_rows],
                color=COLORS[family], marker="o", label=run_id,
            )
            axis.set_xlabel("Optimizer step")
            axis.set_ylabel(label)
    axes[-1].legend(fontsize=6.5)
    save(fig, output, "fig_softness_trajectories")


def selected_heatmap(summary: dict, results_root: Path, output: Path, family: str) -> None:
    run_id = summary["families"][family]["selected_run_id"]
    probe = json.loads((results_root / "probes" / run_id / "step-2000-confirmation.json").read_text())
    metrics = probe["activation_metrics"]["metrics"]
    heads = probe["activation_metrics"]["diagnostic_head_positions"]
    r_act = np.asarray(metrics["r_act"]["layer_group_head"], dtype=float)[..., heads]
    theta = np.asarray(metrics["theta_radians"]["layer_group_head"], dtype=float)[..., heads]
    r_act = np.nanmean(r_act, axis=-1)
    theta = np.nanmean(theta, axis=-1)
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 4.15), constrained_layout=True)
    for axis, matrix, title, cmap in (
        (axes[0], r_act, r"$R_{act}$", "viridis"),
        (axes[1], theta, r"$\theta$ (radians)", "magma"),
    ):
        image = axis.imshow(matrix, aspect="auto", origin="lower", cmap=cmap)
        axis.set_title(title)
        axis.set_xlabel("MHAR / GQA group")
        axis.set_ylabel("Layer")
        axis.set_xticks(range(8))
        fig.colorbar(image, ax=axis, shrink=.8)
    fig.suptitle(f"{run_id}: confirmation effective specialization", fontsize=10)
    save(fig, output, f"fig_{family}_heatmap")


def parameter_table(summary: dict, output: Path) -> None:
    rows = summary["systems"]
    fields = [
        "run_id", "trainable_parameters", "qkv_parameters", "qkv_macs_per_token",
        "training_seconds", "confirmation_tokens_per_second",
        "confirmation_peak_memory_bytes",
    ]
    with (output / "systems_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    nll_curves(summary, args.output_dir)
    selected_contrasts(summary, args.output_dir)
    softness_curves(summary, args.output_dir)
    trajectories(summary, args.output_dir)
    selected_heatmap(summary, args.results_root, args.output_dir, "s2q8")
    selected_heatmap(summary, args.results_root, args.output_dir, "gslq8")
    parameter_table(summary, args.output_dir)


if __name__ == "__main__":
    main()
