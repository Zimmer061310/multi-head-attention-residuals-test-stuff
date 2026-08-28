#!/usr/bin/env python3
"""Publication-ready, data-faithful figures for Experiment 3 analyses."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2563EB"
ORANGE = "#EA580C"
GREEN = "#059669"
GRAY = "#6B7280"


def _rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig, output_dir, stem, caption):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    caption_path = output_dir / f"{stem}.caption.txt"
    caption_path.write_text(caption.strip() + "\n", encoding="utf-8")
    return [png, pdf, caption_path]


def plot_signal(csv_path, output_dir):
    _style()
    rows = sorted(_rows(csv_path), key=lambda row: int(row["boundary"]))
    x = [int(row["boundary"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.plot(x, [float(row["discovery_delta_nll"]) * 1000 for row in rows],
            marker="o", color=BLUE, label="Discovery")
    ax.plot(x, [float(row["confirmation_delta_nll"]) * 1000 for row in rows],
            marker="s", color=ORANGE, label="Confirmation")
    ax.set(xlabel="Removed H16 boundary (between atoms i and i+1)",
           ylabel=r"$\Delta$NLL vs native H16 (millinats/token)", xticks=x)
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    return _save(fig, output_dir, "fig_boundary_signal_step1500",
                 "Boundary-removal NLL relative to native H16 on the locked discovery and confirmation splits. Lower values are better; the figure measures frozen preference, not future training benefit.")


def plot_temporal(csv_path, output_dir):
    _style()
    rows = _rows(csv_path)
    steps = sorted({int(r["step_t"]) for r in rows} | {int(r["step_u"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    image = None
    for ax, split in zip(axes, ("discovery", "confirmation")):
        matrix = np.eye(len(steps))
        for row in rows:
            if row["split"] != split:
                continue
            i, j = steps.index(int(row["step_t"])), steps.index(int(row["step_u"]))
            matrix[i, j] = matrix[j, i] = float(row["spearman"])
        image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set(xticks=range(len(steps)), yticks=range(len(steps)),
               xticklabels=steps, yticklabels=steps, xlabel="Step", ylabel="Step")
        ax.text(0.04, 0.93, split.capitalize(), transform=ax.transAxes,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
    fig.colorbar(image, ax=axes, label="Spearman rank correlation", shrink=0.8)
    return _save(fig, output_dir, "fig_temporal_stability_matrix",
                 "Rank correlation of the 15 boundary-removal scores between checkpoints. Positive correlation means a measurement retains directional information; it does not establish actionability.")


def plot_score_trajectories(csv_path, output_dir):
    _style()
    rows = _rows(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    for ax, split in zip(axes, ("discovery", "confirmation")):
        subset = [row for row in rows if row["split"] == split]
        boundaries = sorted({int(row["boundary"]) for row in subset})
        for boundary in boundaries:
            points = sorted((row for row in subset if int(row["boundary"]) == boundary),
                            key=lambda row: int(row["step"]))
            ax.plot([int(row["step"]) for row in points],
                    [float(row["delta_nll"]) * 1000 for row in points],
                    linewidth=0.9, alpha=0.75)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set(xlabel="Training step", ylabel=r"$\Delta$NLL (millinats/token)")
        ax.text(0.04, 0.92, split.capitalize(), transform=ax.transAxes)
        ax.grid(alpha=0.15)
    return _save(fig, output_dir, "fig_boundary_score_trajectories",
                 "Trajectories of all 15 native-H16 single-boundary-removal scores. Lines track boundary indices within a split; crossings reveal preference drift and do not imply causal training benefit.")


def plot_actionability(csv_path, output_dir):
    _style()
    rows = [row for row in _rows(csv_path) if row["split"] == "confirmation"]
    labels = [f"{r['candidate']} −\n{r['reference']}" for r in rows]
    means = np.asarray([float(r["mean_delta_nll"]) * 1000 for r in rows])
    lows = np.asarray([float(r["ci95_low"]) * 1000 for r in rows])
    highs = np.asarray([float(r["ci95_high"]) * 1000 for r in rows])
    colors = [GREEN if value < 0 else ORANGE for value in means]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.errorbar(np.arange(len(rows)), means, yerr=[means - lows, highs - means],
                fmt="none", ecolor=GRAY, capsize=4, linewidth=1.2)
    ax.scatter(np.arange(len(rows)), means, c=colors, s=45, zorder=3)
    ax.set(xlabel="Branched-training contrast", ylabel=r"Future $\Delta$NLL (millinats/token)",
           xticks=np.arange(len(rows)), xticklabels=labels)
    ax.grid(axis="y", alpha=0.2)
    return _save(fig, output_dir, "fig_actionability_future_nll",
                 "Paired confirmation NLL contrasts after matched continuation training. Points are sequence-paired means and bars are 95% bootstrap intervals; negative predicted-good minus random is the primary actionability direction.")


def plot_training_curves(jsonl_path, output_dir):
    _style()
    rows = [json.loads(line) for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    for role in sorted({row["branch_role"] for row in rows}):
        points = sorted((row for row in rows if row["branch_role"] == role),
                        key=lambda row: int(row["step"]))
        ax.plot([row["step"] for row in points], [row["loss"] for row in points],
                label=role)
    ax.set(xlabel="Training step", ylabel="Training NLL (logged-window mean)")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    return _save(fig, output_dir, "fig_actionability_training_curves",
                 "Logged training-loss trajectories after branching from the identical step-1,500 state. These noisy online losses diagnose optimization behavior; the fixed confirmation NLL is the outcome metric.")


def plot_cross_seed(csv_path, output_dir):
    _style()
    rows = sorted(_rows(csv_path), key=lambda row: int(row["seed"]))
    seeds = [row["seed"] for row in rows]
    means = np.asarray([float(row["good_minus_random"]) * 1000 for row in rows])
    lows = np.asarray([float(row["good_minus_random_ci95_low"]) * 1000 for row in rows])
    highs = np.asarray([float(row["good_minus_random_ci95_high"]) * 1000 for row in rows])
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.errorbar(np.arange(len(rows)), means, yerr=[means - lows, highs - means],
                fmt="o", color=BLUE, ecolor=GRAY, capsize=4)
    ax.set(xlabel="Independent training seed", ylabel=r"Good − random future NLL (millinats/token)",
           xticks=np.arange(len(rows)), xticklabels=seeds)
    ax.grid(axis="y", alpha=0.2)
    outputs = _save(fig, output_dir, "fig_cross_seed_actionability",
                    "Within-seed predicted-good minus random future NLL on confirmation data. Boundary identities may differ across seeds; intervals resample sequences within each seed and are not seed-level significance tests.")
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    roles = ("good_boundary", "random_boundary", "bad_boundary")
    colors = (GREEN, GRAY, ORANGE)
    x = np.arange(len(rows))
    for offset, (role, color) in enumerate(zip(roles, colors)):
        values = [int(row[role]) for row in rows]
        ax.scatter(x + (offset - 1) * 0.12, values, label=role.replace("_boundary", ""),
                   color=color, s=45)
    ax.set(xlabel="Independent training seed", ylabel="Selected H16 boundary index",
           xticks=x, xticklabels=seeds, yticks=range(15), ylim=(-0.7, 14.7))
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.15)
    outputs += _save(fig, output_dir, "fig_cross_seed_boundary_ids",
                     "Descriptive seed-local good, random, and bad boundary IDs. Boundary-ID agreement is not a success criterion and this panel must not be read as evidence for universal coordinates.")
    return outputs


def plot_landscape(csv_path, output_dir, metrics_csv_path=None):
    _style()
    rows = _rows(csv_path)
    fig, axes = plt.subplots(2, 4, figsize=(9.0, 5.3), sharex=True, sharey=True)
    for boundary, ax in zip(range(1, 8), axes.flat):
        subset = [row for row in rows if int(row["boundary_index"]) == boundary]
        for split, color, marker in (("discovery", BLUE, "o"), ("confirmation", ORANGE, "s")):
            points = sorted((row for row in subset if row["split"] == split),
                            key=lambda row: int(row["offset"]))
            ax.plot([int(row["offset"]) for row in points],
                    [float(row["delta_nll"]) * 1000 for row in points],
                    marker=marker, color=color, label=split.capitalize())
        ax.axhline(0, color="black", linewidth=0.6)
        ax.axvline(0, color=GRAY, linewidth=0.6, linestyle="--")
        ax.text(0.04, 0.92, f"Boundary {boundary}", transform=ax.transAxes)
        ax.grid(alpha=0.15)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower right", bbox_to_anchor=(0.96, 0.11))
    fig.supxlabel("Boundary displacement (residual coordinates)")
    fig.supylabel(r"$\Delta$NLL vs native H8 (millinats/token)")
    fig.subplots_adjust(wspace=0.12, hspace=0.15)
    outputs = _save(fig, output_dir, "fig_boundary_landscape_small_multiples",
                    "Local H8 boundary-displacement landscapes. Each panel moves one boundary while preserving all other boundaries; lower is better. Smoothness supports a parameterization choice but does not itself prove future training gains.")
    if metrics_csv_path is not None:
        metrics = sorted(_rows(metrics_csv_path), key=lambda row: int(row["boundary_index"]))
        x = np.arange(len(metrics))
        width = 0.36
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        ax.bar(x - width / 2,
               [float(row["discovery_normalized_roughness"]) for row in metrics],
               width, label="Discovery", color=BLUE)
        ax.bar(x + width / 2,
               [float(row["confirmation_normalized_roughness"]) for row in metrics],
               width, label="Confirmation", color=ORANGE)
        ax.axhline(0.25, color=GRAY, linestyle="--", linewidth=1,
                   label="Preregistered threshold")
        ax.set(xlabel="H8 boundary index", ylabel="Normalized roughness",
               xticks=x, xticklabels=[row["boundary_index"] for row in metrics])
        ax.legend(frameon=False, ncol=3)
        ax.grid(axis="y", alpha=0.15)
        outputs += _save(fig, output_dir, "fig_boundary_landscape_roughness",
                         "Normalized discrete curvature for each boundary landscape. Lower values indicate smoother local responses; the dashed threshold is a preregistered mechanism-selection heuristic, not a differentiability proof.")
    return outputs
