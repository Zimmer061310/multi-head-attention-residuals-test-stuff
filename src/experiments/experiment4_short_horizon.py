"""Freeze and analyze Experiment 4 short-horizon training branches."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.training.train_scratch import atomic_write_json, sha256_tree


ROLES = ("predicted-good", "predicted-bad", "unchanged")
PARTITIONS = {
    "predicted-good": "0__1__2__3-4__5__6__7__8__9__10__11__12__13__14__15",
    "predicted-bad": "0__1__2__3__4__5__6__7__8__9__10__11__12__13-14__15",
    "unchanged": None,
}


def load_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def moving_block_ci(values, *, samples=10_000, block=5, seed=20260829):
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError("empty paired differences")
    rng = np.random.default_rng(seed)
    starts = np.arange(max(1, len(values) - block + 1))
    estimates = np.empty(samples)
    blocks_needed = int(np.ceil(len(values) / block))
    for index in range(samples):
        picked = []
        for start in rng.choice(starts, size=blocks_needed, replace=True):
            picked.extend(values[start:start + block])
        estimates[index] = np.mean(picked[:len(values)])
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def prepare(args):
    parent = Path(args.parent).resolve()
    manifest = json.loads((parent / "training_manifest.json").read_text(encoding="utf-8"))
    if manifest["global_step"] != 1500 or manifest["run_identity"]["seed"] != 43:
        raise RuntimeError("Experiment 4 requires the seed-43 step-1500 checkpoint")
    payload = {
        "format_version": 1,
        "experiment": "4-short-horizon-actionability",
        "seed": 43,
        "source_step": 1500,
        "endpoint_step": 2000,
        "log_every": 10,
        "parent_checkpoint_sha256": sha256_tree(parent),
        "frozen_probe": {
            "best_candidate": "remove-03",
            "worst_candidate": "remove-13",
            "discovery_confirmation_spearman": 0.9392857142857142,
            "confirmation_best_minus_worst_nll": -0.01105651818215847,
            "confirmation_ci95": [-0.012314494664315135, -0.009782705514226109],
        },
        "branches": {
            role: {
                "label": label,
                "candidate_id": {"predicted-good": "remove-03", "predicted-bad": "remove-13", "unchanged": "native-h16"}[role],
                "partition_id": PARTITIONS[role],
            }
            for role, label in zip(ROLES, ("A", "B", "C"))
        },
    }
    atomic_write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def analyze(args):
    root = Path(args.branch_root).resolve()
    series = {}
    for role in ROLES:
        rows = load_jsonl(root / role / "training_metrics.jsonl")
        series[role] = {
            int(row["step"]): float(row["loss"])
            for row in rows if 1500 < int(row["step"]) <= 2000
        }
    steps = sorted(set.intersection(*(set(rows) for rows in series.values())))
    expected = list(range(1510, 2001, 10))
    if steps != expected:
        raise RuntimeError(f"expected 50 paired points, got {steps}")

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    paired_rows = []
    for step in steps:
        a, b, c = (series[role][step] for role in ROLES)
        paired_rows.append({
            "step": step, "loss_A": a, "loss_B": b, "loss_C": c,
            "delta_A_minus_B": a - b, "delta_A_minus_C": a - c,
            "delta_C_minus_B": c - b,
        })
    with (out / "paired_losses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=paired_rows[0].keys())
        writer.writeheader()
        writer.writerows(paired_rows)

    horizons = []
    for horizon in (1600, 1700, 1800, 1900, 2000):
        rows = [row for row in paired_rows if row["step"] <= horizon]
        trailing = [row for row in rows if row["step"] > horizon - 100]
        diffs = [row["delta_A_minus_B"] for row in rows]
        horizons.append({
            "horizon_step": horizon,
            "steps_after_branch": horizon - 1500,
            "endpoint_A_minus_B": rows[-1]["delta_A_minus_B"],
            "cumulative_mean_A_minus_B": float(np.mean(diffs)),
            "cumulative_mean_A_minus_B_ci95": moving_block_ci(diffs),
            "trailing_100_mean_A_minus_B": float(np.mean([row["delta_A_minus_B"] for row in trailing])),
            "cumulative_mean_A_minus_C": float(np.mean([row["delta_A_minus_C"] for row in rows])),
            "cumulative_mean_C_minus_B": float(np.mean([row["delta_C_minus_B"] for row in rows])),
        })
    early, primary = horizons[0], horizons[-1]
    clear = any(row["cumulative_mean_A_minus_B_ci95"][1] < 0 for row in (early, primary))
    directional = early["cumulative_mean_A_minus_B"] < 0 and primary["cumulative_mean_A_minus_B"] < 0
    summary = {
        "format_version": 1,
        "experiment": "4-short-horizon-actionability",
        "paired_observations": len(paired_rows),
        "horizons": horizons,
        "clear_short_horizon_actionability": clear,
        "consistently_directional_actionability": directional,
        "decision": "investigate_adaptive_grouping" if clear or directional else "stop_boundary_learning_direction",
    }
    atomic_write_json(out / "short_horizon_summary.json", summary)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for role, label in zip(ROLES, ("A: best remove-03", "B: worst remove-13", "C: native H16")):
        ax.plot(steps, [series[role][step] for step in steps], label=label, linewidth=1.5)
    ax.set(xlabel="Optimizer step", ylabel="Training NLL", title="Experiment 4: paired short-horizon training loss")
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(out / "fig_short_horizon_losses.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    delta = [row["delta_A_minus_B"] for row in paired_rows]
    ax.axhline(0, color="black", linewidth=1)
    ax.plot(steps, delta, color="#6a3d9a", linewidth=1.5, label="A - B")
    ax.set(xlabel="Optimizer step", ylabel="Delta training NLL", title="Best-boundary advantage (negative favors A)")
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(out / "fig_short_horizon_delta.png", dpi=180); plt.close(fig)

    if args.wandb:
        import wandb
        run = wandb.init(project="MHAR Stuff", group="mhar-exp4-short-horizon-seed43-step1500",
                         job_type="experiment4-analysis", name="mhar-exp4-seed43-analysis")
        table = wandb.Table(columns=list(paired_rows[0]), data=[[row[key] for key in paired_rows[0]] for row in paired_rows])
        run.log({"analysis/paired_losses": table,
                 "analysis/loss_figure": wandb.Image(str(out / "fig_short_horizon_losses.png")),
                 "analysis/delta_figure": wandb.Image(str(out / "fig_short_horizon_delta.png")),
                 "analysis/primary_mean_A_minus_B": primary["cumulative_mean_A_minus_B"],
                 "analysis/early_mean_A_minus_B": early["cumulative_mean_A_minus_B"],
                 "analysis/decision_adaptive": int(summary["decision"] == "investigate_adaptive_grouping")})
        summary["wandb_url"] = run.url
        run.finish()
        atomic_write_json(out / "short_horizon_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--parent", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.set_defaults(func=prepare)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--branch-root", required=True)
    analyze_parser.add_argument("--output-dir", required=True)
    analyze_parser.add_argument("--wandb", action="store_true")
    analyze_parser.set_defaults(func=analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
