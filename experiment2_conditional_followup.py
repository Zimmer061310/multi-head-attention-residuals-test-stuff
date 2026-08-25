#!/usr/bin/env python3
"""Prepare and analyze the pre-registered conditional Experiment 2 follow-up."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from experiment1_partition_compatibility import atomic_write_json, git_commit
from experiment2_boundary_contribution import (
    DEFAULT_SEED,
    DEFAULT_WANDB_PROJECT,
    candidate_rankings,
    fit_additive,
    frozen_json,
    load_k4_design,
    load_transfer_rows,
    regression_metrics,
    selection_manifest,
    sha256_file,
    spearman,
    utc_now,
    wandb_run,
    write_csv,
)


FOLLOWUP_SPACES = {1: 15, 2: 91, 6: 210, 7: 36}
EXHAUSTIVE_K = {1, 7}


def gate_decisions(transfer_summary: dict) -> dict[int, bool]:
    lower = bool(transfer_summary["k3"]["directional_conditions_met"])
    upper = bool(transfer_summary["k5"]["directional_conditions_met"])
    return {1: lower, 2: lower, 6: upper, 7: upper}


def exhaustive_manifest(
    rankings: list[dict], *, num_merges: int, source_hash: str,
    score_hash: str, gate_hash: str, source_commit: str,
) -> dict:
    return {
        "format_version": 1,
        "experiment": 2,
        "stage": "boundary-score-conditional-followup",
        "evidence_label": "sequential follow-up on reused confirmation split",
        "num_merges": num_merges,
        "candidate_space_size": len(rankings),
        "source_discovery_results_sha256": source_hash,
        "boundary_effects_sha256": score_hash,
        "gate_transfer_summary_sha256": gate_hash,
        "source_commit": source_commit,
        "selection_rule": "complete valid candidate space",
        "candidates": [
            {
                "partition_id": row["partition_id"],
                "roles": ["exhaustive_transfer"],
                "predicted_rank": row["predicted_rank"],
                "predicted_score": row["predicted_score"],
                "merged_boundaries": row["merged_boundaries"],
            }
            for row in rankings
        ],
    }


def sampled_manifest(
    rankings: list[dict], *, num_merges: int, source_hash: str,
    score_hash: str, gate_hash: str, source_commit: str, seed: int,
    uniform_size: int,
) -> dict:
    payload = selection_manifest(
        rankings,
        num_merges=num_merges,
        source_hash=source_hash,
        score_hash=score_hash,
        seed=seed,
        uniform_size=uniform_size,
        source_commit=source_commit,
    )
    payload.update({
        "stage": "boundary-score-conditional-followup",
        "evidence_label": "sequential follow-up on reused confirmation split",
        "gate_transfer_summary_sha256": gate_hash,
    })
    return payload


def add_wandb_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"),
        default="disabled")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-run-name", default=None)


def prepare_command(args) -> None:
    discovery_path = Path(args.discovery_results).resolve()
    transfer_path = Path(args.transfer_summary).resolve()
    effects_path = Path(args.boundary_effects).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    transfer = json.loads(transfer_path.read_text(encoding="utf-8"))
    decisions = gate_decisions(transfer)
    eligible = [k for k in FOLLOWUP_SPACES if decisions[k]]
    _, indicators, target, _ = load_k4_design(discovery_path)
    model = fit_additive(indicators, target)
    source_hash = sha256_file(discovery_path)
    score_hash = sha256_file(effects_path)
    gate_hash = sha256_file(transfer_path)
    source_commit = git_commit()
    manifests = {}

    for num_merges in eligible:
        rankings = candidate_rankings(model["beta"], num_merges)
        expected = FOLLOWUP_SPACES[num_merges]
        if len(rankings) != expected:
            raise AssertionError((num_merges, len(rankings), expected))
        write_csv(
            output_dir / f"k{num_merges}_predicted_rankings.csv",
            rankings,
            ["predicted_rank", "partition_id", "predicted_score", "merged_boundaries"],
        )
        if num_merges in EXHAUSTIVE_K:
            manifest = exhaustive_manifest(
                rankings, num_merges=num_merges, source_hash=source_hash,
                score_hash=score_hash, gate_hash=gate_hash,
                source_commit=source_commit)
        else:
            manifest = sampled_manifest(
                rankings, num_merges=num_merges, source_hash=source_hash,
                score_hash=score_hash, gate_hash=gate_hash,
                source_commit=source_commit, seed=args.seed,
                uniform_size=args.uniform_size)
        path = output_dir / f"k{num_merges}_selection.json"
        frozen_json(path, manifest)
        manifests[str(num_merges)] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "candidate_count": len(manifest["candidates"]),
        }

    gate = {
        "source_commit": source_commit,
        "evidence_label": "sequential follow-up on reused confirmation split",
        "operational_gate": (
            "k3 directional_conditions_met enables k1/k2; "
            "k5 directional_conditions_met enables k6/k7"),
        "transfer_summary_sha256": gate_hash,
        "source_discovery_results_sha256": source_hash,
        "boundary_effects_sha256": score_hash,
        "directional_conditions": {
            "k3": bool(transfer["k3"]["directional_conditions_met"]),
            "k5": bool(transfer["k5"]["directional_conditions_met"]),
        },
        "eligible_k": eligible,
        "manifests": manifests,
    }
    frozen_json(output_dir / "followup_gate.json", gate)

    run = wandb_run(args, job_type="conditional-followup-prepare", config=gate)
    if run is not None:
        import wandb
        table = wandb.Table(columns=["k", "eligible", "candidate_space", "evaluated"])
        for k, size in FOLLOWUP_SPACES.items():
            table.add_data(
                k, decisions[k], size,
                manifests.get(str(k), {}).get("candidate_count", 0))
        run.log({"analysis/followup_gate": table})
        artifact = wandb.Artifact(
            "mhar-exp2-conditional-followup-manifests", type="analysis")
        for path in sorted(output_dir.iterdir()):
            if path.is_file():
                artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
        run.finish()
    print(json.dumps(gate, indent=2, sort_keys=True))


def calibrated_metrics(rows: list[dict], role: str) -> dict:
    selected = [row for row in rows if role in row["roles"]]
    predicted = np.asarray(
        [row["predicted_score"] for row in selected], dtype=float)
    actual = np.asarray(
        [row["actual_delta_nll"] for row in selected], dtype=float)
    if len(selected) < 2:
        raise ValueError(f"rank analysis requires at least two {role} rows")
    design = np.column_stack([np.ones(len(predicted)), predicted])
    coefficients = np.linalg.lstsq(design, actual, rcond=None)[0]
    calibrated = design @ coefficients
    metrics = regression_metrics(actual, calibrated)
    return {
        "count": len(selected),
        "spearman": spearman(actual, predicted),
        "affine_intercept": float(coefficients[0]),
        "affine_slope": float(coefficients[1]),
        "descriptive_calibrated_rmse": metrics["rmse"],
        "descriptive_calibrated_mae": metrics["mae"],
    }


def targeted_summary(rows: list[dict]) -> dict | None:
    if not any("target_top" in row["roles"] for row in rows):
        return None
    groups = {}
    for name in ("top", "middle", "bottom"):
        values = [
            row["actual_delta_nll"] for row in rows
            if f"target_{name}" in row["roles"]
        ]
        groups[name] = {
            "count": len(values),
            "mean_actual_delta_nll": statistics.fmean(values),
            "median_actual_delta_nll": statistics.median(values),
        }
    groups["top_minus_middle_mean_delta"] = (
        groups["top"]["mean_actual_delta_nll"]
        - groups["middle"]["mean_actual_delta_nll"])
    groups["top_minus_bottom_mean_delta"] = (
        groups["top"]["mean_actual_delta_nll"]
        - groups["bottom"]["mean_actual_delta_nll"])
    return groups


def write_figure(
    output_dir: Path, by_k: dict[int, list[dict]], summaries: dict,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = sorted(by_k)
    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    }):
        fig, axes = plt.subplots(
            1, len(keys), figsize=(3.3 * len(keys), 3.2), squeeze=False)
        for axis, k in zip(axes[0], keys):
            rows = by_k[k]
            axis.scatter(
                [row["predicted_score"] for row in rows],
                [row["actual_delta_nll"] for row in rows],
                s=18, color="#0072B2", alpha=0.75, linewidths=0)
            rho = summaries[str(k)]["rank_metrics"]["spearman"]
            axis.set_title(f"k={k}: Spearman={rho:.3f}")
            axis.set_xlabel("Frozen additive score")
            axis.grid(color="#D1D5DB", linewidth=0.5, alpha=0.5)
        axes[0][0].set_ylabel("Actual confirmation delta NLL")
        fig.suptitle("Conditional boundary-score follow-up", weight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / "fig_conditional_followup.png", dpi=300)
        fig.savefig(output_dir / "fig_conditional_followup.pdf")
        plt.close(fig)


def analyze_command(args) -> None:
    gate_path = Path(args.gate_manifest).resolve()
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    eligible = [int(k) for k in gate["eligible_k"]]
    by_k = {}
    summaries = {}
    combined = []
    for k in eligible:
        results_value = getattr(args, f"k{k}_results")
        selection_value = getattr(args, f"k{k}_selection")
        if not results_value or not selection_value:
            raise ValueError(
                f"k={k} is eligible but its results/selection path is missing")
        rows, native_nll, selection = load_transfer_rows(
            Path(results_value).resolve(), Path(selection_value).resolve(), k)
        role = "exhaustive_transfer" if k in EXHAUSTIVE_K else "uniform_transfer"
        rank_metrics = calibrated_metrics(rows, role)
        measured = sorted(
            rows, key=lambda row: (row["actual_delta_nll"], row["partition_id"]))
        summaries[str(k)] = {
            "evidence_label": selection["evidence_label"],
            "native_confirmation_nll": native_nll,
            "evaluated_candidate_count": len(rows),
            "rank_metrics": rank_metrics,
            "targeted": targeted_summary(rows),
            "best_partition": measured[0],
            "worst_partition": measured[-1],
            "results_sha256": sha256_file(Path(results_value).resolve()),
            "selection_sha256": sha256_file(Path(selection_value).resolve()),
        }
        by_k[k] = rows
        combined.extend(rows)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "conditional_followup_candidates.csv", combined,
        ["k", "partition_id", "roles", "predicted_rank", "predicted_score",
         "actual_nll", "actual_delta_nll", "merged_boundaries"],
    )
    summary = {
        "created_at": utc_now(),
        "source_commit": git_commit(),
        "gate_manifest_sha256": sha256_file(gate_path),
        "eligible_k": eligible,
        "by_k": summaries,
    }
    atomic_write_json(
        output_dir / "conditional_followup_summary.json", summary)
    write_figure(output_dir, by_k, summaries)
    lines = ["# Experiment 2 conditional boundary-score follow-up", ""]
    for k in eligible:
        item = summaries[str(k)]
        lines.extend([
            f"## k={k}", "",
            f"- Candidates: {item['evaluated_candidate_count']}",
            f"- Rank Spearman: {item['rank_metrics']['spearman']:.6f}",
            f"- Best partition: {item['best_partition']['partition_id']}",
            f"- Worst partition: {item['worst_partition']['partition_id']}", "",
        ])
    (output_dir / "conditional_followup_report.md").write_text(
        "\n".join(lines), encoding="utf-8")

    run = wandb_run(
        args, job_type="conditional-followup-analysis", config=summary)
    if run is not None:
        import wandb
        table = wandb.Table(columns=[
            "k", "partition_id", "roles", "predicted_rank", "predicted_score",
            "actual_delta_nll", "merged_boundaries"])
        for row in combined:
            table.add_data(*[row[column] for column in table.columns])
        payload = {
            "analysis/conditional_candidates": table,
            "figures/conditional_followup": wandb.Image(
                str(output_dir / "fig_conditional_followup.png")),
        }
        for k in eligible:
            payload[f"k{k}/rank_spearman"] = (
                summaries[str(k)]["rank_metrics"]["spearman"])
        run.log(payload)
        artifact = wandb.Artifact(
            "mhar-exp2-conditional-followup", type="analysis")
        for path in sorted(output_dir.iterdir()):
            if path.is_file():
                artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
        run.finish()
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--discovery-results", required=True)
    prepare.add_argument("--boundary-effects", required=True)
    prepare.add_argument("--transfer-summary", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument("--uniform-size", type=int, default=30)
    add_wandb_arguments(prepare)
    prepare.set_defaults(func=prepare_command)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--gate-manifest", required=True)
    analyze.add_argument("--output-dir", required=True)
    for k in FOLLOWUP_SPACES:
        analyze.add_argument(f"--k{k}-results", dest=f"k{k}_results")
        analyze.add_argument(f"--k{k}-selection", dest=f"k{k}_selection")
    add_wandb_arguments(analyze)
    analyze.set_defaults(func=analyze_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
