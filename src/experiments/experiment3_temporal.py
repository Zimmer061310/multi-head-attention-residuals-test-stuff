#!/usr/bin/env python3
"""Experiment 3B: temporal stability analysis for boundary-score vectors."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

from src.experiments.experiment1_partition_compatibility import (
    atomic_write_json,
    git_commit,
    sha256_file,
    utc_now,
)
from src.experiments.experiment3_common import (
    load_jsonl_by_id,
    scored_h16_candidates,
    spearman,
)
from src.experiments.experiment3_signal import PROTOCOL_PATH, add_wandb_args, protocol


def parse_step_paths(values):
    result = {}
    for value in values:
        try:
            raw_step, raw_path = value.split("=", 1)
            step = int(raw_step)
        except (ValueError, TypeError) as exc:
            raise ValueError("step results must use STEP=/path/results.jsonl") from exc
        if step in result:
            raise ValueError(f"duplicate step result {step}")
        result[step] = Path(raw_path).resolve()
    return result


def score_vectors(step_paths):
    vectors = {}
    identities = {}
    for step, path in step_paths.items():
        rows = load_jsonl_by_id(path)
        ranked = scored_h16_candidates(rows)
        if len(ranked) != 15:
            raise ValueError(f"step {step} has {len(ranked)} candidates, expected 15")
        vectors[step] = {
            row["candidate_id"]: row["delta_nll"] for row in ranked}
        identities[step] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "seed": ranked[0].get("seed"),
            "recorded_step": ranked[0].get("step"),
        }
        if identities[step]["recorded_step"] != step:
            raise ValueError(f"result at {path} does not record step {step}")
    return vectors, identities


def pair_metrics(vector_t, vector_u, *, step_t, step_u, split):
    if set(vector_t) != set(vector_u):
        raise RuntimeError("temporal candidate sets differ")
    ids = sorted(vector_t)
    values_t = [vector_t[identifier] for identifier in ids]
    values_u = [vector_u[identifier] for identifier in ids]
    top_t = set(sorted(ids, key=lambda identifier: (vector_t[identifier], identifier))[:3])
    top_u = set(sorted(ids, key=lambda identifier: (vector_u[identifier], identifier))[:3])
    median_t = statistics.median(values_t)
    median_u = statistics.median(values_u)
    sign_agreement = sum(
        (vector_t[identifier] <= median_t) == (vector_u[identifier] <= median_u)
        for identifier in ids
    ) / len(ids)
    selected = min(ids, key=lambda identifier: (vector_t[identifier], identifier))
    best_future = min(values_u)
    return {
        "split": split,
        "step_t": step_t,
        "step_u": step_u,
        "step_gap": step_u - step_t,
        "spearman": spearman(values_t, values_u),
        "pearson": float(np.corrcoef(values_t, values_u)[0, 1]),
        "top_three_overlap": len(top_t & top_u),
        "top_three_jaccard": len(top_t & top_u) / len(top_t | top_u),
        "median_side_agreement": sign_agreement,
        "selected_at_t": selected,
        "selected_future_delta_nll": vector_u[selected],
        "future_best_delta_nll": best_future,
        "future_regret": vector_u[selected] - best_future,
    }


def analyze_temporal(discovery, confirmation, *, adjacent_pairs):
    all_pairs = []
    steps = sorted(discovery)
    for split, vectors in (("discovery", discovery), ("confirmation", confirmation)):
        if set(vectors) != set(steps):
            raise RuntimeError("discovery and confirmation time points differ")
        for index, step_t in enumerate(steps):
            for step_u in steps[index + 1:]:
                all_pairs.append(pair_metrics(
                    vectors[step_t], vectors[step_u],
                    step_t=step_t, step_u=step_u, split=split))

    adjacent = {tuple(pair) for pair in adjacent_pairs}
    primary_discovery = [
        row for row in all_pairs
        if row["split"] == "discovery" and (row["step_t"], row["step_u"]) in adjacent
    ]
    primary_confirmation = {
        (row["step_t"], row["step_u"]): row
        for row in all_pairs
        if row["split"] == "confirmation" and (row["step_t"], row["step_u"]) in adjacent
    }
    if len(primary_discovery) != len(adjacent) or len(primary_confirmation) != len(adjacent):
        raise RuntimeError("missing preregistered adjacent-time comparisons")
    discovery_correlations = [row["spearman"] for row in primary_discovery]
    same_sign = sum(
        np.sign(row["spearman"])
        == np.sign(primary_confirmation[(row["step_t"], row["step_u"])]["spearman"])
        for row in primary_discovery
    )
    gate = (
        all(value > 0 for value in discovery_correlations)
        and statistics.median(discovery_correlations) >= 0.5
        and same_sign >= 2
    )
    return {
        "pair_metrics": all_pairs,
        "primary_discovery_spearman": discovery_correlations,
        "median_primary_discovery_spearman": statistics.median(discovery_correlations),
        "confirmation_same_sign_pairs": same_sign,
        "stability_gate_passed": gate,
        "gate_definition": {
            "all_discovery_spearman_positive": True,
            "minimum_median_discovery_spearman": 0.5,
            "minimum_confirmation_same_sign_pairs": 2,
            "adjacent_pairs": [list(pair) for pair in sorted(adjacent)],
        },
    }


def analyze_command(args):
    spec = protocol()
    discovery_paths = parse_step_paths(args.discovery)
    confirmation_paths = parse_step_paths(args.confirmation)
    expected = set(spec["probe_steps"])
    if set(discovery_paths) != expected or set(confirmation_paths) != expected:
        raise ValueError(
            f"temporal analysis requires steps {sorted(expected)} for both splits")
    discovery, discovery_identity = score_vectors(discovery_paths)
    confirmation, confirmation_identity = score_vectors(confirmation_paths)
    seeds = {
        row["seed"] for row in (*discovery_identity.values(), *confirmation_identity.values())}
    if len(seeds) != 1:
        raise RuntimeError(f"temporal results do not share one seed: {seeds}")
    analysis = analyze_temporal(
        discovery,
        confirmation,
        adjacent_pairs=spec["stability_gate"]["adjacent_pairs"],
    )
    summary = {
        "format_version": 1,
        "created_at": utc_now(),
        "source_commit": git_commit(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "seed": next(iter(seeds)),
        "discovery_inputs": discovery_identity,
        "confirmation_inputs": confirmation_identity,
        **analysis,
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in analysis["pair_metrics"]:
        row["seed"] = summary["seed"]
    atomic_write_json(output_dir / "temporal_summary.json", summary)
    with (output_dir / "temporal_correlations.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=analysis["pair_metrics"][0].keys())
        writer.writeheader()
        writer.writerows(analysis["pair_metrics"])
    regret_fields = (
        "split", "step_t", "step_u", "selected_at_t",
        "selected_future_delta_nll", "future_best_delta_nll", "future_regret")
    with (output_dir / "temporal_regret.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=regret_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in regret_fields}
                         for row in analysis["pair_metrics"])
    trajectory_rows = []
    for split, vectors in (("discovery", discovery), ("confirmation", confirmation)):
        for step, scores in sorted(vectors.items()):
            for candidate_id, delta_nll in sorted(scores.items()):
                trajectory_rows.append({
                    "split": split,
                    "step": step,
                    "candidate_id": candidate_id,
                    "boundary": int(candidate_id.split("-")[1]),
                    "delta_nll": delta_nll,
                })
    with (output_dir / "boundary_score_trajectories.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trajectory_rows[0].keys())
        writer.writeheader()
        writer.writerows(trajectory_rows)
    from figures.gen_fig_experiment3 import plot_score_trajectories, plot_temporal
    figure_files = plot_temporal(output_dir / "temporal_correlations.csv", output_dir)
    figure_files += plot_score_trajectories(
        output_dir / "boundary_score_trajectories.csv", output_dir)
    if args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name or f"exp3b-seed{summary['seed']}-temporal",
            mode=args.wandb_mode,
            job_type="experiment3-temporal-analysis",
            tags=["experiment-3", "temporal-stability", "analysis"],
            config=summary["gate_definition"],
        )
        columns = list(analysis["pair_metrics"][0])
        table = wandb.Table(columns=columns)
        for row in analysis["pair_metrics"]:
            table.add_data(*(row[column] for column in columns))
        run.log({
            "stability/gate_passed": int(summary["stability_gate_passed"]),
            "stability/median_discovery_spearman": (
                summary["median_primary_discovery_spearman"]),
            "stability/pairs": table,
            "stability/matrix_figure": wandb.Image(str(figure_files[0])),
            "stability/trajectory_figure": wandb.Image(str(figure_files[3])),
        })
        summary["wandb"] = {"run_id": run.id, "run_url": run.url}
        atomic_write_json(output_dir / "temporal_summary.json", summary)
        artifact = wandb.Artifact("experiment3-temporal-analysis", type="experiment-results")
        artifact.add_file(str(output_dir / "temporal_summary.json"))
        artifact.add_file(str(output_dir / "temporal_correlations.csv"))
        artifact.add_file(str(output_dir / "temporal_regret.csv"))
        artifact.add_file(str(output_dir / "boundary_score_trajectories.csv"))
        for path in figure_files:
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        run.finish()
    print(json.dumps(summary, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discovery", action="append", required=True,
        help="repeat as STEP=/path/discovery_results.jsonl")
    parser.add_argument(
        "--confirmation", action="append", required=True,
        help="repeat as STEP=/path/confirmation_results.jsonl")
    parser.add_argument("--output-dir", required=True)
    add_wandb_args(parser)
    return parser


def main():
    analyze_command(build_parser().parse_args())


if __name__ == "__main__":
    main()
