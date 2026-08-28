#!/usr/bin/env python3
"""Experiment 3D: within-seed replication and nested cross-seed summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.experiments.experiment1_partition_compatibility import (
    atomic_write_json,
    git_commit,
    sha256_file,
    utc_now,
)
from src.experiments.experiment3_signal import PROTOCOL_PATH, add_wandb_args, protocol


def parse_seed_bundles(values):
    bundles = {}
    for value in values:
        try:
            raw_seed, raw_path = value.split("=", 1)
            seed = int(raw_seed)
        except (ValueError, TypeError) as exc:
            raise ValueError("seed bundles must use SEED=/path/to/bundle") from exc
        if seed in bundles:
            raise ValueError(f"duplicate seed bundle {seed}")
        bundles[seed] = Path(raw_path).resolve()
    return bundles


def load_bundle(seed, root):
    paths = {
        "signal": root / "signal" / "signal_summary.json",
        "temporal": root / "temporal" / "temporal_summary.json",
        "actionability": root / "actionability" / "actionability_results.json",
        "selection": root / "actionability" / "branch_selection.json",
        "good": root / "actionability" / "results" / "predicted-good" / "result.json",
        "random": root / "actionability" / "results" / "random" / "result.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"seed {seed} bundle is incomplete: {missing}")
    values = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    recorded = {
        values["signal"].get("seed"),
        values["temporal"].get("seed"),
        values["actionability"].get("seed"),
        values["selection"].get("seed"),
        values["good"].get("seed"),
        values["random"].get("seed"),
    }
    if recorded != {seed}:
        raise RuntimeError(f"bundle {root} does not consistently record seed {seed}: {recorded}")
    return {
        "root": str(root),
        "hashes": {name: sha256_file(path) for name, path in paths.items()},
        **values,
    }


def find_contrast(summary, candidate, reference, split="confirmation"):
    matches = [
        row for row in summary["contrasts"]
        if row["split"] == split
        and row["candidate"] == candidate
        and row["reference"] == reference
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {candidate} minus {reference} {split} contrast")
    return matches[0]


def nested_seed_sequence_bootstrap(seed_records, *, samples, seed):
    differences = {}
    for seed_id, record in seed_records.items():
        good = np.asarray(
            record["good"]["splits"]["confirmation"]["sequence_nlls"],
            dtype=np.float64)
        random_values = np.asarray(
            record["random"]["splits"]["confirmation"]["sequence_nlls"],
            dtype=np.float64)
        if good.shape != random_values.shape or good.size == 0:
            raise RuntimeError(f"seed {seed_id} has unaligned branch sequences")
        differences[seed_id] = good - random_values
    seed_ids = np.asarray(sorted(differences))
    generator = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        selected = generator.choice(seed_ids, size=len(seed_ids), replace=True)
        seed_means = []
        for selected_seed in selected:
            values = differences[int(selected_seed)]
            indices = generator.integers(0, values.size, size=values.size)
            seed_means.append(values[indices].mean())
        means[sample_index] = np.mean(seed_means)
    observed_seed_means = np.asarray(
        [values.mean() for values in differences.values()], dtype=np.float64)
    return {
        "mean_seed_level_good_minus_random": float(observed_seed_means.mean()),
        "median_seed_level_good_minus_random": float(np.median(observed_seed_means)),
        "nested_bootstrap_ci95_low": float(np.quantile(means, 0.025)),
        "nested_bootstrap_ci95_high": float(np.quantile(means, 0.975)),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "warning": "Only three training seeds; nested interval is descriptive and unstable.",
    }


def cross_seed_analysis(seed_records, *, samples, bootstrap_seed, gate_spec):
    rows = []
    for seed, record in sorted(seed_records.items()):
        good_random = find_contrast(
            record["actionability"], "predicted-good", "random")
        good_bad = find_contrast(
            record["actionability"], "predicted-good", "predicted-bad")
        good_unchanged = find_contrast(
            record["actionability"], "predicted-good", "unchanged")
        branches = record["selection"]["branches"]
        rows.append({
            "seed": seed,
            "signal_gate_passed": bool(record["signal"]["signal_gate_passed"]),
            "signal_split_spearman": record["signal"][
                "discovery_confirmation_spearman"],
            "stability_gate_passed": bool(record["temporal"]["stability_gate_passed"]),
            "median_temporal_spearman": record["temporal"][
                "median_primary_discovery_spearman"],
            "actionability_gate_passed": bool(record["actionability"][
                "actionability_gate_passed"]),
            "good_boundary": branches["predicted-good"]["boundary"],
            "random_boundary": branches["random"]["boundary"],
            "bad_boundary": branches["predicted-bad"]["boundary"],
            "good_minus_random": good_random["mean_delta_nll"],
            "good_minus_random_ci95_low": good_random["ci95_low"],
            "good_minus_random_ci95_high": good_random["ci95_high"],
            "good_minus_bad": good_bad["mean_delta_nll"],
            "good_minus_bad_ci95_low": good_bad["ci95_low"],
            "good_minus_bad_ci95_high": good_bad["ci95_high"],
            "good_minus_unchanged": good_unchanged["mean_delta_nll"],
            "good_minus_unchanged_ci95_low": good_unchanged["ci95_low"],
            "good_minus_unchanged_ci95_high": good_unchanged["ci95_high"],
        })
    nested = nested_seed_sequence_bootstrap(
        seed_records, samples=samples, seed=bootstrap_seed)
    signal_passes = sum(row["signal_gate_passed"] for row in rows)
    stability_passes = sum(row["stability_gate_passed"] for row in rows)
    good_beats_random = sum(row["good_minus_random"] < 0 for row in rows)
    replication_passed = (
        signal_passes >= gate_spec["minimum_signal_pass_seeds"]
        and stability_passes >= gate_spec["minimum_stability_pass_seeds"]
        and good_beats_random >= gate_spec["minimum_good_beats_random_seeds"]
        and nested["mean_seed_level_good_minus_random"] < 0
    )
    contrast_descriptives = {}
    for key in ("good_minus_random", "good_minus_bad", "good_minus_unchanged"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        contrast_descriptives[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "range": float(values.max() - values.min()),
            "negative_seed_count": int((values < 0).sum()),
        }
    good_ids = [row["good_boundary"] for row in rows]
    boundary_id_overlap = {
        "good_ids_by_seed": {
            str(row["seed"]): row["good_boundary"] for row in rows},
        "unique_good_id_count": len(set(good_ids)),
        "all_good_ids_match": len(set(good_ids)) == 1,
        "matching_seed_pair_count": sum(
            good_ids[i] == good_ids[j]
            for i in range(len(good_ids)) for j in range(i + 1, len(good_ids))),
        "interpretation": "descriptive only; boundary-ID agreement is not a gate",
    }
    return {
        "seed_rows": rows,
        "signal_pass_seed_count": signal_passes,
        "stability_pass_seed_count": stability_passes,
        "good_beats_random_seed_count": good_beats_random,
        "strong_replication": all(row["good_minus_bad"] < 0 for row in rows),
        "replication_gate_passed": replication_passed,
        "gate_definition": gate_spec,
        "nested_bootstrap": nested,
        "contrast_descriptives": contrast_descriptives,
        "boundary_id_overlap": boundary_id_overlap,
        "good_boundary_ids_are_descriptive_only": True,
    }


def analyze_command(args):
    spec = protocol()
    bundle_paths = parse_seed_bundles(args.seed_bundle)
    expected = set(spec["seeds"])
    if set(bundle_paths) != expected:
        raise ValueError(f"cross-seed analysis requires bundles for {sorted(expected)}")
    records = {seed: load_bundle(seed, path) for seed, path in bundle_paths.items()}
    analysis = cross_seed_analysis(
        records,
        samples=spec["bootstrap_samples"],
        bootstrap_seed=spec["bootstrap_seed"],
        gate_spec=spec["replication_gate"],
    )
    summary = {
        "format_version": 1,
        "created_at": utc_now(),
        "source_commit": git_commit(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "input_bundles": {
            str(seed): {"root": record["root"], "hashes": record["hashes"]}
            for seed, record in records.items()
        },
        **analysis,
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "cross_seed_summary.json", summary)
    with (output_dir / "cross_seed_summary.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=analysis["seed_rows"][0].keys())
        writer.writeheader()
        writer.writerows(analysis["seed_rows"])
    from figures.gen_fig_experiment3 import plot_cross_seed
    figure_files = plot_cross_seed(output_dir / "cross_seed_summary.csv", output_dir)
    if args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name or "exp3d-cross-seed-analysis",
            mode=args.wandb_mode,
            job_type="experiment3-cross-seed-analysis",
            tags=["experiment-3", "cross-seed", "analysis"],
            config=analysis["gate_definition"],
        )
        columns = list(analysis["seed_rows"][0])
        table = wandb.Table(columns=columns)
        for row in analysis["seed_rows"]:
            table.add_data(*(row[column] for column in columns))
        run.log({
            "replication/gate_passed": int(summary["replication_gate_passed"]),
            "replication/good_beats_random_seed_count": (
                summary["good_beats_random_seed_count"]),
            "replication/mean_good_minus_random": summary["nested_bootstrap"][
                "mean_seed_level_good_minus_random"],
            "replication/seeds": table,
            "replication/actionability_figure": wandb.Image(str(figure_files[0])),
            "replication/boundary_id_figure": wandb.Image(str(figure_files[3])),
        })
        summary["wandb"] = {"run_id": run.id, "run_url": run.url}
        atomic_write_json(output_dir / "cross_seed_summary.json", summary)
        artifact = wandb.Artifact("experiment3-cross-seed-analysis", type="experiment-results")
        artifact.add_file(str(output_dir / "cross_seed_summary.json"))
        artifact.add_file(str(output_dir / "cross_seed_summary.csv"))
        for path in figure_files:
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        run.finish()
    print(json.dumps(summary, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-bundle", action="append", required=True,
        help="repeat as SEED=/path/to/seed-bundle")
    parser.add_argument("--output-dir", required=True)
    add_wandb_args(parser)
    return parser


def main():
    analyze_command(build_parser().parse_args())


if __name__ == "__main__":
    main()
