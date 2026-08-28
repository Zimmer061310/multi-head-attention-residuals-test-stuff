#!/usr/bin/env python3
"""Experiment 3C: frozen branch selection, evaluation, and actionability gate."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch

from src.attention_residuals.mhar_partition import parse_mixed_partition_id
from src.experiments.experiment1_partition_compatibility import (
    atomic_write_json,
    git_commit,
    sha256_file,
    utc_now,
)
from src.experiments.experiment3_common import (
    DEFAULT_WANDB_PROJECT,
    checkpoint_identity,
    evaluate_tokens,
    load_artifact_split,
    load_jsonl_by_id,
    load_mhar_model,
    paired_bootstrap,
    scored_h16_candidates,
)
from src.experiments.experiment3_signal import PROTOCOL_PATH, add_wandb_args, protocol


BRANCH_ROLES = ("predicted-good", "predicted-bad", "random", "unchanged")


def build_branch_selection(
    discovery_rows,
    *,
    parent_checkpoint,
    signal_summary,
    temporal_summary,
    random_seed,
    require_upstream_gates=True,
):
    if require_upstream_gates and not signal_summary.get("signal_gate_passed"):
        raise RuntimeError("Experiment 3A signal gate did not pass")
    if require_upstream_gates and not temporal_summary.get("stability_gate_passed"):
        raise RuntimeError("Experiment 3B stability gate did not pass")
    ranked = scored_h16_candidates(discovery_rows)
    if len(ranked) != 15:
        raise ValueError(f"expected 15 removal candidates, found {len(ranked)}")
    good = ranked[0]
    bad = ranked[-1]
    middle = ranked[4:11]
    random_row = random.Random(random_seed).choice(middle)
    parent_info = checkpoint_identity(Path(parent_checkpoint))
    manifest = {
        "format_version": 1,
        "created_at": utc_now(),
        "source_commit": git_commit(),
        "parent_checkpoint": parent_info["path"],
        "parent_checkpoint_sha256": parent_info["sha256"],
        "parent_step": signal_summary.get("step", 1500),
        "random_seed": random_seed,
        "local_signal_gate_passed": bool(signal_summary.get("signal_gate_passed")),
        "local_stability_gate_passed": bool(temporal_summary.get("stability_gate_passed")),
        "local_gates_required_for_selection": require_upstream_gates,
        "middle_rank_pool": [row["candidate_id"] for row in middle],
        "branches": {
            "predicted-good": {
                "candidate_id": good["candidate_id"],
                "boundary": good["boundary"],
                "partition_id": good["partition_id"],
                "discovery_delta_nll": good["delta_nll"],
            },
            "predicted-bad": {
                "candidate_id": bad["candidate_id"],
                "boundary": bad["boundary"],
                "partition_id": bad["partition_id"],
                "discovery_delta_nll": bad["delta_nll"],
            },
            "random": {
                "candidate_id": random_row["candidate_id"],
                "boundary": random_row["boundary"],
                "partition_id": random_row["partition_id"],
                "discovery_delta_nll": random_row["delta_nll"],
            },
            "unchanged": {
                "candidate_id": "native-h16",
                "boundary": None,
                "partition_id": None,
                "discovery_delta_nll": 0.0,
            },
        },
    }
    return manifest


def select_command(args):
    spec = protocol()
    discovery_path = Path(args.discovery_results).resolve()
    signal_path = Path(args.signal_summary).resolve()
    temporal_path = Path(args.temporal_summary).resolve()
    signal_summary = json.loads(signal_path.read_text(encoding="utf-8"))
    temporal_summary = json.loads(temporal_path.read_text(encoding="utf-8"))
    if signal_summary.get("seed") != args.seed or temporal_summary.get("seed") != args.seed:
        raise RuntimeError("signal and temporal summaries must match --seed")
    if signal_summary.get("discovery_results_sha256") != sha256_file(discovery_path):
        raise RuntimeError("signal summary does not match branch discovery results")
    parent_info = checkpoint_identity(Path(args.parent_checkpoint))
    parent_training = parent_info.get("training_manifest")
    if parent_training is None:
        raise FileNotFoundError("actionability parent requires training_manifest.json")
    if int(parent_training["global_step"]) != spec["primary_probe_step"]:
        raise RuntimeError("actionability parent is not the step-1,500 checkpoint")
    if parent_training["run_identity"].get("seed") != args.seed:
        raise RuntimeError("actionability parent seed does not match --seed")
    replication_authorization = None
    if args.cross_seed_replication:
        if args.seed not in (43, 44):
            raise ValueError("cross-seed replication bypass is only valid for seeds 43 and 44")
        if not args.seed42_actionability_summary:
            raise ValueError("cross-seed replication requires --seed42-actionability-summary")
        authorization_path = Path(args.seed42_actionability_summary).resolve()
        replication_authorization = json.loads(
            authorization_path.read_text(encoding="utf-8"))
        if replication_authorization.get("seed") != 42:
            raise RuntimeError("replication authorization is not a seed-42 result")
        if not replication_authorization.get("actionability_gate_passed"):
            raise RuntimeError("seed-42 actionability did not authorize cross-seed branches")
    manifest = build_branch_selection(
        load_jsonl_by_id(discovery_path),
        parent_checkpoint=Path(args.parent_checkpoint),
        signal_summary=signal_summary,
        temporal_summary=temporal_summary,
        random_seed=spec["branch_random_seed"] + args.seed,
        require_upstream_gates=not args.cross_seed_replication,
    )
    manifest.update({
        "seed": args.seed,
        "endpoint_step": spec["branch_endpoint_step"],
        "discovery_results_sha256": sha256_file(discovery_path),
        "signal_summary_sha256": sha256_file(signal_path),
        "temporal_summary_sha256": sha256_file(temporal_path),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "cross_seed_replication": bool(args.cross_seed_replication),
        "seed42_actionability_authorization_sha256": (
            sha256_file(Path(args.seed42_actionability_summary).resolve())
            if args.cross_seed_replication else None),
    })
    output = Path(args.output).resolve()
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        left = {key: value for key, value in existing.items() if key != "created_at"}
        right = {key: value for key, value in manifest.items() if key != "created_at"}
        if left != right:
            raise RuntimeError("refusing to overwrite a different branch manifest")
        atomic_write_json(
            output.with_name(f"branch_selection_seed{args.seed}.json"), existing)
        return
    atomic_write_json(output, manifest)
    atomic_write_json(
        output.with_name(f"branch_selection_seed{args.seed}.json"), manifest)
    print(json.dumps(manifest, indent=2))


def apply_branch(model, row):
    partition_id = row.get("partition_id")
    if partition_id is None:
        model.set_mhar_mixed_partition(None)
    else:
        model.set_mhar_mixed_partition(parse_mixed_partition_id(
            partition_id, num_atomic_blocks=16))


def evaluate_command(args):
    spec = protocol()
    branch_path = Path(args.branch_manifest).resolve()
    branch_manifest = json.loads(branch_path.read_text(encoding="utf-8"))
    if args.role not in branch_manifest["branches"]:
        raise ValueError(f"branch manifest has no role {args.role!r}")
    branch = branch_manifest["branches"][args.role]
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_info = checkpoint_identity(checkpoint)
    training = checkpoint_info["training_manifest"]
    if training is None:
        raise FileNotFoundError("branch checkpoint lacks training_manifest.json")
    if int(training["global_step"]) != spec["branch_endpoint_step"]:
        raise ValueError("branch checkpoint is not at the preregistered endpoint")
    recorded_branch = training["run_identity"].get("branch", {})
    if recorded_branch.get("role") != args.role:
        raise RuntimeError("checkpoint branch role differs from evaluation role")
    if recorded_branch.get("selection_manifest_sha256") != sha256_file(branch_path):
        raise RuntimeError("checkpoint was not trained from this branch manifest")

    artifact = Path(args.artifact).resolve()
    payload, artifact_hash, _ = load_artifact_split(artifact, "discovery")
    identity = {
        "experiment": "3c-actionability",
        "role": args.role,
        "seed": branch_manifest["seed"],
        "endpoint_step": spec["branch_endpoint_step"],
        "checkpoint": checkpoint_info,
        "artifact_path": str(artifact),
        "artifact_sha256": artifact_hash,
        "branch_manifest": str(branch_path),
        "branch_manifest_sha256": sha256_file(branch_path),
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "source_commit": git_commit(),
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["run_identity"] != identity:
            raise RuntimeError("existing branch evaluation identity differs")
        if result_path.is_file():
            print(f"completed result already exists: {result_path}")
            return
    else:
        manifest = {"created_at": utc_now(), "run_identity": identity}
        atomic_write_json(manifest_path, manifest)

    device = torch.device(args.device)
    model, architecture = load_mhar_model(
        checkpoint, device=device, dtype=args.dtype, required_heads=16)
    apply_branch(model, branch)
    manifest["architecture"] = architecture
    run = None
    if args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name or f"exp3c-{args.role}-eval",
            mode=args.wandb_mode,
            job_type="experiment3-actionability-evaluation",
            tags=["experiment-3", "actionability", args.role, "fixed-eval"],
            config=identity,
        )
        manifest["wandb"] = {"run_id": run.id, "run_url": run.url}
    atomic_write_json(manifest_path, manifest)
    splits = {}
    for split in ("discovery", "confirmation"):
        metrics = evaluate_tokens(
            model,
            payload[f"{split}_input_ids"],
            batch_size=args.batch_size,
            device=device,
        )
        splits[split] = metrics
        if run is not None:
            run.log({f"{split}/nll": metrics["nll"], f"{split}/ppl": metrics["ppl"]})
    result = {
        "format_version": 1,
        "created_at": utc_now(),
        "role": args.role,
        "seed": branch_manifest["seed"],
        "branch": branch,
        "checkpoint_sha256": checkpoint_info["sha256"],
        "artifact_sha256": artifact_hash,
        "splits": splits,
        "wandb": manifest.get("wandb"),
    }
    atomic_write_json(result_path, result)
    if run is not None:
        import wandb

        artifact_result = wandb.Artifact(
            f"exp3c-{args.role}-evaluation", type="experiment-results")
        artifact_result.add_file(str(result_path))
        artifact_result.add_file(str(manifest_path))
        run.log_artifact(artifact_result)
        run.finish()


def actionability_analysis(results, *, samples, seed):
    if set(results) != set(BRANCH_ROLES):
        raise ValueError(f"actionability requires roles {BRANCH_ROLES}")
    contrasts = []
    for split in ("discovery", "confirmation"):
        for candidate, reference in (
            ("predicted-good", "random"),
            ("predicted-good", "predicted-bad"),
            ("predicted-good", "unchanged"),
            ("random", "predicted-bad"),
        ):
            estimate = paired_bootstrap(
                results[candidate]["splits"][split]["sequence_nlls"],
                results[reference]["splits"][split]["sequence_nlls"],
                samples=samples,
                seed=seed,
            )
            contrasts.append({
                "split": split,
                "candidate": candidate,
                "reference": reference,
                **estimate,
            })
    primary = next(
        row for row in contrasts
        if row["split"] == "confirmation"
        and row["candidate"] == "predicted-good"
        and row["reference"] == "random")
    rankings = {}
    for split in ("discovery", "confirmation"):
        ordered = sorted(
            BRANCH_ROLES,
            key=lambda role: (results[role]["splits"][split]["nll"], role))
        rankings[split] = [
            {
                "rank": index,
                "role": role,
                "nll": results[role]["splits"][split]["nll"],
            }
            for index, role in enumerate(ordered, 1)
        ]
    strong = (
        rankings["confirmation"][0]["role"] == "predicted-good"
        and next(row for row in contrasts
                 if row["split"] == "confirmation"
                 and row["candidate"] == "predicted-good"
                 and row["reference"] == "predicted-bad")["ci95_high"] < 0
        and next(row for row in contrasts
                 if row["split"] == "confirmation"
                 and row["candidate"] == "predicted-good"
                 and row["reference"] == "unchanged")["ci95_high"] < 0
    )
    return {
        "contrasts": contrasts,
        "rankings": rankings,
        "actionability_gate_passed": primary["ci95_high"] < 0,
        "strong_actionability_result": strong,
        "gate_definition": (
            "predicted-good minus random confirmation paired 95% CI is below zero"),
    }


def analyze_command(args):
    spec = protocol()
    root = Path(args.results_root).resolve()
    results = {}
    for role in BRANCH_ROLES:
        path = root / role / "result.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing actionability result: {path}")
        results[role] = json.loads(path.read_text(encoding="utf-8"))
    hashes = {row["artifact_sha256"] for row in results.values()}
    seeds = {row["seed"] for row in results.values()}
    if len(hashes) != 1 or len(seeds) != 1:
        raise RuntimeError("actionability results do not share one artifact and seed")
    analysis = actionability_analysis(
        results,
        samples=spec["bootstrap_samples"],
        seed=spec["bootstrap_seed"],
    )
    summary = {
        "format_version": 1,
        "created_at": utc_now(),
        "source_commit": git_commit(),
        "artifact_sha256": next(iter(hashes)),
        "seed": next(iter(seeds)),
        **analysis,
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_root = Path(args.training_root).resolve()
    combined_training_path = output_dir / "branch_training_metrics.jsonl"
    with combined_training_path.open("w", encoding="utf-8") as destination:
        for role in BRANCH_ROLES:
            source = training_root / role / "training_metrics.jsonl"
            if not source.is_file():
                raise FileNotFoundError(f"missing branch training metrics: {source}")
            for line in source.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("branch_role") != role:
                    raise RuntimeError(f"training metric role mismatch in {source}")
                destination.write(json.dumps(row, sort_keys=True) + "\n")
    atomic_write_json(output_dir / "actionability_results.json", summary)
    with (output_dir / "actionability_contrasts.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=analysis["contrasts"][0].keys())
        writer.writeheader()
        writer.writerows(analysis["contrasts"])
    from figures.gen_fig_experiment3 import plot_actionability, plot_training_curves
    figure_files = plot_actionability(
        output_dir / "actionability_contrasts.csv", output_dir)
    figure_files += plot_training_curves(combined_training_path, output_dir)
    if args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name or f"exp3c-seed{summary['seed']}-analysis",
            mode=args.wandb_mode,
            job_type="experiment3-actionability-analysis",
            tags=["experiment-3", "actionability", "analysis"],
            config={"gate_definition": summary["gate_definition"]},
        )
        columns = list(analysis["contrasts"][0])
        table = wandb.Table(columns=columns)
        for row in analysis["contrasts"]:
            table.add_data(*(row[column] for column in columns))
        run.log({
            "actionability/gate_passed": int(summary["actionability_gate_passed"]),
            "actionability/strong_result": int(summary["strong_actionability_result"]),
            "actionability/contrasts": table,
            "actionability/future_nll_figure": wandb.Image(str(figure_files[0])),
            "actionability/training_curve_figure": wandb.Image(str(figure_files[3])),
        })
        summary["wandb"] = {"run_id": run.id, "run_url": run.url}
        atomic_write_json(output_dir / "actionability_results.json", summary)
        artifact = wandb.Artifact("experiment3-actionability-analysis", type="experiment-results")
        artifact.add_file(str(output_dir / "actionability_results.json"))
        artifact.add_file(str(output_dir / "actionability_contrasts.csv"))
        artifact.add_file(str(combined_training_path))
        for path in figure_files:
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        run.finish()
    print(json.dumps(summary, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--seed", required=True, type=int)
    select.add_argument("--discovery-results", required=True)
    select.add_argument("--signal-summary", required=True)
    select.add_argument("--temporal-summary", required=True)
    select.add_argument("--parent-checkpoint", required=True)
    select.add_argument("--output", required=True)
    select.add_argument("--cross-seed-replication", action="store_true")
    select.add_argument("--seed42-actionability-summary", default=None)
    select.set_defaults(func=select_command)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--role", required=True, choices=BRANCH_ROLES)
    evaluate.add_argument("--branch-manifest", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--artifact", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--dtype", default="bf16")
    evaluate.add_argument("--batch-size", type=int, default=1)
    add_wandb_args(evaluate)
    evaluate.set_defaults(func=evaluate_command)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--results-root", required=True)
    analyze.add_argument("--training-root", required=True)
    analyze.add_argument("--output-dir", required=True)
    add_wandb_args(analyze)
    analyze.set_defaults(func=analyze_command)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
