#!/usr/bin/env python3
"""Experiment 3A: frozen single-boundary signal evaluation and gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path

import torch

from src.attention_residuals.mhar_partition import (
    mixed_partition_from_merges,
    parse_mixed_partition_id,
)
from src.experiments.experiment1_partition_compatibility import (
    append_jsonl,
    atomic_write_json,
    git_commit,
    sha256_file,
    utc_now,
)
from src.experiments.experiment3_common import (
    DEFAULT_WANDB_PROJECT,
    NATIVE_H16_ID,
    checkpoint_identity,
    evaluate_tokens,
    h16_boundary_candidates,
    load_artifact_split,
    load_jsonl_by_id,
    load_mhar_model,
    paired_bootstrap,
    scored_h16_candidates,
    sha256_json,
    spearman,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/experiment3/protocol.json"


def protocol():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def add_wandb_args(parser):
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"),
        default="disabled")


def init_wandb(args, identity, manifest):
    if args.wandb_mode == "disabled":
        return None
    import wandb

    prior = manifest.get("wandb")
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.wandb_run_name or f"exp3a-seed{args.seed}-step{args.step}-{args.split}",
        id=None if prior is None else prior["run_id"],
        resume="allow" if prior else None,
        mode=args.wandb_mode,
        job_type="experiment3-boundary-signal",
        tags=["experiment-3", "signal", "h16", args.split],
        config=identity,
    )


@torch.inference_mode()
def check_native_parity(model, input_ids, device, *, atol, rtol):
    sample = input_ids[:1, : min(input_ids.shape[1], 16)].to(device=device)
    model.set_mhar_mixed_partition(None)
    native = model(input_ids=sample, use_cache=False).logits.float()
    model.set_mhar_mixed_partition(mixed_partition_from_merges(()))
    eager = model(input_ids=sample, use_cache=False).logits.float()
    difference = (native - eager).abs()
    if not torch.allclose(native, eager, atol=atol, rtol=rtol):
        raise RuntimeError(
            "native H16 parity failed: "
            f"max_abs={difference.max().item():.6e}, atol={atol}, rtol={rtol}")
    model.set_mhar_mixed_partition(None)
    return {
        "max_abs_logit_error": float(difference.max().item()),
        "atol": atol,
        "rtol": rtol,
    }


def apply_candidate(model, row):
    if row["candidate_id"] == NATIVE_H16_ID:
        model.set_mhar_mixed_partition(None)
    else:
        model.set_mhar_mixed_partition(parse_mixed_partition_id(
            row["partition_id"], num_atomic_blocks=16))


def evaluate_command(args):
    spec = protocol()
    if args.seed not in spec["seeds"]:
        raise ValueError(f"seed {args.seed} is not preregistered")
    if args.step not in spec["probe_steps"]:
        raise ValueError(f"step {args.step} is not preregistered")
    artifact = Path(args.artifact).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _, artifact_hash, input_ids = load_artifact_split(artifact, args.split)
    checkpoint_info = checkpoint_identity(checkpoint)
    training_manifest = checkpoint_info["training_manifest"]
    if training_manifest is None:
        raise FileNotFoundError("Experiment 3 requires checkpoint training_manifest.json")
    if int(training_manifest["global_step"]) != args.step:
        raise ValueError("checkpoint global_step does not match --step")
    saved_seed = training_manifest["run_identity"].get("seed")
    if saved_seed != args.seed:
        raise ValueError(f"checkpoint seed {saved_seed} does not match --seed {args.seed}")

    candidates = list(h16_boundary_candidates())
    native_row = candidates.pop(0)
    random.Random(spec["candidate_order_seed"] + args.seed * 10_000 + args.step).shuffle(
        candidates)
    ordered = [native_row, *candidates]
    identity = {
        "experiment": "3a-boundary-signal",
        "seed": args.seed,
        "step": args.step,
        "split": args.split,
        "checkpoint": checkpoint_info,
        "artifact_path": str(artifact),
        "artifact_sha256": artifact_hash,
        "candidate_order": [row["candidate_id"] for row in ordered],
        "candidate_order_seed": spec["candidate_order_seed"],
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "source_commit": git_commit(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
    }
    manifest_path = output_dir / f"{args.split}_run_manifest.json"
    results_path = output_dir / f"{args.split}_results.jsonl"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["run_identity"] != identity:
            raise RuntimeError("existing Experiment 3A run identity differs")
    else:
        manifest = {"created_at": utc_now(), "run_identity": identity}
        atomic_write_json(manifest_path, manifest)

    device = torch.device(args.device)
    model, architecture = load_mhar_model(
        checkpoint, device=device, dtype=args.dtype, required_heads=16)
    parity = check_native_parity(
        model, input_ids, device, atol=args.parity_atol, rtol=args.parity_rtol)
    manifest["architecture"] = architecture
    manifest["native_h16_parity"] = parity
    run = init_wandb(args, identity, manifest)
    if run is not None:
        manifest["wandb"] = {"run_id": run.id, "run_url": run.url}
    atomic_write_json(manifest_path, manifest)

    completed = load_jsonl_by_id(results_path) if results_path.is_file() else {}
    for row in ordered:
        if row["candidate_id"] in completed:
            continue
        apply_candidate(model, row)
        metrics = evaluate_tokens(
            model, input_ids, batch_size=args.batch_size, device=device)
        result = {
            "created_at": utc_now(),
            "seed": args.seed,
            "step": args.step,
            "split": args.split,
            **row,
            **metrics,
        }
        append_jsonl(results_path, result)
        completed[row["candidate_id"]] = result
        if run is not None:
            run.log({
                "candidate/nll": metrics["nll"],
                "candidate/boundary": -1 if row["boundary"] is None else row["boundary"],
                "candidate/tokens_per_second": metrics["tokens_per_second"],
            })
        print(
            f"{args.split} seed={args.seed} step={args.step} "
            f"{row['candidate_id']} NLL={metrics['nll']:.8f}", flush=True)

    model.set_mhar_mixed_partition(None)
    sentinel = evaluate_tokens(
        model, input_ids, batch_size=args.batch_size, device=device)
    initial_native = completed[NATIVE_H16_ID]
    manifest["native_drift_sentinel"] = {
        "initial_nll": initial_native["nll"],
        "final_nll": sentinel["nll"],
        "delta_nll": sentinel["nll"] - initial_native["nll"],
        "exact_match": sentinel["total_nll"] == initial_native["total_nll"],
    }
    manifest["completed_at"] = utc_now()
    manifest["results_sha256"] = sha256_file(results_path)
    atomic_write_json(manifest_path, manifest)
    if run is not None:
        import wandb

        run.summary.update(manifest["native_drift_sentinel"])
        artifact_result = wandb.Artifact(
            f"exp3a-seed{args.seed}-step{args.step}-{args.split}",
            type="experiment-results")
        artifact_result.add_file(str(results_path))
        artifact_result.add_file(str(manifest_path))
        run.log_artifact(artifact_result)
        run.finish()


def select_command(args):
    rows = load_jsonl_by_id(Path(args.discovery_results))
    ranked = scored_h16_candidates(rows)
    if len(ranked) != 15:
        raise ValueError(f"expected 15 removal candidates, found {len(ranked)}")
    selection = {
        "format_version": 1,
        "created_at": utc_now(),
        "source_results": str(Path(args.discovery_results).resolve()),
        "source_results_sha256": sha256_file(Path(args.discovery_results)),
        "source_commit": git_commit(),
        "best": ranked[0]["candidate_id"],
        "worst": ranked[-1]["candidate_id"],
        "top_three": [row["candidate_id"] for row in ranked[:3]],
        "bottom_three": [row["candidate_id"] for row in ranked[-3:]],
        "ranking": [row["candidate_id"] for row in ranked],
        "delta_nll": {row["candidate_id"]: row["delta_nll"] for row in ranked},
    }
    output = Path(args.output).resolve()
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in existing.items() if key != "created_at"}
        expected = {key: value for key, value in selection.items() if key != "created_at"}
        if comparable != expected:
            raise RuntimeError("refusing to overwrite different frozen selection")
        return
    atomic_write_json(output, selection)
    print(json.dumps(selection, indent=2))


def analyze_signal(discovery, confirmation, selection, *, samples, seed):
    discovery_ranked = scored_h16_candidates(discovery)
    confirmation_ranked = scored_h16_candidates(confirmation)
    discovery_by_id = {row["candidate_id"]: row for row in discovery_ranked}
    confirmation_by_id = {row["candidate_id"]: row for row in confirmation_ranked}
    if set(discovery_by_id) != set(confirmation_by_id):
        raise RuntimeError("discovery and confirmation candidate sets differ")
    ids = sorted(discovery_by_id)
    rho = spearman(
        [discovery_by_id[i]["delta_nll"] for i in ids],
        [confirmation_by_id[i]["delta_nll"] for i in ids],
    )
    best = selection["best"]
    worst = selection["worst"]
    contrast = paired_bootstrap(
        confirmation[best]["sequence_nlls"],
        confirmation[worst]["sequence_nlls"],
        samples=samples,
        seed=seed,
    )
    confirmation_top = {
        row["candidate_id"] for row in confirmation_ranked[:3]}
    top_overlap = len(set(selection["top_three"]) & confirmation_top)
    gate = rho >= 0.5 and contrast["ci95_high"] < 0
    return {
        "discovery_confirmation_spearman": rho,
        "discovery_selected_best": best,
        "discovery_selected_worst": worst,
        "best_minus_worst_confirmation": contrast,
        "top_three_overlap_count": top_overlap,
        "top_three_jaccard": top_overlap / len(
            set(selection["top_three"]) | confirmation_top),
        "signal_gate_passed": gate,
        "gate_definition": {
            "minimum_spearman": 0.5,
            "best_minus_worst_confirmation_ci_high_below_zero": True,
        },
        "discovery_ranking": [row["candidate_id"] for row in discovery_ranked],
        "confirmation_ranking": [row["candidate_id"] for row in confirmation_ranked],
        "candidate_rows": [
            {
                "candidate_id": identifier,
                "boundary": discovery_by_id[identifier]["boundary"],
                "discovery_delta_nll": discovery_by_id[identifier]["delta_nll"],
                "confirmation_delta_nll": confirmation_by_id[identifier]["delta_nll"],
            }
            for identifier in ids
        ],
    }


def analyze_command(args):
    spec = protocol()
    discovery_path = Path(args.discovery_results).resolve()
    confirmation_path = Path(args.confirmation_results).resolve()
    selection_path = Path(args.selection).resolve()
    discovery = load_jsonl_by_id(discovery_path)
    confirmation = load_jsonl_by_id(confirmation_path)
    seeds = {row.get("seed") for row in (*discovery.values(), *confirmation.values())}
    steps = {row.get("step") for row in (*discovery.values(), *confirmation.values())}
    if len(seeds) != 1 or len(steps) != 1:
        raise RuntimeError("signal results do not share one seed and checkpoint step")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["source_results_sha256"] != sha256_file(discovery_path):
        raise RuntimeError("selection manifest does not match discovery results")
    analysis = analyze_signal(
        discovery,
        confirmation,
        selection,
        samples=spec["bootstrap_samples"],
        seed=spec["bootstrap_seed"],
    )
    summary = {
        "format_version": 1,
        "created_at": utc_now(),
        "source_commit": git_commit(),
        "seed": next(iter(seeds)),
        "step": next(iter(steps)),
        "discovery_results_sha256": sha256_file(discovery_path),
        "confirmation_results_sha256": sha256_file(confirmation_path),
        "selection_sha256": sha256_file(selection_path),
        **analysis,
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "signal_summary.json", summary)
    with (output_dir / "boundary_signal_map.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary["candidate_rows"][0].keys())
        writer.writeheader()
        writer.writerows(summary["candidate_rows"])
    combined_results = output_dir / "signal_results.jsonl"
    with combined_results.open("w", encoding="utf-8") as handle:
        for split, rows in (("discovery", discovery), ("confirmation", confirmation)):
            for candidate_id in sorted(rows):
                handle.write(json.dumps({**rows[candidate_id], "split": split}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    planned_selection = output_dir / f"signal_selection_step-{summary['step']}.json"
    atomic_write_json(planned_selection, selection)
    discovery_manifest_path = discovery_path.parent / "discovery_run_manifest.json"
    confirmation_manifest_path = confirmation_path.parent / "confirmation_run_manifest.json"
    if not discovery_manifest_path.is_file() or not confirmation_manifest_path.is_file():
        raise FileNotFoundError("signal split run manifests are required")
    discovery_identity = json.loads(
        discovery_manifest_path.read_text(encoding="utf-8"))["run_identity"]
    confirmation_identity = json.loads(
        confirmation_manifest_path.read_text(encoding="utf-8"))["run_identity"]
    if discovery_identity["checkpoint"] != confirmation_identity["checkpoint"]:
        raise RuntimeError("signal splits used different checkpoints")
    if discovery_identity["artifact_sha256"] != confirmation_identity["artifact_sha256"]:
        raise RuntimeError("signal splits used different artifacts")
    atomic_write_json(output_dir / "signal_run_manifest.json", {
        "format_version": 1,
        "seed": summary["seed"],
        "step": summary["step"],
        "source_commit": summary["source_commit"],
        "discovery_results_sha256": summary["discovery_results_sha256"],
        "confirmation_results_sha256": summary["confirmation_results_sha256"],
        "combined_results_sha256": sha256_file(combined_results),
        "selection_sha256": sha256_file(planned_selection),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "checkpoint": discovery_identity["checkpoint"],
        "artifact_sha256": discovery_identity["artifact_sha256"],
        "discovery_run_manifest_sha256": sha256_file(discovery_manifest_path),
        "confirmation_run_manifest_sha256": sha256_file(confirmation_manifest_path),
    })
    from figures.gen_fig_experiment3 import plot_signal
    figure_files = plot_signal(output_dir / "boundary_signal_map.csv", output_dir)
    if args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name or "exp3a-signal-analysis",
            mode=args.wandb_mode,
            job_type="experiment3-signal-analysis",
            tags=["experiment-3", "signal", "analysis"],
            config=summary["gate_definition"],
        )
        table = wandb.Table(columns=list(summary["candidate_rows"][0]))
        for row in summary["candidate_rows"]:
            table.add_data(*(row[column] for column in summary["candidate_rows"][0]))
        run.log({
            "signal/discovery_confirmation_spearman": (
                summary["discovery_confirmation_spearman"]),
            "signal/gate_passed": int(summary["signal_gate_passed"]),
            "signal/top_three_jaccard": summary["top_three_jaccard"],
            "signal/candidates": table,
            "signal/figure": wandb.Image(str(figure_files[0])),
        })
        summary["wandb"] = {"run_id": run.id, "run_url": run.url}
        atomic_write_json(output_dir / "signal_summary.json", summary)
        artifact = wandb.Artifact("experiment3-signal-analysis", type="experiment-results")
        artifact.add_file(str(output_dir / "signal_summary.json"))
        artifact.add_file(str(output_dir / "boundary_signal_map.csv"))
        artifact.add_file(str(combined_results))
        artifact.add_file(str(output_dir / "signal_run_manifest.json"))
        artifact.add_file(str(planned_selection))
        for path in figure_files:
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        run.finish()
    print(json.dumps(summary, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--artifact", required=True)
    evaluate.add_argument("--seed", required=True, type=int)
    evaluate.add_argument("--step", required=True, type=int)
    evaluate.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--dtype", default="bf16")
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--parity-atol", type=float, default=0.0)
    evaluate.add_argument("--parity-rtol", type=float, default=0.0)
    add_wandb_args(evaluate)
    evaluate.set_defaults(func=evaluate_command)

    select = subparsers.add_parser("select")
    select.add_argument("--discovery-results", required=True)
    select.add_argument("--output", required=True)
    select.set_defaults(func=select_command)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--discovery-results", required=True)
    analyze.add_argument("--confirmation-results", required=True)
    analyze.add_argument("--selection", required=True)
    analyze.add_argument("--output-dir", required=True)
    add_wandb_args(analyze)
    analyze.set_defaults(func=analyze_command)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
