#!/usr/bin/env python3
"""Experiment 3E: local H8 boundary-movement landscape and mechanism gate."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

import numpy as np
import torch

from src.attention_residuals.mhar_partition import (
    contiguous_partition_id,
    parse_contiguous_partition_id,
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
    NATIVE_H8_ID,
    checkpoint_identity,
    evaluate_tokens,
    h8_boundary_move_candidates,
    load_artifact_split,
    load_jsonl_by_id,
    load_mhar_model,
    paired_bootstrap,
    spearman,
)
from src.experiments.experiment3_signal import PROTOCOL_PATH, add_wandb_args, protocol


@torch.inference_mode()
def check_native_h8_parity(model, input_ids, device, *, atol, rtol):
    sample = input_ids[:1, : min(input_ids.shape[1], 16)].to(device=device)
    model.set_mhar_contiguous_partition(None)
    native = model(input_ids=sample, use_cache=False).logits.float()
    uniform = tuple((start, start + 160) for start in range(0, 1280, 160))
    model.set_mhar_contiguous_partition(uniform, min_width=120)
    eager = model(input_ids=sample, use_cache=False).logits.float()
    difference = (native - eager).abs()
    if not torch.allclose(native, eager, atol=atol, rtol=rtol):
        raise RuntimeError(
            "native H8 contiguous parity failed: "
            f"max_abs={difference.max().item():.6e}, atol={atol}, rtol={rtol}")
    model.set_mhar_contiguous_partition(None)
    return {
        "max_abs_logit_error": float(difference.max().item()),
        "atol": atol,
        "rtol": rtol,
    }


def apply_candidate(model, row):
    if row["candidate_id"] == NATIVE_H8_ID:
        model.set_mhar_contiguous_partition(None)
    else:
        model.set_mhar_contiguous_partition(row["partition"], min_width=120)


def evaluate_command(args):
    spec = protocol()
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_info = checkpoint_identity(checkpoint)
    training = checkpoint_info["training_manifest"]
    if training is None:
        raise FileNotFoundError("Experiment 3E requires checkpoint training_manifest.json")
    if int(training["global_step"]) != args.step:
        raise ValueError("checkpoint global_step does not match --step")
    if training["run_identity"].get("seed") != args.seed:
        raise ValueError("checkpoint seed does not match --seed")
    artifact = Path(args.artifact).resolve()
    _, artifact_hash, input_ids = load_artifact_split(artifact, args.split)
    candidates = list(h8_boundary_move_candidates(
        spec["landscape"]["offsets"],
        min_width=spec["landscape"]["minimum_segment_width"],
    ))
    native = candidates.pop(0)
    random.Random(spec["candidate_order_seed"] + args.seed + args.step).shuffle(candidates)
    ordered = [native, *candidates]
    serializable = [
        {
            **{key: value for key, value in row.items() if key != "partition"},
            "partition_id": contiguous_partition_id(row["partition"], hidden_size=1280),
        }
        for row in ordered
    ]
    identity = {
        "experiment": "3e-boundary-landscape",
        "seed": args.seed,
        "step": args.step,
        "split": args.split,
        "checkpoint": checkpoint_info,
        "artifact_path": str(artifact),
        "artifact_sha256": artifact_hash,
        "candidate_order": [row["candidate_id"] for row in serializable],
        "offsets": spec["landscape"]["offsets"],
        "minimum_segment_width": spec["landscape"]["minimum_segment_width"],
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "source_commit": git_commit(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{args.split}_run_manifest.json"
    results_path = output_dir / f"{args.split}_results.jsonl"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["run_identity"] != identity:
            raise RuntimeError("existing Experiment 3E run identity differs")
    else:
        manifest = {"created_at": utc_now(), "run_identity": identity}
        atomic_write_json(manifest_path, manifest)

    device = torch.device(args.device)
    model, architecture = load_mhar_model(
        checkpoint, device=device, dtype=args.dtype, required_heads=8)
    parity = check_native_h8_parity(
        model, input_ids, device, atol=args.parity_atol, rtol=args.parity_rtol)
    manifest["architecture"] = architecture
    manifest["native_h8_parity"] = parity
    run = None
    if args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name or f"exp3e-step{args.step}-{args.split}",
            mode=args.wandb_mode,
            job_type="experiment3-landscape-evaluation",
            tags=["experiment-3", "landscape", "h8", args.split],
            config=identity,
        )
        manifest["wandb"] = {"run_id": run.id, "run_url": run.url}
    atomic_write_json(manifest_path, manifest)

    completed = load_jsonl_by_id(results_path) if results_path.is_file() else {}
    serializable_by_id = {row["candidate_id"]: row for row in serializable}
    for raw in ordered:
        candidate_id = raw["candidate_id"]
        if candidate_id in completed:
            continue
        apply_candidate(model, raw)
        metrics = evaluate_tokens(
            model, input_ids, batch_size=args.batch_size, device=device)
        result = {
            "created_at": utc_now(),
            "seed": args.seed,
            "step": args.step,
            "split": args.split,
            **serializable_by_id[candidate_id],
            **metrics,
        }
        append_jsonl(results_path, result)
        completed[candidate_id] = result
        if run is not None:
            run.log({
                "candidate/nll": metrics["nll"],
                "candidate/boundary_index": raw["boundary_index"] or 0,
                "candidate/offset": raw["offset"],
            })
        print(f"{args.split} {candidate_id} NLL={metrics['nll']:.8f}", flush=True)

    model.set_mhar_contiguous_partition(None)
    sentinel = evaluate_tokens(
        model, input_ids, batch_size=args.batch_size, device=device)
    initial_native = completed[NATIVE_H8_ID]
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

        artifact_result = wandb.Artifact(
            f"exp3e-step{args.step}-{args.split}", type="experiment-results")
        artifact_result.add_file(str(results_path))
        artifact_result.add_file(str(manifest_path))
        run.log_artifact(artifact_result)
        run.finish()


def quadratic_loocv_r2(offsets, values):
    x = np.asarray(offsets, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    predictions = np.empty_like(y)
    for index in range(len(x)):
        keep = np.arange(len(x)) != index
        coefficients = np.polyfit(x[keep], y[keep], deg=2)
        predictions[index] = np.polyval(coefficients, x[index])
    denominator = np.square(y - y.mean()).sum()
    if denominator <= 1e-24:
        return 1.0 if np.allclose(predictions, y) else float("-inf")
    return float(1 - np.square(y - predictions).sum() / denominator)


def normalized_roughness(values):
    values = np.asarray(values, dtype=np.float64)
    second = np.abs(values[2:] - 2 * values[1:-1] + values[:-2])
    return float(np.median(second) / (values.max() - values.min() + 1e-12))


def landscape_analysis(discovery, confirmation, *, spec, samples, seed):
    if set(discovery) != set(confirmation):
        raise RuntimeError("landscape discovery and confirmation candidates differ")
    if NATIVE_H8_ID not in discovery:
        raise ValueError("missing native H8 landscape baseline")
    native = {
        "discovery": discovery[NATIVE_H8_ID],
        "confirmation": confirmation[NATIVE_H8_ID],
    }
    curve_rows = []
    boundary_metrics = []
    for boundary_index in range(1, 8):
        by_split = {}
        for split, rows in (("discovery", discovery), ("confirmation", confirmation)):
            points = [{
                "offset": 0,
                "candidate_id": NATIVE_H8_ID,
                "delta_nll": 0.0,
                "nll": native[split]["nll"],
            }]
            for row in rows.values():
                if row.get("boundary_index") == boundary_index:
                    points.append({
                        "offset": row["offset"],
                        "candidate_id": row["candidate_id"],
                        "delta_nll": row["nll"] - native[split]["nll"],
                        "nll": row["nll"],
                    })
            points.sort(key=lambda row: row["offset"])
            if [row["offset"] for row in points] != spec["offsets"]:
                raise RuntimeError(f"boundary {boundary_index} has incomplete offset grid")
            by_split[split] = points
            for point in points:
                curve_rows.append({
                    "split": split,
                    "boundary_index": boundary_index,
                    **point,
                })
        nonzero_discovery = [
            row["delta_nll"] for row in by_split["discovery"] if row["offset"] != 0]
        nonzero_confirmation = [
            row["delta_nll"] for row in by_split["confirmation"] if row["offset"] != 0]
        metric = {
            "boundary_index": boundary_index,
            "split_spearman": spearman(nonzero_discovery, nonzero_confirmation),
        }
        for split in ("discovery", "confirmation"):
            values = [row["delta_nll"] for row in by_split[split]]
            metric[f"{split}_quadratic_loocv_r2"] = quadratic_loocv_r2(
                spec["offsets"], values)
            metric[f"{split}_normalized_roughness"] = normalized_roughness(values)
            metric[f"{split}_best_offset"] = by_split[split][
                int(np.argmin(values))]["offset"]
        boundary_metrics.append(metric)

    discovery_order = sorted(
        discovery,
        key=lambda identifier: (discovery[identifier]["nll"], identifier))
    selected_best = discovery_order[0]
    selected_worst = discovery_order[-1]
    contrast = paired_bootstrap(
        confirmation[selected_best]["sequence_nlls"],
        confirmation[selected_worst]["sequence_nlls"],
        samples=samples,
        seed=seed,
    )
    median_spearman = statistics.median(
        row["split_spearman"] for row in boundary_metrics)
    median_r2_discovery = statistics.median(
        row["discovery_quadratic_loocv_r2"] for row in boundary_metrics)
    median_r2_confirmation = statistics.median(
        row["confirmation_quadratic_loocv_r2"] for row in boundary_metrics)
    conservative_r2 = min(median_r2_discovery, median_r2_confirmation)
    median_roughness_discovery = statistics.median(
        row["discovery_normalized_roughness"] for row in boundary_metrics)
    median_roughness_confirmation = statistics.median(
        row["confirmation_normalized_roughness"] for row in boundary_metrics)
    conservative_roughness = max(
        median_roughness_discovery, median_roughness_confirmation)
    replicable = (
        median_spearman >= spec["minimum_median_split_spearman"]
        and contrast["ci95_high"] < 0)
    soft_compatible = (
        replicable
        and conservative_r2 >= spec["minimum_median_quadratic_cv_r2"]
        and conservative_roughness <= spec["maximum_median_normalized_roughness"])
    classification = (
        "soft-learning-compatible" if soft_compatible
        else "repeatable-but-discrete" if replicable
        else "insufficient-landscape-evidence")
    return {
        "curve_rows": curve_rows,
        "boundary_metrics": boundary_metrics,
        "median_split_spearman": median_spearman,
        "median_quadratic_loocv_r2": conservative_r2,
        "median_normalized_roughness": conservative_roughness,
        "discovery_selected_best": selected_best,
        "discovery_selected_worst": selected_worst,
        "best_minus_worst_confirmation": contrast,
        "replicable_landscape": replicable,
        "soft_learning_compatible": soft_compatible,
        "mechanism_classification": classification,
        "gate_definition": spec,
    }


def analyze_command(args):
    full_spec = protocol()
    discovery_path = Path(args.discovery_results).resolve()
    confirmation_path = Path(args.confirmation_results).resolve()
    discovery = load_jsonl_by_id(discovery_path)
    confirmation = load_jsonl_by_id(confirmation_path)
    analysis = landscape_analysis(
        discovery,
        confirmation,
        spec=full_spec["landscape"],
        samples=full_spec["bootstrap_samples"],
        seed=full_spec["bootstrap_seed"],
    )
    summary = {
        "format_version": 1,
        "created_at": utc_now(),
        "source_commit": git_commit(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "discovery_results_sha256": sha256_file(discovery_path),
        "confirmation_results_sha256": sha256_file(confirmation_path),
        **{key: value for key, value in analysis.items()
           if key not in {"curve_rows", "boundary_metrics"}},
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "boundary_landscape_summary.json", summary)
    with (output_dir / "boundary_landscape_curves.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=analysis["curve_rows"][0].keys())
        writer.writeheader()
        writer.writerows(analysis["curve_rows"])
    with (output_dir / "boundary_landscape_metrics.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=analysis["boundary_metrics"][0].keys())
        writer.writeheader()
        writer.writerows(analysis["boundary_metrics"])
    if args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name or "exp3e-landscape-analysis",
            mode=args.wandb_mode,
            job_type="experiment3-landscape-analysis",
            tags=["experiment-3", "landscape", "analysis"],
            config=summary["gate_definition"],
        )
        columns = list(analysis["boundary_metrics"][0])
        table = wandb.Table(columns=columns)
        for row in analysis["boundary_metrics"]:
            table.add_data(*(row[column] for column in columns))
        run.log({
            "landscape/median_split_spearman": summary["median_split_spearman"],
            "landscape/median_quadratic_cv_r2": summary["median_quadratic_loocv_r2"],
            "landscape/median_roughness": summary["median_normalized_roughness"],
            "landscape/soft_compatible": int(summary["soft_learning_compatible"]),
            "landscape/boundaries": table,
        })
        summary["wandb"] = {"run_id": run.id, "run_url": run.url}
        atomic_write_json(output_dir / "boundary_landscape_summary.json", summary)
        artifact = wandb.Artifact("experiment3-landscape-analysis", type="experiment-results")
        for name in (
            "boundary_landscape_summary.json",
            "boundary_landscape_curves.csv",
            "boundary_landscape_metrics.csv",
        ):
            artifact.add_file(str(output_dir / name))
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

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--discovery-results", required=True)
    analyze.add_argument("--confirmation-results", required=True)
    analyze.add_argument("--output-dir", required=True)
    add_wandb_args(analyze)
    analyze.set_defaults(func=analyze_command)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
