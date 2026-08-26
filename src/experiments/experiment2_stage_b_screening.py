#!/usr/bin/env python3
"""Fixed-data evaluation and decision gate for Stage B trained models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]

from figures.gen_fig_stage_b_milestone import plot_summary
from src.attention_residuals.mhar_partition import parse_mixed_partition_id
from src.experiments.experiment1_partition_compatibility import (
    atomic_write_json,
    dtype_from_name,
    git_commit,
    load_fixed_eval_artifact,
    sha256_path,
    utc_now,
)


SCREENING_CONFIG = ROOT / "configs/experiment2/stage-b-screening.json"
DEFAULT_WANDB_PROJECT = "MHAR Stuff"


def load_screening_spec():
    payload = json.loads(SCREENING_CONFIG.read_text(encoding="utf-8"))
    return payload, {row["id"]: row for row in payload["runs"]}


def validate_model(model, row):
    observed = {
        "attnres_mode": getattr(model.config, "attnres_mode", None),
        "attnres_num_heads": getattr(model.config, "attnres_num_heads", None),
        "hidden_size": getattr(model.config, "hidden_size", None),
        "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
        "num_attention_heads": getattr(model.config, "num_attention_heads", None),
        "num_key_value_heads": getattr(model.config, "num_key_value_heads", None),
        "intermediate_size": getattr(model.config, "intermediate_size", None),
    }
    expected = {
        "attnres_mode": "full_mh",
        "attnres_num_heads": row["attnres_heads"],
        "hidden_size": 1280,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 5120,
    }
    mismatch = {
        key: {"expected": value, "observed": observed[key]}
        for key, value in expected.items()
        if observed[key] != value
    }
    if mismatch:
        raise ValueError(f"checkpoint architecture mismatch: {json.dumps(mismatch)}")
    return observed


def load_variant_model(checkpoint, row, device, dtype):
    from src.attention_residuals.modeling_qwen3_attnres import (
        Qwen3AttnResForCausalLM,
        enable_fused_mhar,
    )

    try:
        enable_fused_mhar(False)
    except RuntimeError:
        pass
    model = Qwen3AttnResForCausalLM.from_pretrained(str(checkpoint), dtype=dtype)
    model = model.to(device=device).eval()
    architecture = validate_model(model, row)
    if row["id"].startswith("mixed-"):
        partition = parse_mixed_partition_id(
            row["partition_id"], num_atomic_blocks=16)
        model.set_mhar_mixed_partition(partition)
    else:
        model.set_mhar_mixed_partition(None)
    return model, architecture


@torch.inference_mode()
def evaluate_split(model, input_ids, *, batch_size, device):
    total_nll = 0.0
    valid_tokens = 0
    sequence_nlls = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for offset in range(0, input_ids.shape[0], batch_size):
        batch = input_ids[offset:offset + batch_size].to(
            device=device, dtype=torch.long, non_blocking=True)
        logits = model(input_ids=batch, use_cache=False).logits
        labels = batch[:, 1:]
        losses = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1), reduction="none",
        ).view(labels.shape)
        total_nll += losses.sum().item()
        valid_tokens += losses.numel()
        sequence_nlls.extend(losses.mean(dim=1).cpu().tolist())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    nll = total_nll / valid_tokens
    return {
        "total_nll": total_nll,
        "valid_tokens": valid_tokens,
        "nll": nll,
        "ppl": math.exp(nll) if nll < 20 else float("inf"),
        "sequence_nlls": sequence_nlls,
        "elapsed_seconds": elapsed,
        "tokens_per_second": valid_tokens / elapsed,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
    }


def init_wandb(args, identity, prior=None):
    if args.wandb_mode == "disabled":
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.wandb_run_name,
        id=None if prior is None else prior.get("run_id"),
        resume="allow" if prior else None,
        mode=args.wandb_mode,
        job_type="stage-b-milestone-evaluation",
        tags=["experiment-2", "stage-b", "fixed-eval", args.variant],
        config=identity,
    )


def evaluate_command(args):
    screening, rows = load_screening_spec()
    if args.variant not in rows:
        raise ValueError(f"unknown variant {args.variant!r}")
    row = rows[args.variant]
    checkpoint = Path(args.checkpoint).resolve()
    artifact = Path(args.artifact).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    manifest_path = output_dir / "run_manifest.json"
    payload, artifact_hash = load_fixed_eval_artifact(artifact)
    checkpoint_manifest = json.loads(
        (checkpoint / "training_manifest.json").read_text(encoding="utf-8"))
    if int(checkpoint_manifest["global_step"]) != args.milestone:
        raise ValueError("checkpoint global step does not match requested milestone")
    identity = {
        "experiment": "2-stage-b-screening",
        "variant": args.variant,
        "milestone": args.milestone,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_path(checkpoint),
        "artifact": str(artifact),
        "artifact_sha256": artifact_hash,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "source_commit": git_commit(),
        "screening_seed": screening["seed"],
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["run_identity"] != identity:
            raise RuntimeError("existing evaluation identity differs")
        if result_path.is_file():
            print(f"completed result already exists: {result_path}")
            return
    else:
        manifest = {"created_at": utc_now(), "run_identity": identity}
        atomic_write_json(manifest_path, manifest)

    device = torch.device(args.device)
    model, architecture = load_variant_model(
        checkpoint, row, device, dtype_from_name(args.dtype))
    manifest["architecture"] = architecture
    run = init_wandb(args, identity, manifest.get("wandb"))
    if run is not None:
        manifest["wandb"] = {
            "project": args.wandb_project,
            "group": args.wandb_group,
            "run_id": run.id,
            "run_url": run.url,
        }
    atomic_write_json(manifest_path, manifest)

    splits = {}
    for split in ("discovery", "confirmation"):
        metrics = evaluate_split(
            model, payload[f"{split}_input_ids"],
            batch_size=args.batch_size, device=device)
        splits[split] = metrics
        if run is not None:
            run.log({
                f"{split}/nll": metrics["nll"],
                f"{split}/ppl": metrics["ppl"],
                f"{split}/tokens_per_second": metrics["tokens_per_second"],
            })
        print(f"{args.variant} {split} NLL={metrics['nll']:.8f}", flush=True)
    result = {
        "created_at": utc_now(),
        "variant": args.variant,
        "milestone": args.milestone,
        "routing_groups": row["routing_groups"],
        "partition_id": row.get("partition_id"),
        "merged_boundaries": row["merged_boundaries"],
        "artifact_sha256": artifact_hash,
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "splits": splits,
        "wandb": manifest.get("wandb"),
    }
    atomic_write_json(result_path, result)
    if run is not None:
        import wandb

        artifact_result = wandb.Artifact(
            f"stage-b-{args.variant}-step-{args.milestone}",
            type="experiment-results")
        artifact_result.add_file(str(result_path))
        artifact_result.add_file(str(manifest_path))
        run.log_artifact(artifact_result)
        run.finish()


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def spearman(values_a, values_b):
    ranks_a = average_ranks(values_a)
    ranks_b = average_ranks(values_b)
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def paired_bootstrap(candidate, reference, *, samples=10000, seed=20260827):
    differences = np.asarray(candidate, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64)
    if differences.size == 0:
        raise ValueError("paired bootstrap requires nonempty aligned sequences")
    generator = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = generator.integers(
            0, differences.size, size=(count, differences.size))
        means[start:start + count] = differences[indices].mean(axis=1)
    return {
        "mean_delta_nll": float(differences.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "bootstrap_samples": samples,
    }


def build_contrast(candidate, reference, split="confirmation"):
    metrics_a = candidate["splits"][split]
    metrics_b = reference["splits"][split]
    result = paired_bootstrap(
        metrics_a["sequence_nlls"], metrics_b["sequence_nlls"])
    result.update({
        "candidate": candidate["variant"],
        "reference": reference["variant"],
        "split": split,
        "decisive": result["ci95_high"] < 0 or result["ci95_low"] > 0,
    })
    return result


def analyze_results(results):
    variants = sorted(results)
    discovery_values = [results[v]["splits"]["discovery"]["nll"] for v in variants]
    confirmation_values = [
        results[v]["splits"]["confirmation"]["nll"] for v in variants]
    rank_spearman = spearman(discovery_values, confirmation_values)
    h8 = results["h8"]
    contrasts = [
        build_contrast(results[variant], h8)
        for variant in variants if variant != "h8"
    ]
    h16 = results["h16"]
    contrasts.extend(
        build_contrast(results[variant], h16)
        for variant in variants
        if variant.startswith("mixed-")
    )
    contrasts.append(
        build_contrast(results["mixed-k4-best"], results["mixed-k4-worst"]))

    def ranking(split):
        reference = results["h8"]["splits"][split]["nll"]
        ordered = sorted(
            results.values(),
            key=lambda result: (result["splits"][split]["nll"], result["variant"]),
        )
        return [
            {
                "rank": index,
                "variant": result["variant"],
                "nll": result["splits"][split]["nll"],
                "delta_nll_vs_h8": result["splits"][split]["nll"] - reference,
            }
            for index, result in enumerate(ordered, 1)
        ]

    scientific = [
        contrast
        for contrast in contrasts
        if (
            (contrast["candidate"].startswith("mixed-")
             and contrast["reference"] in {"h8", "h16"})
            or {
                contrast["candidate"], contrast["reference"]
            } == {"mixed-k4-best", "mixed-k4-worst"}
        )
    ]
    reasons = []
    if rank_spearman < 0.5:
        reasons.append("discovery/confirmation rank Spearman is below 0.5")
    if not any(item["decisive"] for item in scientific):
        reasons.append("no preregistered mixed-width contrast has a paired 95% CI excluding zero")
    return {
        "rank_spearman": rank_spearman,
        "discovery_ranking": ranking("discovery"),
        "confirmation_ranking": ranking("confirmation"),
        "contrasts": contrasts,
        "chaos": bool(reasons),
        "chaos_reasons": reasons,
        "decision": "resume_to_5000" if reasons else "stop_at_2000",
        "gate_definition": {
            "minimum_rank_spearman": 0.5,
            "requires_decisive_mixed_contrast": True,
            "decisive_definition": "paired sequence bootstrap 95% CI excludes zero",
        },
    }


def analyze_command(args):
    _, rows = load_screening_spec()
    root = Path(args.results_root).resolve()
    results = {}
    for variant in rows:
        path = root / variant / "result.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing Stage B result: {path}")
        results[variant] = json.loads(path.read_text(encoding="utf-8"))
    hashes = {result["artifact_sha256"] for result in results.values()}
    milestones = {int(result["milestone"]) for result in results.values()}
    if len(hashes) != 1 or milestones != {args.milestone}:
        raise RuntimeError("results do not share one artifact and milestone")
    summary = {
        "format_version": 1,
        "created_at": utc_now(),
        "milestone": args.milestone,
        "artifact_sha256": next(iter(hashes)),
        "source_commit": git_commit(),
        **analyze_results(results),
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    with (output_dir / "ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("split", "rank", "variant", "nll", "delta_nll_vs_h8"))
        writer.writeheader()
        for split in ("discovery", "confirmation"):
            for row in summary[f"{split}_ranking"]:
                writer.writerow({"split": split, **row})
    plot_summary(summary, output_dir)

    run = None
    if args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            job_type="stage-b-milestone-analysis",
            tags=["experiment-2", "stage-b", "milestone-analysis"],
            config=summary["gate_definition"],
        )
        run.log({
            "rank_spearman": summary["rank_spearman"],
            "chaos": int(summary["chaos"]),
        })
        table = wandb.Table(
            columns=["split", "rank", "variant", "nll", "delta_nll_vs_h8"])
        for split in ("discovery", "confirmation"):
            for row in summary[f"{split}_ranking"]:
                table.add_data(
                    split, row["rank"], row["variant"], row["nll"],
                    row["delta_nll_vs_h8"])
        run.log({
            "rankings": table,
            "delta_nll_figure": wandb.Image(str(output_dir / "fig_stage_b_milestone.png")),
        })
        summary["wandb"] = {"run_id": run.id, "run_url": run.url}
        atomic_write_json(summary_path, summary)
        artifact = wandb.Artifact(
            f"stage-b-step-{args.milestone}-analysis", type="experiment-results")
        for name in (
            "summary.json", "ranking.csv", "fig_stage_b_milestone.png",
            "fig_stage_b_milestone.pdf",
        ):
            artifact.add_file(str(output_dir / name))
        run.log_artifact(artifact)
        run.finish()
    print(json.dumps(summary, indent=2))


def add_wandb_args(parser):
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"),
        default="disabled")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--variant", required=True)
    evaluate.add_argument("--milestone", required=True, type=int)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--artifact", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--dtype", default="bf16")
    evaluate.add_argument("--batch-size", type=int, default=1)
    add_wandb_args(evaluate)
    evaluate.set_defaults(func=evaluate_command)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--milestone", required=True, type=int)
    analyze.add_argument("--results-root", required=True)
    analyze.add_argument("--output-dir", required=True)
    add_wandb_args(analyze)
    analyze.set_defaults(func=analyze_command)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
