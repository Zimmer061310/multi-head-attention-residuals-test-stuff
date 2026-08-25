#!/usr/bin/env python3
"""Experiment 2: exhaustive mixed-width contiguous MHAR boundary search."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Attention-Residuals"))

from experiment1_partition_compatibility import (  # noqa: E402
    append_jsonl,
    atomic_write_json,
    dtype_from_name,
    git_commit,
    load_fixed_eval_artifact,
    load_jsonl,
    materialize_command,
    sha256_file,
    sha256_path,
    utc_now,
)
from mhar_partition import (  # noqa: E402
    generate_adjacent_merge_partitions,
    merged_boundaries,
    mixed_partition_from_merges,
    mixed_partition_id,
    mixed_segment_widths,
    parse_mixed_partition_id,
)


EXPECTED_MIXED_PARTITIONS = 495
NATIVE_H16_ID = "native_h16"
DEFAULT_WANDB_PROJECT = "MHAR Stuff"


def add_wandb_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")


def wandb_init(args, *, config, job_type, default_name, run_id=None):
    if args.wandb_mode == "disabled":
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        id=run_id,
        resume="allow" if run_id else None,
        name=args.wandb_run_name or default_name,
        job_type=job_type,
        tags=["experiment-2", "mhar", "h16", "mixed-width", job_type],
        mode=args.wandb_mode,
        config=config,
        settings=wandb.Settings(x_disable_stats=args.wandb_mode == "offline"),
    )


def validate_h16_model(model, allow_nonstandard: bool) -> dict[str, Any]:
    config = model.config
    observed = {
        "attnres_mode": getattr(config, "attnres_mode", None),
        "attnres_num_heads": getattr(config, "attnres_num_heads", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "head_dim": getattr(config, "head_dim", None),
    }
    required = {
        "attnres_mode": "full_mh",
        "attnres_num_heads": 16,
        "hidden_size": 1280,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 5120,
        "head_dim": 80,
    }
    if observed["attnres_mode"] != "full_mh" or observed["attnres_num_heads"] != 16:
        raise ValueError(
            "Experiment 2 requires a native H=16 full_mh checkpoint; observed "
            f"mode={observed['attnres_mode']!r}, H={observed['attnres_num_heads']!r}")
    mismatches = {
        key: {"required": value, "observed": observed[key]}
        for key, value in required.items() if observed[key] != value
    }
    if mismatches and not allow_nonstandard:
        raise ValueError(
            "checkpoint is not the preregistered 1B architecture: "
            + json.dumps(mismatches, sort_keys=True))
    return {"observed": observed, "required": required, "mismatches": mismatches}


def load_model(checkpoint: Path, device, dtype, allow_nonstandard):
    from modeling_qwen3_attnres import Qwen3AttnResForCausalLM, enable_fused_mhar

    try:
        enable_fused_mhar(False)
    except RuntimeError:
        pass
    model = Qwen3AttnResForCausalLM.from_pretrained(str(checkpoint), dtype=dtype)
    model = model.to(device=device).eval()
    return model, validate_h16_model(model, allow_nonstandard)


@torch.inference_mode()
def check_native_h16_parity(model, input_ids, device, atol, rtol):
    sample = input_ids[:1].to(device=device, dtype=torch.long)
    model.set_mhar_mixed_partition(None)
    native = model(input_ids=sample, use_cache=False).logits.float()
    pure_h16 = mixed_partition_from_merges((), num_atomic_blocks=16)
    model.set_mhar_mixed_partition(pure_h16)
    eager = model(input_ids=sample, use_cache=False).logits.float()
    difference = (native - eager).abs()
    if not torch.allclose(native, eager, atol=atol, rtol=rtol):
        raise RuntimeError(
            "native H16 parity failed: "
            f"max_abs={difference.max().item():.6e}, atol={atol}, rtol={rtol}")
    model.set_mhar_mixed_partition(None)
    return {
        "max_abs_logit_error": difference.max().item(),
        "atol": atol,
        "rtol": rtol,
    }


@torch.inference_mode()
def evaluate_tokens(model, input_ids, partition, *, batch_size, device, record_sequences):
    model.set_mhar_mixed_partition(partition)
    total_nll = 0.0
    valid_tokens = 0
    sequence_nlls: list[float] = []
    uses_cuda = device.type == "cuda" and torch.cuda.is_available()
    if uses_cuda:
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
        if record_sequences:
            sequence_nlls.extend(losses.mean(dim=1).cpu().tolist())
    if uses_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "total_nll": total_nll,
        "valid_tokens": valid_tokens,
        "nll": total_nll / valid_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": valid_tokens / elapsed,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if uses_cuda else None),
        "peak_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if uses_cuda else None),
        "sequence_nlls": sequence_nlls if record_sequences else None,
    }


def load_selection(path: Path) -> tuple[list[tuple[tuple[int, ...], ...]], dict[str, list[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    partitions = []
    roles = {}
    for candidate in payload["candidates"]:
        if candidate["partition_id"] == NATIVE_H16_ID:
            continue
        partitions.append(parse_mixed_partition_id(candidate["partition_id"]))
        roles[candidate["partition_id"]] = candidate["roles"]
    return partitions, roles


def evaluation_partitions(args):
    if args.split == "discovery":
        partitions = list(generate_adjacent_merge_partitions(16, 4))
        if args.smoke_limit is not None:
            if not 1 <= args.smoke_limit <= len(partitions):
                raise ValueError("--smoke-limit must be between 1 and 495")
            partitions = partitions[:args.smoke_limit]
        return partitions, {}, None
    if not args.selection_manifest:
        raise ValueError("--selection-manifest is required for confirmation")
    path = Path(args.selection_manifest).resolve()
    partitions, roles = load_selection(path)
    return partitions, roles, sha256_file(path)


def evaluate_command(args):
    artifact_path = Path(args.artifact).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload, artifact_hash = load_fixed_eval_artifact(artifact_path)
    input_ids = payload[f"{args.split}_input_ids"]
    if input_ids.shape[1] != 1024 and not args.allow_nonstandard_model:
        raise ValueError("primary Experiment 2 requires sequence length 1024")
    partitions, roles, selection_hash = evaluation_partitions(args)
    checkpoint_hash = sha256_path(checkpoint)
    device = torch.device(args.device)
    model, validation = load_model(
        checkpoint, device, dtype_from_name(args.dtype), args.allow_nonstandard_model)
    parity = check_native_h16_parity(
        model, input_ids, device, args.parity_atol, args.parity_rtol)

    results_path = output_dir / f"{args.split}_results.jsonl"
    manifest_path = output_dir / f"{args.split}_run_manifest.json"
    identity = {
        "experiment": 2,
        "split": args.split,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_hash,
        "selection_manifest_sha256": selection_hash,
        "dtype": args.dtype,
        "device_type": device.type,
        "batch_size": args.batch_size,
        "mixed_partition_count": len(partitions),
        "source_commit": git_commit(),
        "routing_site_count": 2 * validation["observed"]["num_hidden_layers"] + 1,
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["run_identity"] != identity:
            raise RuntimeError("existing run manifest differs; use a new output directory")
    else:
        manifest = {
            "created_at": utc_now(),
            "run_identity": identity,
            "architecture_validation": validation,
            "native_h16_parity": parity,
            "fixed_data_metadata": payload.get("metadata", {}),
        }
        atomic_write_json(manifest_path, manifest)

    args.wandb_group = args.wandb_group or f"mhar-exp2-{artifact_hash[:10]}"
    tracked_before = "wandb" in manifest
    run = wandb_init(
        args,
        config={**identity, "architecture": validation["observed"], "parity": parity},
        job_type=args.split,
        default_name=f"exp2-{args.split}-{artifact_hash[:8]}",
        run_id=manifest.get("wandb", {}).get("run_id"),
    )
    if run is not None:
        manifest["wandb"] = {
            "project": args.wandb_project,
            "group": args.wandb_group,
            "run_id": run.id,
            "run_name": run.name,
            "run_url": run.url,
        }
        atomic_write_json(manifest_path, manifest)
        if not tracked_before:
            import wandb

            fixed_artifact = wandb.Artifact(
                f"mhar-exp2-fixed-eval-{artifact_hash[:12]}",
                type="dataset",
                metadata={
                    "sha256": artifact_hash,
                    "discovery_shape": list(payload["discovery_input_ids"].shape),
                    "confirmation_shape": list(payload["confirmation_input_ids"].shape),
                },
            )
            fixed_artifact.add_file(str(artifact_path), name=artifact_path.name)
            sidecar = artifact_path.with_suffix(artifact_path.suffix + ".manifest.json")
            if sidecar.is_file():
                fixed_artifact.add_file(str(sidecar), name=sidecar.name)
            run.log_artifact(fixed_artifact, aliases=[artifact_hash[:12]])

    existing = load_jsonl(results_path) if results_path.exists() else []
    completed = {row["partition_id"] for row in existing}
    if len(completed) != len(existing):
        raise RuntimeError("results contain duplicate partition ids")

    jobs = [(NATIVE_H16_ID, None, ["native_h16"])] + [
        (mixed_partition_id(partition), partition, roles.get(mixed_partition_id(partition), []))
        for partition in partitions
    ]
    for index, (identifier, partition, candidate_roles) in enumerate(jobs):
        if identifier in completed:
            print(f"[{index + 1}/{len(jobs)}] skip {identifier}", flush=True)
            continue
        metrics = evaluate_tokens(
            model, input_ids, partition,
            batch_size=args.batch_size, device=device,
            record_sequences=args.split == "confirmation",
        )
        nll = metrics["nll"]
        row = {
            "created_at": utc_now(),
            "split": args.split,
            "partition_id": identifier,
            "partition": None if partition is None else [list(group) for group in partition],
            "roles": candidate_roles,
            "merged_boundaries": [] if partition is None else list(merged_boundaries(partition)),
            "segment_widths": [80] * 16 if partition is None else list(mixed_segment_widths(partition)),
            "routing_groups": 16 if partition is None else len(partition),
            **metrics,
            "ppl": math.exp(nll) if nll < 20 else float("inf"),
        }
        append_jsonl(results_path, row)
        if run is not None:
            run.log({
                "partition/index": index,
                "partition/nll": nll,
                "partition/ppl": row["ppl"],
                "partition/elapsed_seconds": metrics["elapsed_seconds"],
                "partition/tokens_per_second": metrics["tokens_per_second"],
            }, step=index)
        print(
            f"[{index + 1}/{len(jobs)}] {identifier} NLL={nll:.8f} "
            f"elapsed={metrics['elapsed_seconds']:.1f}s", flush=True)

    rows = load_jsonl(results_path)
    expected = {identifier for identifier, _, _ in jobs}
    if {row["partition_id"] for row in rows} != expected:
        raise RuntimeError("evaluation ended with an incomplete result set")
    if run is not None:
        import wandb

        run.summary["completed_rows"] = len(rows)
        run.summary["results_sha256"] = sha256_file(results_path)
        result_artifact = wandb.Artifact(
            f"mhar-exp2-{args.split}-{artifact_hash[:12]}", type="experiment-results")
        result_artifact.add_file(str(results_path))
        result_artifact.add_file(str(manifest_path))
        run.log_artifact(result_artifact)
        run.finish()
    print(f"completed {args.split}: {results_path}")


def paired_bootstrap(candidate, reference, *, samples=10000, seed=20260824):
    if len(candidate) != len(reference) or not candidate:
        raise ValueError("paired bootstrap requires aligned nonempty sequence losses")
    differences = [a - b for a, b in zip(candidate, reference)]
    generator = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(statistics.fmean(
            differences[generator.randrange(len(differences))]
            for _ in differences))
    means.sort()
    return {
        "mean_delta_nll": statistics.fmean(differences),
        "ci95_low": means[int(0.025 * samples)],
        "ci95_high": means[int(0.975 * samples)],
        "bootstrap_samples": samples,
    }


def build_selection(ranked, discovery_hash):
    roles: dict[str, list[str]] = defaultdict(list)
    roles[NATIVE_H16_ID].append("native_h16")
    for index, row in enumerate(ranked[:5], 1):
        roles[row["partition_id"]].append(f"discovery_top_{index}")
    roles[ranked[len(ranked) // 2]["partition_id"]].append("discovery_median")
    roles[ranked[-1]["partition_id"]].append("discovery_worst")
    return {
        "format_version": 1,
        "discovery_results_sha256": discovery_hash,
        "selection_rule": "native H16, discovery top 5, median, and worst; ties by partition id",
        "candidates": [
            {"partition_id": identifier, "roles": candidate_roles}
            for identifier, candidate_roles in sorted(roles.items())
        ],
    }


def build_partition_choice_rows(native, ranked):
    """Return one display/export row for native H16 plus every mixed choice."""
    rows = [{
        "choice_rank": 1,
        "mixed_rank": 0,
        "partition_id": NATIVE_H16_ID,
        "nll": native["nll"],
        "delta_nll_vs_native_h16": 0.0,
        "merged_boundaries": [],
        "routing_groups": native.get("routing_groups", 16),
        "segment_widths": native.get("segment_widths", [80] * 16),
    }]
    for row in ranked:
        rows.append({
            "choice_rank": row["rank"] + 1,
            "mixed_rank": row["rank"],
            "partition_id": row["partition_id"],
            "nll": row["nll"],
            "delta_nll_vs_native_h16": row["delta_nll_vs_native_h16"],
            "merged_boundaries": row["merged_boundaries"],
            "routing_groups": row.get("routing_groups", 12),
            "segment_widths": row["segment_widths"],
        })
    return rows


def analyze_command(args):
    discovery_path = Path(args.discovery_results).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(discovery_path)
    by_id = {row["partition_id"]: row for row in rows}
    if NATIVE_H16_ID not in by_id:
        raise ValueError("native H16 baseline is missing")
    mixed = [row for row in rows if row["partition_id"] != NATIVE_H16_ID]
    if len(mixed) != EXPECTED_MIXED_PARTITIONS and not args.allow_incomplete:
        raise ValueError(f"expected 495 mixed partitions, got {len(mixed)}")
    native_nll = by_id[NATIVE_H16_ID]["nll"]
    ranked = []
    for rank, row in enumerate(sorted(mixed, key=lambda x: (x["nll"], x["partition_id"])), 1):
        enriched = dict(row)
        enriched["rank"] = rank
        enriched["delta_nll_vs_native_h16"] = row["nll"] - native_nll
        ranked.append(enriched)
    choice_rows = build_partition_choice_rows(by_id[NATIVE_H16_ID], ranked)

    discovery_hash = sha256_file(discovery_path)
    selection = build_selection(ranked, discovery_hash)
    selection_path = output_dir / "confirmation_selection.json"
    if selection_path.exists():
        existing = json.loads(selection_path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in existing.items() if key != "created_at"}
        if comparable != selection:
            raise RuntimeError("frozen selection manifest differs from current discovery results")
        selection = existing
    else:
        selection = {"created_at": utc_now(), **selection}
        atomic_write_json(selection_path, selection)

    all_partitions = generate_adjacent_merge_partitions(16, 4)
    null_rates = {
        edge: statistics.fmean(edge in merged_boundaries(partition) for partition in all_partitions)
        for edge in range(15)
    }
    top_count = min(args.top_count, len(ranked))
    top = ranked[:top_count]
    boundary_rows = []
    for edge in range(15):
        removed = [row["nll"] for row in ranked if edge in row["merged_boundaries"]]
        kept = [row["nll"] for row in ranked if edge not in row["merged_boundaries"]]
        top_rate = statistics.fmean(edge in row["merged_boundaries"] for row in top)
        boundary_rows.append({
            "boundary": edge,
            "association_kept_minus_removed": statistics.fmean(kept) - statistics.fmean(removed),
            "null_removal_rate": null_rates[edge],
            "top_removal_rate": top_rate,
            "top_rate_enrichment": top_rate - null_rates[edge],
        })

    confirmation = None
    if args.confirmation_results:
        confirmation_rows = load_jsonl(Path(args.confirmation_results).resolve())
        confirmation_by_id = {row["partition_id"]: row for row in confirmation_rows}
        reference = confirmation_by_id[NATIVE_H16_ID]
        confirmation = []
        for candidate in selection["candidates"]:
            row = confirmation_by_id[candidate["partition_id"]]
            value = {
                "partition_id": row["partition_id"],
                "roles": candidate["roles"],
                "nll": row["nll"],
                "delta_nll_vs_native_h16": row["nll"] - reference["nll"],
            }
            if row["partition_id"] != NATIVE_H16_ID:
                value.update(paired_bootstrap(
                    row["sequence_nlls"], reference["sequence_nlls"],
                    samples=args.bootstrap_samples))
            confirmation.append(value)

    summary = {
        "created_at": utc_now(),
        "discovery_results_sha256": discovery_hash,
        "mixed_partition_count": len(ranked),
        "native_h16_nll": native_nll,
        "best_partition_id": ranked[0]["partition_id"],
        "best_nll": ranked[0]["nll"],
        "best_delta_nll_vs_native_h16": ranked[0]["delta_nll_vs_native_h16"],
        "median_nll": statistics.median(row["nll"] for row in ranked),
        "worst_partition_id": ranked[-1]["partition_id"],
        "worst_nll": ranked[-1]["nll"],
        "nll_range": ranked[-1]["nll"] - ranked[0]["nll"],
        "boundary_association_is_descriptive_not_causal": True,
        "boundary_associations": boundary_rows,
        "confirmation": confirmation,
    }
    atomic_write_json(output_dir / "analysis.json", summary)
    with (output_dir / "ranked_partitions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "rank", "partition_id", "nll", "delta_nll_vs_native_h16", "ppl",
            "merged_boundaries", "segment_widths", "elapsed_seconds", "tokens_per_second",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranked)
    with (output_dir / "partition_choice_map.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        fields = [
            "choice_rank", "mixed_rank", "partition_id", "nll",
            "delta_nll_vs_native_h16", "merged_boundaries", "routing_groups",
            "segment_widths",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(choice_rows)
    with (output_dir / "boundary_associations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(boundary_rows[0]))
        writer.writeheader()
        writer.writerows(boundary_rows)
    write_figures(output_dir, ranked, boundary_rows, confirmation, choice_rows)
    lines = [
        "# Experiment 2 mixed-width boundary search",
        "",
        f"- Native H16 NLL: {native_nll:.8f}",
        f"- Best mixed: `{ranked[0]['partition_id']}` ({ranked[0]['nll']:.8f})",
        f"- Best delta vs native H16: {ranked[0]['delta_nll_vs_native_h16']:.8f}",
        f"- Median mixed NLL: {summary['median_nll']:.8f}",
        f"- Worst mixed NLL: {ranked[-1]['nll']:.8f}",
        f"- Mixed NLL range: {summary['nll_range']:.8f}",
        "",
        "Boundary associations are descriptive because adjacent non-overlap constraints "
        "couple the boundary choices. Architectural H8 claims require Stage B retraining.",
    ]
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if getattr(args, "wandb_mode", "disabled") != "disabled":
        args.wandb_group = args.wandb_group or f"mhar-exp2-{discovery_hash[:10]}"
        run = wandb_init(
            args,
            config={
                "discovery_results_sha256": discovery_hash,
                "confirmation_results": args.confirmation_results,
                "mixed_partition_count": len(ranked),
                "bootstrap_samples": args.bootstrap_samples,
            },
            job_type="analysis",
            default_name=f"exp2-analysis-{discovery_hash[:8]}",
        )
        if run is not None:
            import wandb

            ranked_table = wandb.Table(columns=[
                "rank", "partition_id", "nll", "delta_nll_vs_native_h16",
                "merged_boundaries", "elapsed_seconds", "tokens_per_second",
            ])
            for row in ranked:
                ranked_table.add_data(*[
                    row.get(column) for column in ranked_table.columns])
            choice_table = wandb.Table(columns=[
                "choice_rank", "mixed_rank", "partition_id", "nll",
                "delta_nll_vs_native_h16", "merged_boundaries", "routing_groups",
            ])
            for row in choice_rows:
                choice_table.add_data(*[
                    row.get(column) for column in choice_table.columns])
            run.log({
                "analysis/ranked_partitions": ranked_table,
                "analysis/partition_choice_map": choice_table,
                "analysis/best_delta_nll_vs_native_h16": summary["best_delta_nll_vs_native_h16"],
                "analysis/nll_range": summary["nll_range"],
                "figures/partition_ranking": wandb.Image(
                    str(output_dir / "fig_partition_ranking.png")),
                "figures/partition_choice_map": wandb.Image(
                    str(output_dir / "fig_partition_choice_map.png")),
                "figures/boundary_associations": wandb.Image(
                    str(output_dir / "fig_boundary_associations.png")),
            })
            analysis_artifact = wandb.Artifact(
                f"mhar-exp2-analysis-{discovery_hash[:12]}", type="analysis")
            for path in sorted(output_dir.iterdir()):
                if path.is_file():
                    analysis_artifact.add_file(str(path), name=path.name)
            run.log_artifact(analysis_artifact)
            run.summary.update({
                "best_partition_id": summary["best_partition_id"],
                "best_delta_nll_vs_native_h16": summary["best_delta_nll_vs_native_h16"],
                "selection_manifest_sha256": sha256_file(selection_path),
            })
            run.finish()
    print(json.dumps(summary, indent=2, sort_keys=True))


def write_figures(output_dir, ranked, boundaries, confirmation, choice_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    scale = 1000.0
    fig, axis = plt.subplots(figsize=(6.0, 3.8))
    axis.plot(
        [row["rank"] for row in ranked],
        [row["delta_nll_vs_native_h16"] * scale for row in ranked],
        color="#0072B2")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Mixed partition rank", ylabel="Delta NLL vs native H16 (millinats/token)",
             title="Exhaustive ranking of 495 mixed-width partitions")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_partition_ranking.png", dpi=300)
    fig.savefig(output_dir / "fig_partition_ranking.pdf")
    plt.close(fig)

    # The ranking curve alone hides which four adjacent H16 boundaries define each
    # mixed choice. This aligned matrix makes all 496 choices directly inspectable.
    choice_x = [row["choice_rank"] for row in choice_rows]
    choice_delta = [row["delta_nll_vs_native_h16"] for row in choice_rows]
    boundary_matrix = [
        [int(boundary in row["merged_boundaries"]) for row in choice_rows]
        for boundary in range(15)
    ]
    median_index = 1 + len(ranked) // 2
    summary_rows = {
        "Native H16": choice_rows[0],
        "Best mixed": choice_rows[1],
        "Median mixed": choice_rows[median_index],
        "Worst mixed": choice_rows[-1],
    }

    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    }):
        fig = plt.figure(figsize=(7.4, 6.0), constrained_layout=True)
        grid = fig.add_gridspec(2, 1, height_ratios=(2.15, 2.65), hspace=0.04)
        ranking_axis = fig.add_subplot(grid[0])
        matrix_axis = fig.add_subplot(grid[1], sharex=ranking_axis)

        ranking_axis.plot(
            choice_x[1:], choice_delta[1:], color="#0072B2", linewidth=1.6,
            zorder=2, label="495 mixed-width choices")
        ranking_axis.scatter(
            choice_x[1:], choice_delta[1:], color="#0072B2", s=7,
            linewidths=0, zorder=3)
        ranking_axis.scatter(
            [choice_x[0]], [choice_delta[0]], color="#D55E00", marker="D",
            s=34, linewidths=0.6, edgecolors="white", zorder=4,
            label="Native H16")
        ranking_axis.axhline(0.0, color="#2E3440", linewidth=0.8, zorder=1)
        ranking_axis.set_ylim(-0.05, max(choice_delta) * 1.08)
        ranking_axis.set_ylabel(r"$\Delta$NLL vs native H16 (nats/token)")
        fig.suptitle("All 496 routing choices at step 2,000", fontsize=13, weight="bold")
        ranking_axis.set_title(
            r"Choices are sorted by discovery-set $\Delta$NLL; lower is better.",
            loc="left", fontsize=8.5, color="#4B5563", pad=7)
        summary_text = "\n".join(
            f"{label}: {row['delta_nll_vs_native_h16']:+.3f}"
            for label, row in summary_rows.items())
        ranking_axis.text(
            0.985, 0.05, summary_text, transform=ranking_axis.transAxes,
            ha="right", va="bottom", fontsize=8.2, linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                  "edgecolor": "#CBD5E1", "alpha": 0.94})
        ranking_axis.legend(loc="upper left", frameon=False, fontsize=8.2)
        ranking_axis.tick_params(axis="x", labelbottom=False)
        ranking_axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.65)

        matrix_axis.imshow(
            boundary_matrix, aspect="auto", interpolation="nearest",
            cmap=ListedColormap(["#F1F5F9", "#0072B2"]), vmin=0, vmax=1,
            extent=(0.5, len(choice_rows) + 0.5, 14.5, -0.5))
        matrix_axis.axvline(1.5, color="#D55E00", linewidth=1.2)
        matrix_axis.set_yticks(
            range(15), [f"{boundary}–{boundary + 1}" for boundary in range(15)])
        tick_values = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 496]
        matrix_axis.set_xticks(tick_values, [str(value) for value in tick_values])
        matrix_axis.set_xlim(0.5, len(choice_rows) + 0.5)
        matrix_axis.set_xlabel(
            r"Choice rank (1 native H16 + 495 mixed choices, sorted by $\Delta$NLL)")
        matrix_axis.set_ylabel("Merged adjacent H16 heads")
        matrix_axis.spines["top"].set_visible(True)
        matrix_axis.spines["right"].set_visible(True)
        matrix_axis.legend(
            handles=[
                Patch(facecolor="#0072B2", label="Pair shares one router"),
                Patch(facecolor="#F1F5F9", edgecolor="#CBD5E1", label="Not merged"),
            ],
            loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2,
            frameon=False, fontsize=8.2)
        matrix_axis.text(
            0.995, -0.19,
            "Each mixed column has four blue cells; native H16 has none.",
            transform=matrix_axis.transAxes, ha="right", va="top", fontsize=7.8,
            color="#4B5563")

        fig.savefig(output_dir / "fig_partition_choice_map.png", dpi=300)
        fig.savefig(output_dir / "fig_partition_choice_map.pdf")
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.0, 3.8))
    axis.bar(
        [row["boundary"] for row in boundaries],
        [row["association_kept_minus_removed"] * scale for row in boundaries],
        color="#009E73")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(xlabel="H16 boundary left atom index", ylabel="Kept minus removed NLL (millinats/token)",
             title="Descriptive boundary association map")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_boundary_associations.png", dpi=300)
    fig.savefig(output_dir / "fig_boundary_associations.pdf")
    plt.close(fig)

    if confirmation:
        fig, axis = plt.subplots(figsize=(7.0, 3.8))
        labels = ["/".join(row["roles"]) for row in confirmation]
        values = [row["delta_nll_vs_native_h16"] * scale for row in confirmation]
        axis.bar(range(len(values)), values, color="#E69F00")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        axis.set_ylabel("Delta confirmation NLL (millinats/token)")
        axis.set_title("Untouched confirmation of frozen candidates")
        fig.tight_layout()
        fig.savefig(output_dir / "fig_confirmation.png", dpi=300)
        fig.savefig(output_dir / "fig_confirmation.pdf")
        plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--output", required=True)
    materialize.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    materialize.add_argument("--dataset-name", default="sample-10BT")
    materialize.add_argument("--dataset-revision", default=None)
    materialize.add_argument("--data-files", default=None)
    materialize.add_argument("--split", default="train")
    materialize.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    materialize.add_argument("--tokenizer-revision", default=None)
    materialize.add_argument("--seq-len", type=int, default=1024)
    materialize.add_argument("--discovery-sequences", type=int, default=512)
    materialize.add_argument("--confirmation-sequences", type=int, default=512)
    materialize.add_argument("--seed", type=int, default=20260824)
    materialize.add_argument("--shuffle-buffer", type=int, default=10000)
    materialize.set_defaults(func=materialize_command)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--artifact", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    evaluate.add_argument("--selection-manifest", default=None)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--dtype", default="bf16")
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--smoke-limit", type=int, default=None)
    evaluate.add_argument("--allow-nonstandard-model", action="store_true")
    evaluate.add_argument("--parity-atol", type=float, default=2e-2)
    evaluate.add_argument("--parity-rtol", type=float, default=2e-3)
    add_wandb_arguments(evaluate)
    evaluate.set_defaults(func=evaluate_command)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--discovery-results", required=True)
    analyze.add_argument("--confirmation-results", default=None)
    analyze.add_argument("--output-dir", required=True)
    analyze.add_argument("--allow-incomplete", action="store_true")
    analyze.add_argument("--top-count", type=int, default=50)
    analyze.add_argument("--bootstrap-samples", type=int, default=10000)
    add_wandb_arguments(analyze)
    analyze.set_defaults(func=analyze_command)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
