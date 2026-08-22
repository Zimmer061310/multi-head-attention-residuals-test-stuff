#!/usr/bin/env python3
"""Experiment 1: exhaustive residual partition compatibility in MHAR.

The workflow has four explicit stages:

1. ``materialize`` creates immutable discovery/confirmation token tensors.
2. ``evaluate --split discovery`` evaluates all 105 H=4 partitions.
3. ``analyze`` ranks the discovery results and freezes best/reference/worst.
4. ``evaluate --split confirmation`` evaluates only those selected partitions
   on the untouched confirmation tensor.

Every partition is applied globally at all full_mh routing sites through the
runtime-only model intervention implemented in ``modeling_qwen3_attnres.py``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
ATTNRES_DIR = ROOT / "Attention-Residuals"
sys.path.insert(0, str(ATTNRES_DIR))

from mhar_partition import (  # noqa: E402
    REFERENCE_PARTITION_H4,
    coordinate_distance,
    generate_pair_partitions,
    original_pair_retention,
    parse_partition_id,
    partition_id,
)


ARTIFACT_FORMAT_VERSION = 1
RESULT_FORMAT_VERSION = 1
EXPECTED_DISCOVERY_PARTITIONS = 105


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash one file or a directory tree, including relative file names."""

    path = path.resolve()
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)

    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"cannot hash empty checkpoint directory: {path}")
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with candidate.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def save_fixed_eval_artifact(
    output: Path,
    discovery_input_ids: torch.Tensor,
    confirmation_input_ids: torch.Tensor,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Serialize fixed token tensors and return a hashed sidecar manifest."""

    tensors = {
        "discovery": discovery_input_ids,
        "confirmation": confirmation_input_ids,
    }
    for split, tensor in tensors.items():
        if tensor.ndim != 2:
            raise ValueError(f"{split} input_ids must be rank 2, got {tuple(tensor.shape)}")
        if tensor.shape[0] < 1 or tensor.shape[1] < 2:
            raise ValueError(f"{split} input_ids must contain sequences with at least 2 tokens")
        if tensor.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{split} input_ids must use an integer tensor dtype")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "created_at": utc_now(),
        "metadata": metadata,
        "discovery_input_ids": discovery_input_ids.to(dtype=torch.int32, device="cpu").contiguous(),
        "confirmation_input_ids": confirmation_input_ids.to(dtype=torch.int32, device="cpu").contiguous(),
    }
    torch.save(payload, temporary)
    os.replace(temporary, output)

    artifact_hash = sha256_file(output)
    manifest = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "artifact_path": str(output),
        "artifact_sha256": artifact_hash,
        "created_at": payload["created_at"],
        "discovery_shape": list(discovery_input_ids.shape),
        "confirmation_shape": list(confirmation_input_ids.shape),
        "predicted_tokens": {
            "discovery": int(discovery_input_ids.shape[0] * (discovery_input_ids.shape[1] - 1)),
            "confirmation": int(confirmation_input_ids.shape[0] * (confirmation_input_ids.shape[1] - 1)),
        },
        "metadata": metadata,
    }
    atomic_write_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    return manifest


def load_fixed_eval_artifact(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    artifact_hash = sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported artifact format {payload.get('format_version')!r}; "
            f"expected {ARTIFACT_FORMAT_VERSION}")
    for split in ("discovery", "confirmation"):
        key = f"{split}_input_ids"
        tensor = payload.get(key)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape[1] < 2:
            raise ValueError(f"invalid {key} tensor in {path}")
    return payload, artifact_hash


def materialize_command(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.tokenizer_revision)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")

    if args.data_files:
        matched_files = sorted(glob.glob(args.data_files))
        if not matched_files:
            raise FileNotFoundError(f"--data-files matched no files: {args.data_files}")
        dataset = load_dataset(
            "parquet", data_files=args.data_files, split=args.split, streaming=True)
        dataset_description = {
            "loader": "parquet",
            "data_files": args.data_files,
            "matched_files": [
                {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size}
                for path in matched_files
            ],
        }
    else:
        dataset = load_dataset(
            args.dataset, name=args.dataset_name, split=args.split,
            revision=args.dataset_revision, streaming=True)
        dataset_description = {
            "loader": args.dataset,
            "name": args.dataset_name,
            "revision": args.dataset_revision,
        }
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    required = args.discovery_sequences + args.confirmation_sequences
    sequences: list[list[int]] = []
    token_buffer: list[int] = []
    documents_consumed = 0
    for sample in dataset:
        documents_consumed += 1
        text = sample.get("text") or sample.get("content") or sample.get("wikitext") or ""
        if not text:
            continue
        token_buffer.extend(tokenizer.encode(text, add_special_tokens=False))
        token_buffer.append(tokenizer.eos_token_id)
        while len(token_buffer) >= args.seq_len and len(sequences) < required:
            sequences.append(token_buffer[: args.seq_len])
            del token_buffer[: args.seq_len]
        if len(sequences) == required:
            break
    if len(sequences) != required:
        raise RuntimeError(
            f"dataset ended after {len(sequences)} sequences; required {required}")

    all_input_ids = torch.tensor(sequences, dtype=torch.int32)
    discovery = all_input_ids[: args.discovery_sequences]
    confirmation = all_input_ids[args.discovery_sequences :]
    metadata = {
        "dataset": dataset_description,
        "split": args.split,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "documents_consumed": documents_consumed,
        "sequence_length": args.seq_len,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "tokenizer_commit_hash": tokenizer.init_kwargs.get("_commit_hash"),
        "tokenizer_class": tokenizer.__class__.__name__,
        "eos_token_id": tokenizer.eos_token_id,
        "selection": "single shuffled token stream; discovery precedes confirmation; no overlap",
        "source_commit": git_commit(),
    }
    manifest = save_fixed_eval_artifact(
        Path(args.output), discovery, confirmation, metadata)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def validate_primary_model(model, allow_nonstandard: bool) -> dict[str, Any]:
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
        "attnres_num_heads": 4,
        "hidden_size": 1280,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 5120,
        "head_dim": 80,
    }
    if observed["attnres_mode"] != "full_mh" or observed["attnres_num_heads"] != 4:
        raise ValueError(
            "Experiment 1 requires an H=4 full_mh checkpoint; observed "
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


def load_experiment_model(
    checkpoint: Path,
    device: torch.device,
    dtype: torch.dtype,
    allow_nonstandard: bool,
):
    from modeling_qwen3_attnres import (
        Qwen3AttnResForCausalLM,
        enable_fused_mhar,
    )

    try:
        enable_fused_mhar(False)
    except RuntimeError:
        pass
    model = Qwen3AttnResForCausalLM.from_pretrained(
        str(checkpoint), dtype=dtype)
    model = model.to(device=device)
    model.eval()
    validation = validate_primary_model(model, allow_nonstandard)
    return model, validation


@torch.inference_mode()
def evaluate_input_ids(
    model,
    input_ids: torch.Tensor,
    partition,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[float, int, float]:
    """Return total NLL, valid-token count, and elapsed seconds."""

    model.set_mhar_partition(partition)
    total_nll = 0.0
    valid_tokens = 0
    started = time.perf_counter()
    for offset in range(0, input_ids.shape[0], batch_size):
        batch = input_ids[offset : offset + batch_size].to(
            device=device, dtype=torch.long, non_blocking=True)
        logits = model(input_ids=batch, use_cache=False).logits
        shift_logits = logits[:, :-1].float().reshape(-1, logits.shape[-1])
        shift_labels = batch[:, 1:].reshape(-1)
        total_nll += F.cross_entropy(
            shift_logits, shift_labels, reduction="sum").item()
        valid_tokens += shift_labels.numel()
    return total_nll, valid_tokens, time.perf_counter() - started


@torch.inference_mode()
def check_end_to_end_reference_parity(
    model,
    input_ids: torch.Tensor,
    *,
    device: torch.device,
    atol: float,
    rtol: float,
) -> dict[str, float]:
    sample = input_ids[:1].to(device=device, dtype=torch.long)
    model.set_mhar_partition(None)
    ordinary = model(input_ids=sample, use_cache=False).logits.float()
    model.set_mhar_partition(REFERENCE_PARTITION_H4)
    arbitrary = model(input_ids=sample, use_cache=False).logits.float()
    difference = (arbitrary - ordinary).abs()
    max_abs = difference.max().item()
    scale = ordinary.abs().max().item()
    max_rel_to_scale = max_abs / max(scale, 1e-12)
    if not torch.allclose(arbitrary, ordinary, atol=atol, rtol=rtol):
        raise RuntimeError(
            "end-to-end reference parity failed: "
            f"max_abs={max_abs:.6e}, scale_relative={max_rel_to_scale:.6e}, "
            f"atol={atol}, rtol={rtol}")
    return {
        "max_abs_logit_error": max_abs,
        "max_scale_relative_logit_error": max_rel_to_scale,
        "atol": atol,
        "rtol": rtol,
    }


def select_confirmation_partitions(
    discovery_results: list[dict[str, Any]],
) -> tuple[list[tuple[tuple[int, int], ...]], dict[str, list[str]]]:
    if len(discovery_results) != EXPECTED_DISCOVERY_PARTITIONS:
        raise ValueError(
            f"confirmation requires all 105 discovery results, got {len(discovery_results)}")
    by_id = {row["partition_id"]: row for row in discovery_results}
    if len(by_id) != EXPECTED_DISCOVERY_PARTITIONS:
        raise ValueError("discovery results contain duplicate partition ids")

    best = min(discovery_results, key=lambda row: (row["nll"], row["partition_id"]))
    worst = max(discovery_results, key=lambda row: (row["nll"], row["partition_id"]))
    reference_id = partition_id(REFERENCE_PARTITION_H4)
    if reference_id not in by_id:
        raise ValueError("discovery results do not contain the reference partition")

    roles_by_id: dict[str, list[str]] = defaultdict(list)
    roles_by_id[reference_id].append("reference")
    roles_by_id[best["partition_id"]].append("discovery_best")
    roles_by_id[worst["partition_id"]].append("discovery_worst")
    partitions = [parse_partition_id(value) for value in sorted(roles_by_id)]
    return partitions, dict(roles_by_id)


def evaluation_partitions(
    split: str,
    discovery_results_path: Path | None,
    smoke_limit: int | None,
) -> tuple[list[tuple[tuple[int, int], ...]], dict[str, list[str]], str | None]:
    if split == "discovery":
        partitions = list(generate_pair_partitions(8))
        if smoke_limit is not None:
            if smoke_limit < 1 or smoke_limit > len(partitions):
                raise ValueError("--smoke-limit must be between 1 and 105")
            partitions = partitions[:smoke_limit]
        return partitions, {}, None

    if discovery_results_path is None:
        raise ValueError("--discovery-results is required for confirmation")
    results = load_jsonl(discovery_results_path)
    partitions, roles = select_confirmation_partitions(results)
    return partitions, roles, sha256_file(discovery_results_path)


def evaluate_command(args: argparse.Namespace) -> None:
    artifact_path = Path(args.artifact).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, artifact_hash = load_fixed_eval_artifact(artifact_path)
    input_ids = payload[f"{args.split}_input_ids"]
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if input_ids.shape[1] != 1024 and not args.allow_nonstandard_model:
        raise ValueError(
            f"primary Experiment 1 requires sequence length 1024, got {input_ids.shape[1]}")
    partitions, roles_by_id, discovery_results_hash = evaluation_partitions(
        args.split,
        Path(args.discovery_results).resolve() if args.discovery_results else None,
        args.smoke_limit,
    )

    checkpoint_hash = sha256_path(checkpoint)
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    model, architecture_validation = load_experiment_model(
        checkpoint, device, dtype, args.allow_nonstandard_model)

    parity = check_end_to_end_reference_parity(
        model, input_ids, device=device,
        atol=args.parity_atol, rtol=args.parity_rtol)

    results_path = output_dir / f"{args.split}_results.jsonl"
    run_manifest_path = output_dir / f"{args.split}_run_manifest.json"
    run_identity = {
        "result_format_version": RESULT_FORMAT_VERSION,
        "split": args.split,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_hash,
        "discovery_results_sha256": discovery_results_hash,
        "dtype": args.dtype,
        "device_type": device.type,
        "batch_size": args.batch_size,
        "partition_count": len(partitions),
        "smoke_limit": args.smoke_limit,
        "source_commit": git_commit(),
        "routing_site_count": 2 * architecture_validation["observed"]["num_hidden_layers"] + 1,
    }

    if run_manifest_path.exists():
        existing_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest["run_identity"] != run_identity:
            raise RuntimeError(
                "existing run manifest does not match this invocation; use a new output directory")
    else:
        atomic_write_json(run_manifest_path, {
            "created_at": utc_now(),
            "run_identity": run_identity,
            "architecture_validation": architecture_validation,
            "artifact_metadata": payload.get("metadata", {}),
            "reference_parity": parity,
            "partition_ids": [partition_id(p) for p in partitions],
            "selection_roles": roles_by_id,
        })

    existing_rows = load_jsonl(results_path) if results_path.exists() else []
    completed = {row["partition_id"] for row in existing_rows}
    if len(completed) != len(existing_rows):
        raise RuntimeError("results file contains duplicate partition ids")

    for index, partition in enumerate(partitions, 1):
        identifier = partition_id(partition)
        if identifier in completed:
            print(f"[{index:3d}/{len(partitions)}] skip completed {identifier}", flush=True)
            continue
        total_nll, valid_tokens, elapsed = evaluate_input_ids(
            model, input_ids, partition,
            batch_size=args.batch_size, device=device)
        nll = total_nll / valid_tokens
        retained, retention = original_pair_retention(partition)
        total_distance, mean_distance = coordinate_distance(partition)
        row = {
            "result_format_version": RESULT_FORMAT_VERSION,
            "created_at": utc_now(),
            "split": args.split,
            "partition_id": identifier,
            "partition": [list(pair) for pair in partition],
            "roles": roles_by_id.get(identifier, []),
            "original_pairs_retained": retained,
            "retention": retention,
            "total_coordinate_distance": total_distance,
            "mean_coordinate_distance": mean_distance,
            "total_nll": total_nll,
            "valid_tokens": valid_tokens,
            "nll": nll,
            "ppl": math.exp(nll) if nll < 20 else float("inf"),
            "elapsed_seconds": elapsed,
        }
        append_jsonl(results_path, row)
        print(
            f"[{index:3d}/{len(partitions)}] {identifier} "
            f"NLL={nll:.8f} delta=pending elapsed={elapsed:.1f}s",
            flush=True,
        )

    rows = load_jsonl(results_path)
    expected_ids = {partition_id(p) for p in partitions}
    observed_ids = {row["partition_id"] for row in rows}
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        raise RuntimeError(f"incomplete result set; missing={missing}, extra={extra}")
    print(f"completed {args.split}: {results_path} sha256={sha256_file(results_path)}")


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for offset in range(start, end):
            ranks[indexed[offset][0]] = average_rank
        start = end
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    mean_x, mean_y = statistics.fmean(x), statistics.fmean(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denom_x = sum((a - mean_x) ** 2 for a in x)
    denom_y = sum((b - mean_y) ** 2 for b in y)
    if denom_x == 0 or denom_y == 0:
        return float("nan")
    return numerator / math.sqrt(denom_x * denom_y)


def linear_slope(x: list[float], y: list[float]) -> float:
    mean_x, mean_y = statistics.fmean(x), statistics.fmean(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    if denominator == 0:
        return float("nan")
    return sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / denominator


def grouped_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row["delta_nll"])
    output = []
    for value in sorted(grouped):
        deltas = grouped[value]
        output.append({
            key: value,
            "count": len(deltas),
            "mean_delta_nll": statistics.fmean(deltas),
            "median_delta_nll": statistics.median(deltas),
            "min_delta_nll": min(deltas),
            "max_delta_nll": max(deltas),
        })
    return output


def analyze_discovery_rows(
    rows: list[dict[str, Any]],
    *,
    allow_incomplete: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len({row["partition_id"] for row in rows}) != len(rows):
        raise ValueError("discovery results contain duplicate partition ids")
    if len(rows) != EXPECTED_DISCOVERY_PARTITIONS and not allow_incomplete:
        raise ValueError(f"expected 105 discovery results, got {len(rows)}")

    reference_id = partition_id(REFERENCE_PARTITION_H4)
    by_id = {row["partition_id"]: row for row in rows}
    if reference_id not in by_id:
        raise ValueError("reference partition is missing")
    reference_nll = by_id[reference_id]["nll"]
    ranked = sorted(rows, key=lambda row: (row["nll"], row["partition_id"]))
    enriched = []
    for rank, row in enumerate(ranked, 1):
        value = dict(row)
        value["rank"] = rank
        value["delta_nll"] = value["nll"] - reference_nll
        enriched.append(value)

    distances = [float(row["mean_coordinate_distance"]) for row in enriched]
    nlls = [float(row["nll"]) for row in enriched]
    spearman = pearson(average_ranks(distances), average_ranks(nlls))
    slope = linear_slope(distances, nlls)
    reference_rank = next(row["rank"] for row in enriched if row["partition_id"] == reference_id)
    summary = {
        "created_at": utc_now(),
        "complete_exhaustive_run": len(rows) == EXPECTED_DISCOVERY_PARTITIONS,
        "partition_count": len(rows),
        "reference_partition_id": reference_id,
        "reference_nll": reference_nll,
        "reference_rank": reference_rank,
        "best_partition_id": enriched[0]["partition_id"],
        "best_nll": enriched[0]["nll"],
        "worst_partition_id": enriched[-1]["partition_id"],
        "worst_nll": enriched[-1]["nll"],
        "nll_min": min(nlls),
        "nll_max": max(nlls),
        "nll_mean": statistics.fmean(nlls),
        "nll_median": statistics.median(nlls),
        "nll_range": max(nlls) - min(nlls),
        "spearman_distance_vs_nll": spearman,
        "ols_nll_per_unit_mean_distance": slope,
        "retention_summary": grouped_summary(enriched, "retention"),
        "distance_summary": grouped_summary(enriched, "mean_coordinate_distance"),
    }
    return enriched, summary


def write_ranked_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank", "partition_id", "nll", "delta_nll", "ppl", "total_nll",
        "valid_tokens", "original_pairs_retained", "retention",
        "total_coordinate_distance", "mean_coordinate_distance", "elapsed_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_analysis_markdown(
    path: Path,
    summary: dict[str, Any],
    confirmation_rows: list[dict[str, Any]] | None,
) -> None:
    lines = [
        "# Experiment 1 partition analysis",
        "",
        f"- Partitions evaluated: {summary['partition_count']}",
        f"- Reference rank: {summary['reference_rank']}",
        f"- Reference NLL: {summary['reference_nll']:.8f}",
        f"- Best: `{summary['best_partition_id']}` ({summary['best_nll']:.8f})",
        f"- Worst: `{summary['worst_partition_id']}` ({summary['worst_nll']:.8f})",
        f"- NLL range: {summary['nll_range']:.8f}",
        f"- Spearman(distance, NLL): {summary['spearman_distance_vs_nll']:.6f}",
        f"- OLS NLL per unit mean distance: {summary['ols_nll_per_unit_mean_distance']:.8f}",
        "",
        "## Retention summary",
        "",
        "| Retention | Count | Mean ΔNLL | Median ΔNLL | Min | Max |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["retention_summary"]:
        lines.append(
            f"| {row['retention']:.2f} | {row['count']} | {row['mean_delta_nll']:.8f} | "
            f"{row['median_delta_nll']:.8f} | {row['min_delta_nll']:.8f} | "
            f"{row['max_delta_nll']:.8f} |")
    lines.extend([
        "",
        "## Coordinate-distance summary",
        "",
        "| Mean distance | Count | Mean ΔNLL | Median ΔNLL | Min | Max |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["distance_summary"]:
        lines.append(
            f"| {row['mean_coordinate_distance']:.1f} | {row['count']} | "
            f"{row['mean_delta_nll']:.8f} | {row['median_delta_nll']:.8f} | "
            f"{row['min_delta_nll']:.8f} | {row['max_delta_nll']:.8f} |")
    if confirmation_rows is not None:
        lines.extend([
            "",
            "## Untouched confirmation set",
            "",
            "| Roles | Partition | NLL | PPL |",
            "|---|---|---:|---:|",
        ])
        for row in confirmation_rows:
            roles = ", ".join(row.get("roles", []))
            lines.append(
                f"| {roles} | `{row['partition_id']}` | {row['nll']:.8f} | {row['ppl']:.6f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_command(args: argparse.Namespace) -> None:
    discovery_path = Path(args.discovery_results).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    discovery_rows = load_jsonl(discovery_path)
    ranked, summary = analyze_discovery_rows(
        discovery_rows, allow_incomplete=args.allow_incomplete)
    summary["discovery_results_path"] = str(discovery_path)
    summary["discovery_results_sha256"] = sha256_file(discovery_path)

    confirmation_rows = None
    if args.confirmation_results:
        confirmation_path = Path(args.confirmation_results).resolve()
        confirmation_rows = load_jsonl(confirmation_path)
        summary["confirmation_results_path"] = str(confirmation_path)
        summary["confirmation_results_sha256"] = sha256_file(confirmation_path)

    atomic_write_json(output_dir / "analysis.json", summary)
    write_ranked_csv(output_dir / "ranked_partitions.csv", ranked)
    write_analysis_markdown(output_dir / "analysis.md", summary, confirmation_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize", help="create fixed token artifacts")
    materialize.add_argument("--output", required=True)
    materialize.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    materialize.add_argument("--tokenizer-revision", default=None)
    source = materialize.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset")
    source.add_argument("--data-files")
    materialize.add_argument("--dataset-name", default="default")
    materialize.add_argument("--dataset-revision", default=None)
    materialize.add_argument("--split", default="train")
    materialize.add_argument("--seed", type=int, default=10041)
    materialize.add_argument("--shuffle-buffer", type=int, default=10_000)
    materialize.add_argument("--seq-len", type=int, default=1024)
    materialize.add_argument("--discovery-sequences", type=int, default=512)
    materialize.add_argument("--confirmation-sequences", type=int, default=512)
    materialize.set_defaults(func=materialize_command)

    evaluate = subparsers.add_parser("evaluate", help="evaluate discovery or confirmation")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--artifact", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    evaluate.add_argument("--discovery-results", default=None)
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--dtype", default="bf16")
    evaluate.add_argument("--parity-atol", type=float, default=1e-5)
    evaluate.add_argument("--parity-rtol", type=float, default=1e-5)
    evaluate.add_argument("--allow-nonstandard-model", action="store_true",
                          help="testing only: retain H=4/full_mh but relax the 1B shape gate")
    evaluate.add_argument("--smoke-limit", type=int, default=None,
                          help="testing only: evaluate the first N discovery partitions")
    evaluate.set_defaults(func=evaluate_command)

    analyze = subparsers.add_parser("analyze", help="rank and summarize results")
    analyze.add_argument("--discovery-results", required=True)
    analyze.add_argument("--confirmation-results", default=None)
    analyze.add_argument("--output-dir", required=True)
    analyze.add_argument("--allow-incomplete", action="store_true",
                         help="testing only: analyze fewer than 105 discovery rows")
    analyze.set_defaults(func=analyze_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
