"""Shared, checkpoint-safe utilities for Experiment 3."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from src.attention_residuals.mhar_partition import (
    contiguous_partition_from_boundaries,
    mixed_partition_from_merges,
    mixed_partition_id,
)
from src.experiments.experiment1_partition_compatibility import (
    atomic_write_json,
    dtype_from_name,
    load_fixed_eval_artifact,
    sha256_path,
)


NATIVE_H16_ID = "native-h16"
NATIVE_H8_ID = "native-h8"
DEFAULT_WANDB_PROJECT = "MHAR Stuff"


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def h16_boundary_candidates() -> tuple[dict[str, Any], ...]:
    rows = [{
        "candidate_id": NATIVE_H16_ID,
        "boundary": None,
        "partition_id": None,
        "routing_groups": 16,
    }]
    for boundary in range(15):
        partition = mixed_partition_from_merges((boundary,), num_atomic_blocks=16)
        rows.append({
            "candidate_id": f"remove-{boundary:02d}",
            "boundary": boundary,
            "partition_id": mixed_partition_id(partition),
            "routing_groups": 15,
        })
    return tuple(rows)


def h8_boundary_move_candidates(
    offsets: Iterable[int] = (-40, -30, -20, -10, 0, 10, 20, 30, 40),
    *,
    hidden_size: int = 1280,
    group_width: int = 160,
    min_width: int = 120,
) -> tuple[dict[str, Any], ...]:
    offsets = tuple(int(value) for value in offsets)
    if 0 not in offsets:
        raise ValueError("boundary-move offsets must include zero")
    native_boundaries = tuple(range(group_width, hidden_size, group_width))
    native = contiguous_partition_from_boundaries(
        native_boundaries,
        hidden_size=hidden_size,
        num_groups=len(native_boundaries) + 1,
        min_width=min_width,
    )
    rows = [{
        "candidate_id": NATIVE_H8_ID,
        "boundary_index": None,
        "native_location": None,
        "offset": 0,
        "location": None,
        "partition": native,
    }]
    for boundary_index, native_location in enumerate(native_boundaries, 1):
        for offset in offsets:
            if offset == 0:
                continue
            moved = list(native_boundaries)
            moved[boundary_index - 1] = native_location + offset
            partition = contiguous_partition_from_boundaries(
                moved,
                hidden_size=hidden_size,
                num_groups=len(native_boundaries) + 1,
                min_width=min_width,
            )
            rows.append({
                "candidate_id": f"boundary-{boundary_index:02d}-offset-{offset:+04d}",
                "boundary_index": boundary_index,
                "native_location": native_location,
                "offset": offset,
                "location": native_location + offset,
                "partition": partition,
            })
    return tuple(rows)


def validate_mhar_model(model, *, required_heads: int) -> dict[str, Any]:
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
        "attnres_num_heads": required_heads,
        "hidden_size": 1280,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 5120,
    }
    mismatch = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected if observed[key] != expected[key]
    }
    if mismatch:
        raise ValueError(f"Experiment 3 checkpoint mismatch: {json.dumps(mismatch)}")
    return observed


def load_mhar_model(checkpoint: Path, *, device, dtype: str, required_heads: int):
    from src.attention_residuals.modeling_qwen3_attnres import (
        Qwen3AttnResForCausalLM,
        enable_fused_mhar,
    )

    try:
        enable_fused_mhar(False)
    except RuntimeError:
        pass
    model = Qwen3AttnResForCausalLM.from_pretrained(
        str(checkpoint), dtype=dtype_from_name(dtype))
    model = model.to(device=device).eval()
    return model, validate_mhar_model(model, required_heads=required_heads)


@torch.inference_mode()
def evaluate_tokens(model, input_ids, *, batch_size: int, device) -> dict[str, Any]:
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
            labels.reshape(-1),
            reduction="none",
        ).view(labels.shape)
        total_nll += losses.sum().item()
        valid_tokens += losses.numel()
        sequence_nlls.extend(losses.mean(dim=1).cpu().tolist())
    if uses_cuda:
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
            torch.cuda.max_memory_allocated(device) if uses_cuda else None),
    }


def average_ranks(values: Iterable[float]) -> list[float]:
    values = list(values)
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


def spearman(values_a: Iterable[float], values_b: Iterable[float]) -> float:
    values_a = list(values_a)
    values_b = list(values_b)
    if len(values_a) != len(values_b) or len(values_a) < 2:
        raise ValueError("Spearman inputs must have the same length of at least two")
    ranks_a = average_ranks(values_a)
    ranks_b = average_ranks(values_b)
    value = float(np.corrcoef(ranks_a, ranks_b)[0, 1])
    return value


def paired_bootstrap(
    candidate: Iterable[float],
    reference: Iterable[float],
    *,
    samples: int = 10_000,
    seed: int = 20260828,
) -> dict[str, Any]:
    differences = np.asarray(list(candidate), dtype=np.float64) - np.asarray(
        list(reference), dtype=np.float64)
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
        "bootstrap_seed": seed,
    }


def load_jsonl_by_id(path: Path, *, key: str = "candidate_id") -> dict[str, Any]:
    rows: dict[str, Any] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = row[key]
            if identifier in rows and rows[identifier] != row:
                raise RuntimeError(f"conflicting duplicate {key}={identifier!r} in {path}")
            rows[identifier] = row
    return rows


def scored_h16_candidates(rows: dict[str, Any]) -> list[dict[str, Any]]:
    if NATIVE_H16_ID not in rows:
        raise ValueError("missing native H16 result")
    native = rows[NATIVE_H16_ID]
    candidates = []
    for identifier, row in rows.items():
        if identifier == NATIVE_H16_ID:
            continue
        candidate = dict(row)
        candidate["delta_nll"] = row["nll"] - native["nll"]
        candidates.append(candidate)
    return sorted(candidates, key=lambda row: (row["delta_nll"], row["candidate_id"]))


def load_artifact_split(path: Path, split: str):
    payload, digest = load_fixed_eval_artifact(Path(path))
    key = f"{split}_input_ids"
    if key not in payload:
        raise KeyError(f"fixed artifact has no {key}")
    return payload, digest, payload[key]


def checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    manifest_path = checkpoint / "training_manifest.json"
    manifest = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "path": str(checkpoint),
        "sha256": sha256_path(checkpoint),
        "training_manifest": manifest,
    }


def write_summary(path: Path, value: Any) -> None:
    atomic_write_json(Path(path), value)
