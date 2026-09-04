#!/usr/bin/env python3
"""Frozen HQ8 head-contribution and chunk-alignment interventions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np
import torch

from src.experiments.experiment1_partition_compatibility import (
    atomic_write_json,
    dtype_from_name,
    git_commit,
    load_fixed_eval_artifact,
    sha256_path,
    utc_now,
)
from src.experiments.experiment6_screening import evaluate_split
from src.experiments.experiment8_hybrid_q import (
    Experiment8MHARForCausalLM,
    HybridQueryLinear,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiment9/protocol.json"
ACCEPTED_HQ8_RESULT = (
    ROOT / "results/experiment8/step-2000/hq8/evaluation/result.json"
)
LOCAL_HEADS = tuple(range(0, 16, 2))
GLOBAL_HEADS = tuple(range(1, 16, 2))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_spec() -> dict:
    spec = read_json(CONFIG)
    if spec["training"] is not False or spec["seed"] != 42 or spec["milestone"] != 2000:
        raise ValueError("Experiment 9 must remain a frozen seed-42 step-2000 evaluation")
    family = spec["phase_9a"]["matched_random_family"]
    if family["count"] != 70 or family["removed_heads_per_group"] != 1:
        raise ValueError("matched random family must be the exhaustive 70-mask design")
    if spec["phase_9b"]["derangement_count"] != 32:
        raise ValueError("Experiment 9B must use exactly 32 frozen derangements")
    return spec


def phase_9a_conditions() -> list[dict]:
    conditions = [
        {"id": "hq8-unchanged", "kind": "reference", "removed_heads": []},
        {"id": "zero-local", "kind": "structured", "removed_heads": list(LOCAL_HEADS)},
        {"id": "zero-global", "kind": "structured", "removed_heads": list(GLOBAL_HEADS)},
    ]
    groups = set(range(8))
    for global_removed in itertools.combinations(range(8), 4):
        global_removed = set(global_removed)
        local_removed = groups - global_removed
        removed = sorted(
            [2 * group + 1 for group in global_removed]
            + [2 * group for group in local_removed]
        )
        suffix = "".join(str(group) for group in sorted(global_removed))
        conditions.append({
            "id": f"balanced-g{suffix}",
            "kind": "balanced-random-8",
            "removed_heads": removed,
            "global_removed_groups": sorted(global_removed),
            "local_removed_groups": sorted(local_removed),
        })
    if len(conditions) != 73 or len({row["id"] for row in conditions}) != 73:
        raise AssertionError("Experiment 9A condition construction is not unique")
    return conditions


def phase_9b_conditions(seed: int = 20260910, count: int = 32) -> list[dict]:
    rng = np.random.default_rng(seed)
    permutations: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(permutations) < count:
        permutation = tuple(int(value) for value in rng.permutation(8))
        if any(source == target for target, source in enumerate(permutation)):
            continue
        if permutation in seen:
            continue
        seen.add(permutation)
        permutations.append(permutation)
    return [
        {
            "id": f"derangement-{index:02d}",
            "kind": "local-chunk-derangement",
            "local_chunk_permutation": list(permutation),
        }
        for index, permutation in enumerate(permutations)
    ]


def condition_manifest() -> dict:
    spec = load_spec()
    payload = {
        "format_version": 1,
        "experiment": "9-hq8-head-contribution",
        "created_from_commit": git_commit(),
        "checkpoint_sha256": spec["checkpoint"]["sha256"],
        "artifact_sha256": spec["fixed_evaluation"]["sha256"],
        "phase_9a": phase_9a_conditions(),
        "phase_9b": phase_9b_conditions(
            seed=spec["phase_9b"]["derangement_seed"],
            count=spec["phase_9b"]["derangement_count"],
        ),
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_hq8_model(model) -> None:
    expected = {
        "hidden_size": 1280,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 5120,
        "attnres_mode": "full_mh",
        "attnres_num_heads": 8,
        "experiment8_variant": "hq8",
        "experiment8_hybrid_q_groups": 8,
        "experiment8_local_head_position": "even",
        "experiment8_global_head_position": "odd",
    }
    mismatch = {
        key: {"expected": value, "observed": getattr(model.config, key, None)}
        for key, value in expected.items()
        if getattr(model.config, key, None) != value
    }
    if mismatch:
        raise ValueError(f"accepted HQ8 architecture mismatch: {json.dumps(mismatch)}")
    for index, layer in enumerate(model.model.layers):
        if not isinstance(layer.self_attn.q_proj, HybridQueryLinear):
            raise TypeError(f"layer {index} is not an Experiment 8 hybrid-Q layer")


def validate_checkpoint(checkpoint: Path, spec: dict) -> dict:
    if sha256_path(checkpoint) != spec["checkpoint"]["sha256"]:
        raise ValueError("HQ8 checkpoint content hash differs from preregistration")
    manifest = read_json(checkpoint / "training_manifest.json")
    identity = manifest.get("run_identity", {})
    expected = {
        "global_step": 2000,
        "seed": 42,
        "mode": "full_mh",
        "attnres_heads": 8,
        "num_heads": 16,
        "num_kv_heads": 8,
        "experiment8_variant": "hq8",
        "experiment8_hybrid_q_groups": 8,
        "experiment8_local_head_position": "even",
        "experiment8_global_head_position": "odd",
    }
    observed = {"global_step": manifest.get("global_step"), **identity}
    mismatch = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatch:
        raise ValueError(f"HQ8 training manifest mismatch: {json.dumps(mismatch)}")
    return identity


def mask_head_outputs(
    values: torch.Tensor,
    removed_heads,
    *,
    num_heads: int = 16,
    head_dim: int = 80,
) -> torch.Tensor:
    """Zero complete head slices in the tensor immediately before W_O."""

    removed = tuple(sorted(int(head) for head in removed_heads))
    if values.shape[-1] != num_heads * head_dim:
        raise ValueError("unexpected pre-W_O attention width")
    if len(set(removed)) != len(removed):
        raise ValueError("removed head indices must be unique")
    if any(head < 0 or head >= num_heads for head in removed):
        raise ValueError("removed head index outside the attention-head range")
    keep = values.new_ones(num_heads)
    keep[list(removed)] = 0
    grouped = values.view(*values.shape[:-1], num_heads, head_dim)
    keep_shape = (*((1,) * (values.ndim - 1)), num_heads, 1)
    return (grouped * keep.view(keep_shape)).reshape_as(values)


class HQ8Intervention(AbstractContextManager):
    """Apply a frozen head-output mask and/or local-chunk permutation."""

    def __init__(self, model, *, removed_heads=(), local_chunk_permutation=None):
        self.model = model
        self.removed_heads = tuple(sorted(int(head) for head in removed_heads))
        self.local_chunk_permutation = (
            None
            if local_chunk_permutation is None
            else tuple(int(group) for group in local_chunk_permutation)
        )
        self.handles = []
        if len(set(self.removed_heads)) != len(self.removed_heads):
            raise ValueError("removed head indices must be unique")
        if any(head < 0 or head >= 16 for head in self.removed_heads):
            raise ValueError("removed head index outside [0, 15]")
        if self.local_chunk_permutation is not None:
            if sorted(self.local_chunk_permutation) != list(range(8)):
                raise ValueError("local chunk assignment must be a permutation of 0..7")
            if any(source == target for target, source in enumerate(self.local_chunk_permutation)):
                raise ValueError("Experiment 9B requires a complete derangement")

    def __enter__(self):
        validate_hq8_model(self.model)
        for layer in self.model.model.layers:
            query = layer.self_attn.q_proj
            if query.local_group_permutation is not None:
                raise RuntimeError("an Experiment 9 local permutation is already active")
            query.local_group_permutation = self.local_chunk_permutation
            if self.removed_heads:
                removed = self.removed_heads

                def mask_before_output_projection(module, args, removed=removed):
                    return (mask_head_outputs(args[0], removed), *args[1:])

                self.handles.append(
                    layer.self_attn.o_proj.register_forward_pre_hook(
                        mask_before_output_projection
                    )
                )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        for layer in self.model.model.layers:
            layer.self_attn.q_proj.local_group_permutation = None
        self.handles.clear()
        return False


def bootstrap_delta(candidate: np.ndarray, reference: np.ndarray, *, samples: int, seed: int) -> dict:
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError("paired sequence arrays must be one-dimensional and equal length")
    values = candidate - reference
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        draws[start:start + count] = values[
            rng.integers(0, len(values), size=(count, len(values)))
        ].mean(axis=1)
    return {
        "mean_delta_nll": float(values.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def hierarchical_bootstrap(values: np.ndarray, *, samples: int, seed: int) -> dict:
    if values.ndim != 2:
        raise ValueError("hierarchical values must have shape [condition, sequence]")
    rng = np.random.default_rng(seed)
    conditions, sequences = values.shape
    draws = np.empty(samples)
    for index in range(samples):
        condition_draw = rng.integers(0, conditions, size=conditions)
        sequence_draw = rng.integers(0, sequences, size=sequences)
        draws[index] = values[np.ix_(condition_draw, sequence_draw)].mean()
    return {
        "mean_delta_nll": float(values.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "bootstrap_unit": "derangements_and_paired_sequences",
    }


def result_path(results_root: Path, phase: str, condition_id: str) -> Path:
    return results_root / phase / "conditions" / condition_id / "result.json"


def evaluate_command(args) -> None:
    spec = load_spec()
    manifest = read_json(Path(args.manifest))
    expected_manifest = condition_manifest()
    if manifest.get("content_sha256") != expected_manifest["content_sha256"]:
        raise ValueError("condition manifest differs from reviewed code/config")
    conditions = manifest[args.phase.replace("-", "_")]
    if args.worker_index < 0 or args.worker_index >= args.worker_count:
        raise ValueError("worker index outside worker count")
    assigned = [
        condition for index, condition in enumerate(conditions)
        if index % args.worker_count == args.worker_index
    ]
    checkpoint = Path(args.checkpoint).resolve()
    artifact = Path(args.artifact).resolve()
    output_root = Path(args.output_root).resolve()
    identity = validate_checkpoint(checkpoint, spec)
    payload, artifact_hash = load_fixed_eval_artifact(artifact)
    if artifact_hash != spec["fixed_evaluation"]["sha256"]:
        raise ValueError("fixed artifact hash differs from preregistration")
    device = torch.device(args.device)
    model = Experiment8MHARForCausalLM.from_pretrained(
        str(checkpoint), dtype=dtype_from_name(args.dtype)
    ).to(device=device).eval()
    validate_hq8_model(model)
    for condition in assigned:
        target = result_path(output_root, args.phase, condition["id"])
        if target.exists():
            existing = read_json(target)
            if (
                existing.get("condition") == condition
                and existing.get("manifest_sha256") == manifest["content_sha256"]
                and existing.get("checkpoint_sha256") == spec["checkpoint"]["sha256"]
                and existing.get("artifact_sha256") == artifact_hash
            ):
                continue
            raise ValueError(f"refusing to overwrite mismatched result {target}")
        with HQ8Intervention(
            model,
            removed_heads=condition.get("removed_heads", ()),
            local_chunk_permutation=condition.get("local_chunk_permutation"),
        ):
            splits = {
                split: evaluate_split(
                    model,
                    payload[f"{split}_input_ids"],
                    batch_size=args.batch_size,
                    device=device,
                )
                for split in spec["fixed_evaluation"]["splits"]
            }
        result = {
            "format_version": 1,
            "created_at": utc_now(),
            "experiment": "9-hq8-head-contribution",
            "phase": args.phase,
            "condition": condition,
            "seed": 42,
            "milestone": 2000,
            "checkpoint_sha256": spec["checkpoint"]["sha256"],
            "artifact_sha256": artifact_hash,
            "manifest_sha256": manifest["content_sha256"],
            "source_commit": git_commit(),
            "training_identity": identity,
            "splits": splits,
        }
        atomic_write_json(target, result)
    atomic_write_json(
        output_root / args.phase / f"WORKER_{args.worker_index}_COMPLETE.json",
        {
            "completed_at": utc_now(),
            "worker_index": args.worker_index,
            "worker_count": args.worker_count,
            "condition_ids": [condition["id"] for condition in assigned],
            "manifest_sha256": manifest["content_sha256"],
        },
    )


def load_phase_results(results_root: Path, phase: str, conditions: list[dict]) -> dict[str, dict]:
    results = {}
    for condition in conditions:
        path = result_path(results_root, phase, condition["id"])
        if not path.is_file():
            raise FileNotFoundError(f"missing result for {condition['id']}")
        result = read_json(path)
        if result.get("condition") != condition:
            raise ValueError(f"condition mismatch for {condition['id']}")
        results[condition["id"]] = result
    return results


def reference_gate(reference: dict, spec: dict) -> dict:
    accepted = read_json(ACCEPTED_HQ8_RESULT)
    if accepted["checkpoint_sha256"] != spec["checkpoint"]["sha256"]:
        raise ValueError("accepted Experiment 8 result has a different checkpoint")
    metrics = {}
    passed = True
    for split in spec["fixed_evaluation"]["splits"]:
        current = np.asarray(reference["splits"][split]["sequence_nlls"], dtype=float)
        original = np.asarray(accepted["splits"][split]["sequence_nlls"], dtype=float)
        max_abs = float(np.max(np.abs(current - original)))
        aggregate_abs = abs(reference["splits"][split]["nll"] - accepted["splits"][split]["nll"])
        split_passed = max_abs == 0 and aggregate_abs == 0
        passed = passed and split_passed
        metrics[split] = {
            "max_abs_sequence_nll_error": max_abs,
            "aggregate_nll_error": aggregate_abs,
            "passed": split_passed,
        }
    return {"passed": passed, "splits": metrics}


def analyze_9a(args) -> dict:
    spec = load_spec()
    conditions = phase_9a_conditions()
    results = load_phase_results(Path(args.results_root), "phase-9a", conditions)
    reference = results["hq8-unchanged"]
    reproduction = reference_gate(reference, spec)
    if not reproduction["passed"]:
        raise ValueError("unchanged HQ8 did not exactly reproduce Experiment 8")
    samples = spec["phase_9a"]["bootstrap_samples"]
    base_seed = spec["phase_9a"]["bootstrap_seed"]
    rows = []
    for condition_index, condition in enumerate(conditions):
        result = results[condition["id"]]
        row = {"condition": condition, "splits": {}}
        for split_index, split in enumerate(spec["fixed_evaluation"]["splits"]):
            candidate = np.asarray(result["splits"][split]["sequence_nlls"], dtype=float)
            baseline = np.asarray(reference["splits"][split]["sequence_nlls"], dtype=float)
            row["splits"][split] = {
                "nll": float(result["splits"][split]["nll"]),
                "aggregate_delta_nll": float(
                    result["splits"][split]["nll"] - reference["splits"][split]["nll"]
                ),
                **bootstrap_delta(
                    candidate,
                    baseline,
                    samples=samples,
                    seed=base_seed + condition_index * 2 + split_index,
                ),
            }
        rows.append(row)
    by_id = {row["condition"]["id"]: row for row in rows}
    balanced = [row for row in rows if row["condition"]["kind"] == "balanced-random-8"]
    distribution = {}
    for split in spec["fixed_evaluation"]["splits"]:
        values = np.asarray([row["splits"][split]["aggregate_delta_nll"] for row in balanced])
        distribution[split] = {
            "count": len(values),
            "minimum": float(values.min()),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
            "maximum": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
        for structured_id in ("zero-local", "zero-global"):
            value = by_id[structured_id]["splits"][split]["aggregate_delta_nll"]
            by_id[structured_id]["splits"][split]["balanced_empirical_percentile"] = float(
                100 * np.mean(values <= value)
            )
    local_minus_global_damage = {}
    for split_index, split in enumerate(spec["fixed_evaluation"]["splits"]):
        local_values = np.asarray(
            results["zero-local"]["splits"][split]["sequence_nlls"], dtype=float
        )
        global_values = np.asarray(
            results["zero-global"]["splits"][split]["sequence_nlls"], dtype=float
        )
        local_minus_global_damage[split] = {
            "aggregate_delta_nll": float(
                results["zero-local"]["splits"][split]["nll"]
                - results["zero-global"]["splits"][split]["nll"]
            ),
            **bootstrap_delta(
                local_values,
                global_values,
                samples=samples,
                seed=base_seed + 10_000 + split_index,
            ),
            "interpretation": "positive means local heads are more important; negative means global heads are more important",
        }
    gate_spec = spec["phase_9a"]["local_contribution_gate"]
    discovery = by_id["zero-local"]["splits"]["discovery"]
    confirmation = by_id["zero-local"]["splits"]["confirmation"]
    checks = {
        "discovery_delta_positive": discovery["aggregate_delta_nll"] > 0,
        "confirmation_delta_at_least_margin": confirmation["aggregate_delta_nll"] >= gate_spec["practical_margin_nll"],
        "confirmation_ci_lower_above_zero": confirmation["ci95_low"] > 0,
    }
    summary = {
        "format_version": 1,
        "created_at": utc_now(),
        "experiment": "9-hq8-head-contribution",
        "phase": "9A-contribution",
        "reference_reproduction": reproduction,
        "structured": {key: by_id[key] for key in ("zero-local", "zero-global")},
        "local_minus_global_damage": local_minus_global_damage,
        "balanced_random_8_distribution": distribution,
        "conditions": rows,
        "local_contribution_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "rule": gate_spec["rule"],
        },
        "interpretation_limits": spec["interpretation_limits"],
    }
    return write_analysis(args, summary, "phase-9a")


def analyze_9b(args) -> dict:
    spec = load_spec()
    phase_9a_summary = read_json(Path(args.phase_9a_summary))
    if not phase_9a_summary["local_contribution_gate"]["passed"]:
        raise ValueError("Experiment 9B is not authorized by the frozen 9A gate")
    conditions = phase_9b_conditions(
        seed=spec["phase_9b"]["derangement_seed"],
        count=spec["phase_9b"]["derangement_count"],
    )
    results = load_phase_results(Path(args.results_root), "phase-9b", conditions)
    reference = load_phase_results(
        Path(args.results_root), "phase-9a", [phase_9a_conditions()[0]]
    )["hq8-unchanged"]
    samples = spec["phase_9b"]["bootstrap_samples"]
    base_seed = spec["phase_9b"]["bootstrap_seed"]
    rows = []
    for condition_index, condition in enumerate(conditions):
        result = results[condition["id"]]
        row = {"condition": condition, "splits": {}}
        for split_index, split in enumerate(spec["fixed_evaluation"]["splits"]):
            candidate = np.asarray(result["splits"][split]["sequence_nlls"], dtype=float)
            baseline = np.asarray(reference["splits"][split]["sequence_nlls"], dtype=float)
            row["splits"][split] = {
                "nll": float(result["splits"][split]["nll"]),
                "aggregate_delta_nll": float(
                    result["splits"][split]["nll"] - reference["splits"][split]["nll"]
                ),
                **bootstrap_delta(
                    candidate,
                    baseline,
                    samples=samples,
                    seed=base_seed + condition_index * 2 + split_index,
                ),
            }
        rows.append(row)
    aggregate = {}
    for split_index, split in enumerate(spec["fixed_evaluation"]["splits"]):
        baseline = np.asarray(reference["splits"][split]["sequence_nlls"], dtype=float)
        values = np.stack([
            np.asarray(results[row["condition"]["id"]]["splits"][split]["sequence_nlls"], dtype=float)
            - baseline
            for row in rows
        ])
        condition_means = values.mean(axis=1)
        aggregate[split] = {
            **hierarchical_bootstrap(
                values, samples=samples, seed=base_seed + 1000 + split_index
            ),
            "condition_median_delta_nll": float(np.median(condition_means)),
            "condition_q25_delta_nll": float(np.quantile(condition_means, 0.25)),
            "condition_q75_delta_nll": float(np.quantile(condition_means, 0.75)),
            "positive_fraction": float(np.mean(condition_means > 0)),
            "minimum_condition_delta_nll": float(condition_means.min()),
            "maximum_condition_delta_nll": float(condition_means.max()),
        }
    rule = spec["phase_9b"]["alignment_evidence_rule"]
    checks = {
        "discovery_mean_positive": aggregate["discovery"]["mean_delta_nll"] > 0,
        "confirmation_mean_at_least_margin": aggregate["confirmation"]["mean_delta_nll"] >= rule["practical_margin_nll"],
        "confirmation_ci_lower_above_zero": aggregate["confirmation"]["ci95_low"] > 0,
        "confirmation_positive_fraction_sufficient": aggregate["confirmation"]["positive_fraction"] >= rule["minimum_positive_fraction"],
    }
    summary = {
        "format_version": 1,
        "created_at": utc_now(),
        "experiment": "9-hq8-head-contribution",
        "phase": "9B-alignment",
        "conditions": rows,
        "aggregate": aggregate,
        "alignment_evidence": {
            "passed": all(checks.values()),
            "checks": checks,
            "rule": rule["rule"],
        },
        "interpretation_limits": spec["interpretation_limits"],
    }
    return write_analysis(args, summary, "phase-9b")


def write_analysis(args, summary: dict, directory_name: str) -> dict:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run = None
    if args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            job_type=f"experiment9-{directory_name}-analysis",
            tags=["experiment-9", directory_name, "frozen-ablation", "single-seed"],
            config={
                "source_commit": git_commit(),
                "checkpoint_sha256": load_spec()["checkpoint"]["sha256"],
                "artifact_sha256": load_spec()["fixed_evaluation"]["sha256"],
            },
        )
        summary["wandb"] = {"run_id": run.id, "run_url": run.url}
    atomic_write_json(output / "summary.json", summary)
    rows = summary["conditions"]
    with (output / "conditions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "condition", "kind", "split", "nll", "delta_nll", "ci95_low", "ci95_high"
        ])
        for row in rows:
            for split, metrics in row["splits"].items():
                writer.writerow([
                    row["condition"]["id"], row["condition"]["kind"], split,
                    metrics["nll"], metrics["aggregate_delta_nll"],
                    metrics["ci95_low"], metrics["ci95_high"],
                ])
    if run:
        table = __import__("wandb").Table(
            columns=["condition", "kind", "split", "nll", "delta_nll", "ci95_low", "ci95_high"]
        )
        for row in rows:
            for split, metrics in row["splits"].items():
                table.add_data(
                    row["condition"]["id"], row["condition"]["kind"], split,
                    metrics["nll"], metrics["aggregate_delta_nll"],
                    metrics["ci95_low"], metrics["ci95_high"],
                )
        run.log({f"{directory_name}/condition_table": table})
        if directory_name == "phase-9a":
            for name, row in summary["structured"].items():
                for split, metrics in row["splits"].items():
                    run.log({
                        f"{split}/{name}/delta_nll": metrics["aggregate_delta_nll"],
                        f"{split}/{name}/ci95_low": metrics["ci95_low"],
                        f"{split}/{name}/ci95_high": metrics["ci95_high"],
                    })
            for split, metrics in summary["local_minus_global_damage"].items():
                run.log({
                    f"{split}/local_minus_global_damage/delta_nll": metrics["aggregate_delta_nll"],
                    f"{split}/local_minus_global_damage/ci95_low": metrics["ci95_low"],
                    f"{split}/local_minus_global_damage/ci95_high": metrics["ci95_high"],
                })
            run.summary["local_contribution_gate"] = summary["local_contribution_gate"]["passed"]
        else:
            for split, metrics in summary["aggregate"].items():
                run.log({
                    f"{split}/derangement_mean_delta_nll": metrics["mean_delta_nll"],
                    f"{split}/derangement_ci95_low": metrics["ci95_low"],
                    f"{split}/derangement_ci95_high": metrics["ci95_high"],
                    f"{split}/positive_fraction": metrics["positive_fraction"],
                })
            run.summary["alignment_evidence"] = summary["alignment_evidence"]["passed"]
        run.finish()
        atomic_write_json(output / "summary.json", summary)
    return summary


def signed(value: float) -> str:
    return f"{value:+.6f}"


def finalize_command(args) -> None:
    phase_9a = read_json(Path(args.phase_9a_summary))
    phase_9b = read_json(Path(args.phase_9b_summary)) if args.phase_9b_summary else None
    gate_passed = phase_9a["local_contribution_gate"]["passed"]
    if gate_passed != (phase_9b is not None):
        raise ValueError("final report does not match the frozen 9A gate")
    local = phase_9a["structured"]["zero-local"]
    global_ = phase_9a["structured"]["zero-global"]
    lines = [
        "# Experiment 9 — HQ8 Local-vs-Global Head Contribution",
        "",
        "## Frozen result",
        "",
        f"Experiment 9A local-contribution gate: `{'pass' if gate_passed else 'fail'}`.",
        "The unchanged condition exactly reproduced the accepted Experiment 8 HQ8 result.",
        "Positive delta NLL means the removed heads helped; negative means the ablation improved NLL.",
        "",
        "## Structured contribution ablations",
        "",
        "| Ablation | Split | Delta NLL | Paired 95% CI | Balanced-mask percentile |",
        "|---|---|---:|---:|---:|",
    ]
    for label, row in (("zero local", local), ("zero global", global_)):
        for split in ("discovery", "confirmation"):
            metric = row["splits"][split]
            lines.append(
                f"| {label} | {split} | {signed(metric['aggregate_delta_nll'])} | "
                f"[{signed(metric['ci95_low'])}, {signed(metric['ci95_high'])}] | "
                f"{metric['balanced_empirical_percentile']:.1f}% |"
            )
    distribution = phase_9a["balanced_random_8_distribution"]["confirmation"]
    population = phase_9a["local_minus_global_damage"]
    lines.extend([
        "",
        "The 70 exhaustive balanced masks remove four local and four global heads, "
        "with exactly one removed head per GQA group.",
        f"Their confirmation delta-NLL median is {signed(distribution['median'])} "
        f"(IQR [{signed(distribution['q25'])}, {signed(distribution['q75'])}]).",
        "",
        "## Local versus global population",
        "",
        "Positive values mean removing local heads hurts more than removing global heads.",
        "",
        "| Split | Local-damage minus global-damage | Paired 95% CI |",
        "|---|---:|---:|",
        *[
            f"| {split} | {signed(population[split]['aggregate_delta_nll'])} | "
            f"[{signed(population[split]['ci95_low'])}, {signed(population[split]['ci95_high'])}] |"
            for split in ("discovery", "confirmation")
        ],
    ])
    if phase_9b is None:
        lines.extend([
            "",
            "## Alignment gate",
            "",
            "Experiment 9B was not run because useful local-head contribution was "
            "not established by the preregistered Experiment 9A rule.",
        ])
    else:
        confirmation = phase_9b["aggregate"]["confirmation"]
        discovery = phase_9b["aggregate"]["discovery"]
        lines.extend([
            "",
            "## Local-chunk alignment",
            "",
            f"Alignment evidence: `{'pass' if phase_9b['alignment_evidence']['passed'] else 'fail'}`.",
            "",
            "| Split | Mean derangement delta NLL | Two-stage 95% CI | Positive fraction |",
            "|---|---:|---:|---:|",
            f"| discovery | {signed(discovery['mean_delta_nll'])} | "
            f"[{signed(discovery['ci95_low'])}, {signed(discovery['ci95_high'])}] | "
            f"{discovery['positive_fraction']:.1%} |",
            f"| confirmation | {signed(confirmation['mean_delta_nll'])} | "
            f"[{signed(confirmation['ci95_low'])}, {signed(confirmation['ci95_high'])}] | "
            f"{confirmation['positive_fraction']:.1%} |",
        ])
    links = []
    for name, summary in (("9A analysis", phase_9a), ("9B analysis", phase_9b)):
        if summary and summary.get("wandb"):
            links.append(f"- {name}: {summary['wandb']['run_url']}")
    lines.extend([
        "",
        "## Figures",
        "",
        "- [Structured contribution ablations against 70 matched masks](fig_head_contribution.pdf)",
        *(
            ["- [Frozen local-chunk derangements](fig_local_chunk_alignment.pdf)"]
            if phase_9b is not None else []
        ),
        "",
        "## W&B",
        "",
        *(links or ["- W&B disabled for this execution."]),
        "",
        "## Interpretation limits",
        "",
        "- This is one trained seed at one checkpoint.",
        "- Head zeroing is an off-distribution intervention, not retraining without those heads.",
        "- Confidence intervals quantify held-out sequence sampling; they do not quantify seed uncertainty.",
        "- Experiment 9B uses 32 frozen sampled derangements, not all 14,833 derangements.",
        "- This result does not authorize additional training or architecture experiments.",
        "",
        "Figures are generated reproducibly by `figures/gen_fig_experiment9_head_contribution.py`.",
        "",
    ])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines)
    (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    atomic_write_json(output / "final_summary.json", {
        "format_version": 1,
        "created_at": utc_now(),
        "experiment": "9-hq8-head-contribution",
        "phase_9a_gate_passed": gate_passed,
        "phase_9b_ran": phase_9b is not None,
        "alignment_evidence": (
            phase_9b["alignment_evidence"]["passed"] if phase_9b else None
        ),
        "phase_9a_summary_sha256": sha256_path(Path(args.phase_9a_summary)),
        "phase_9b_summary_sha256": (
            sha256_path(Path(args.phase_9b_summary)) if phase_9b else None
        ),
    })


def manifest_command(args) -> None:
    target = Path(args.output)
    payload = condition_manifest()
    if target.exists() and read_json(target) != payload:
        raise ValueError("refusing to overwrite a different condition manifest")
    atomic_write_json(target, payload)


def add_common_analysis_arguments(parser) -> None:
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb-project", default="MHAR Stuff")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="mhar-exp9-hq8-head-contribution-seed42-step2000")
    parser.add_argument("--wandb-run-name", default=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(func=manifest_command)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--phase", required=True, choices=("phase-9a", "phase-9b"))
    evaluate.add_argument("--worker-index", type=int, required=True)
    evaluate.add_argument("--worker-count", type=int, default=2)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--artifact", required=True)
    evaluate.add_argument("--output-root", required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.set_defaults(func=evaluate_command)
    analysis_9a = sub.add_parser("analyze-9a")
    add_common_analysis_arguments(analysis_9a)
    analysis_9a.set_defaults(func=analyze_9a)
    analysis_9b = sub.add_parser("analyze-9b")
    add_common_analysis_arguments(analysis_9b)
    analysis_9b.add_argument("--phase-9a-summary", required=True)
    analysis_9b.set_defaults(func=analyze_9b)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--phase-9a-summary", required=True)
    finalize.add_argument("--phase-9b-summary")
    finalize.add_argument("--output-dir", required=True)
    finalize.set_defaults(func=finalize_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
