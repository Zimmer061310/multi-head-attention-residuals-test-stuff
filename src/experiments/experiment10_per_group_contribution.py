#!/usr/bin/env python3
"""Frozen per-group HQ8 query-head contribution and alignment experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
from src.experiments.experiment8_hybrid_q import Experiment8MHARForCausalLM
from src.experiments.experiment9_head_contribution import (
    ACCEPTED_HQ8_RESULT,
    bootstrap_delta,
    hierarchical_bootstrap,
    mask_head_outputs,
    validate_checkpoint,
    validate_hq8_model,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiment10/protocol.json"
EXP9_ZERO_LOCAL = (
    ROOT / "results/experiment9/phase-9a/conditions/zero-local/result.json"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_spec() -> dict:
    spec = read_json(CONFIG)
    if spec["training"] is not False or spec["seed"] != 42 or spec["milestone"] != 2000:
        raise ValueError("Experiment 10 must remain a frozen seed-42 step-2000 evaluation")
    counts = spec["phase_10abc"]["conditions"]
    if counts != {"reference": 1, "single_local": 8, "single_global": 8, "whole_group": 8}:
        raise ValueError("Experiment 10ABC must retain the frozen 25-condition design")
    if spec["phase_10d"]["condition_count"] != 56:
        raise ValueError("Experiment 10D must exhaust exactly 56 one-group substitutions")
    return spec


def phase_10abc_conditions() -> list[dict]:
    conditions = [{"id": "hq8-unchanged", "kind": "reference", "removed_heads": []}]
    for group in range(8):
        conditions.extend([
            {
                "id": f"remove-local-g{group}",
                "kind": "single-local",
                "group": group,
                "removed_heads": [2 * group],
            },
            {
                "id": f"remove-global-g{group}",
                "kind": "single-global",
                "group": group,
                "removed_heads": [2 * group + 1],
            },
            {
                "id": f"remove-group-g{group}",
                "kind": "whole-group",
                "group": group,
                "removed_heads": [2 * group, 2 * group + 1],
            },
        ])
    if len(conditions) != 25 or len({row["id"] for row in conditions}) != 25:
        raise AssertionError("Experiment 10ABC condition construction is not unique")
    return conditions


def phase_10d_conditions() -> list[dict]:
    conditions = []
    identity = list(range(8))
    for target_group in range(8):
        for source_chunk in range(8):
            if source_chunk == target_group:
                continue
            sources = identity.copy()
            sources[target_group] = source_chunk
            conditions.append({
                "id": f"misalign-g{target_group}-from-c{source_chunk}",
                "kind": "single-local-misalignment",
                "target_group": target_group,
                "source_chunk": source_chunk,
                "local_chunk_sources": sources,
            })
    if len(conditions) != 56 or len({row["id"] for row in conditions}) != 56:
        raise AssertionError("Experiment 10D condition construction is not exhaustive")
    return conditions


def condition_manifest() -> dict:
    spec = load_spec()
    payload = {
        "format_version": 1,
        "experiment": "10-per-group-query-contribution",
        "created_from_commit": git_commit(),
        "checkpoint_sha256": spec["checkpoint"]["sha256"],
        "artifact_sha256": spec["fixed_evaluation"]["sha256"],
        "phase_10abc": phase_10abc_conditions(),
        "phase_10d": phase_10d_conditions(),
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    return payload


class Experiment10Intervention(AbstractContextManager):
    """Apply a frozen head mask or one-group local-chunk source substitution."""

    def __init__(self, model, *, removed_heads=(), local_chunk_sources=None):
        self.model = model
        self.removed_heads = tuple(sorted(int(head) for head in removed_heads))
        self.local_chunk_sources = (
            None if local_chunk_sources is None
            else tuple(int(group) for group in local_chunk_sources)
        )
        self.handles = []
        if len(set(self.removed_heads)) != len(self.removed_heads):
            raise ValueError("removed head indices must be unique")
        if any(head < 0 or head >= 16 for head in self.removed_heads):
            raise ValueError("removed head index outside [0, 15]")
        if self.local_chunk_sources is not None:
            if len(self.local_chunk_sources) != 8:
                raise ValueError("local chunk source map must contain eight entries")
            if any(source < 0 or source >= 8 for source in self.local_chunk_sources):
                raise ValueError("local chunk source index outside [0, 7]")
            changed = [
                target for target, source in enumerate(self.local_chunk_sources)
                if source != target
            ]
            if len(changed) != 1:
                raise ValueError("Experiment 10D must change exactly one local group")

    def __enter__(self):
        validate_hq8_model(self.model)
        for layer in self.model.model.layers:
            query = layer.self_attn.q_proj
            if query.local_group_permutation is not None:
                raise RuntimeError("a local chunk intervention is already active")
            query.local_group_permutation = self.local_chunk_sources
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


def result_path(results_root: Path, phase: str, condition_id: str) -> Path:
    return results_root / phase / "conditions" / condition_id / "result.json"


def phase_conditions(phase: str) -> list[dict]:
    if phase == "phase-10abc":
        return phase_10abc_conditions()
    if phase == "phase-10d":
        return phase_10d_conditions()
    raise ValueError(f"unknown phase {phase}")


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
        with Experiment10Intervention(
            model,
            removed_heads=condition.get("removed_heads", ()),
            local_chunk_sources=condition.get("local_chunk_sources"),
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
        atomic_write_json(target, {
            "format_version": 1,
            "created_at": utc_now(),
            "experiment": "10-per-group-query-contribution",
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
        })
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


def condition_metrics(result: dict, reference: dict, split: str, *, samples: int, seed: int) -> dict:
    candidate = np.asarray(result["splits"][split]["sequence_nlls"], dtype=float)
    baseline = np.asarray(reference["splits"][split]["sequence_nlls"], dtype=float)
    return {
        "nll": float(result["splits"][split]["nll"]),
        "aggregate_delta_nll": float(
            result["splits"][split]["nll"] - reference["splits"][split]["nll"]
        ),
        **bootstrap_delta(candidate, baseline, samples=samples, seed=seed),
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError("Spearman inputs must be equal one-dimensional vectors")
    return float(np.corrcoef(average_ranks(left), average_ranks(right))[0, 1])


def bootstrap_linear_combination(arrays: list[np.ndarray], coefficients: list[float], *, samples: int, seed: int) -> dict:
    if len(arrays) != len(coefficients) or not arrays:
        raise ValueError("arrays and coefficients must be non-empty and equal length")
    shape = arrays[0].shape
    if len(shape) != 1 or any(array.shape != shape for array in arrays):
        raise ValueError("all arrays must be equal one-dimensional paired vectors")
    values = sum(coefficient * array for array, coefficient in zip(arrays, coefficients))
    return bootstrap_delta(values, np.zeros_like(values), samples=samples, seed=seed)


def analyze_10abc(args) -> dict:
    spec = load_spec()
    conditions = phase_10abc_conditions()
    results = load_phase_results(Path(args.results_root), "phase-10abc", conditions)
    reference = results["hq8-unchanged"]
    reproduction = reference_gate(reference, spec)
    if not reproduction["passed"]:
        raise ValueError("unchanged HQ8 did not exactly reproduce Experiment 8")
    samples = spec["phase_10abc"]["bootstrap_samples"]
    base_seed = spec["phase_10abc"]["bootstrap_seed"]
    rows = []
    for condition_index, condition in enumerate(conditions):
        result = results[condition["id"]]
        row = {"condition": condition, "splits": {}}
        for split_index, split in enumerate(spec["fixed_evaluation"]["splits"]):
            row["splits"][split] = condition_metrics(
                result, reference, split,
                samples=samples,
                seed=base_seed + condition_index * 2 + split_index,
            )
        rows.append(row)
    by_id = {row["condition"]["id"]: row for row in rows}
    vectors = {}
    correlations = {}
    interactions = []
    for split_index, split in enumerate(spec["fixed_evaluation"]["splits"]):
        local = np.asarray([
            by_id[f"remove-local-g{group}"]["splits"][split]["aggregate_delta_nll"]
            for group in range(8)
        ])
        global_ = np.asarray([
            by_id[f"remove-global-g{group}"]["splits"][split]["aggregate_delta_nll"]
            for group in range(8)
        ])
        whole = np.asarray([
            by_id[f"remove-group-g{group}"]["splits"][split]["aggregate_delta_nll"]
            for group in range(8)
        ])
        vectors[split] = {
            "D_L": local.tolist(),
            "D_G": global_.tolist(),
            "D_GL": whole.tolist(),
            "interaction": (whole - local - global_).tolist(),
        }
        correlations[split] = {
            "spearman_D_L_vs_D_G": spearman(local, global_),
            "note": "descriptive across eight trained groups; no seed-level inference",
        }
        for group in range(8):
            ref_values = np.asarray(reference["splits"][split]["sequence_nlls"], dtype=float)
            local_values = np.asarray(results[f"remove-local-g{group}"]["splits"][split]["sequence_nlls"], dtype=float)
            global_values = np.asarray(results[f"remove-global-g{group}"]["splits"][split]["sequence_nlls"], dtype=float)
            whole_values = np.asarray(results[f"remove-group-g{group}"]["splits"][split]["sequence_nlls"], dtype=float)
            interaction = bootstrap_linear_combination(
                [whole_values, local_values, global_values, ref_values],
                [1.0, -1.0, -1.0, 1.0],
                samples=samples,
                seed=base_seed + 1000 + group * 2 + split_index,
            )
            interactions.append({"group": group, "split": split, **interaction})

    useful_rule = spec["phase_10abc"]["useful_local_group_rule"]
    useful = []
    useful_checks = {}
    for group in range(8):
        discovery = by_id[f"remove-local-g{group}"]["splits"]["discovery"]
        confirmation = by_id[f"remove-local-g{group}"]["splits"]["confirmation"]
        checks = {
            "discovery_delta_positive": discovery["aggregate_delta_nll"] > 0,
            "confirmation_delta_at_least_margin": confirmation["aggregate_delta_nll"] >= useful_rule["practical_margin_nll"],
            "confirmation_ci_lower_above_zero": confirmation["ci95_low"] > 0,
        }
        useful_checks[str(group)] = checks
        if all(checks.values()):
            useful.append(group)
    confirmation_local = np.asarray(vectors["confirmation"]["D_L"])
    positive = np.clip(confirmation_local, 0, None)
    top2_share = float(np.sort(positive)[-2:].sum() / positive.sum()) if positive.sum() else 0.0
    classification_spec = spec["phase_10abc"]["distribution_classification"]
    if (
        len(useful) >= classification_spec["distributed_minimum_useful_groups"]
        and top2_share <= classification_spec["distributed_maximum_top2_positive_share"]
    ):
        classification = "distributed"
    elif (
        len(useful) <= classification_spec["concentrated_maximum_useful_groups"]
        or top2_share >= classification_spec["concentrated_minimum_top2_positive_share"]
    ):
        classification = "concentrated"
    else:
        classification = "mixed_or_inconclusive"

    exp9 = read_json(EXP9_ZERO_LOCAL)
    if (
        exp9.get("checkpoint_sha256") != spec["checkpoint"]["sha256"]
        or exp9.get("artifact_sha256") != spec["fixed_evaluation"]["sha256"]
    ):
        raise ValueError("accepted Experiment 9 zero-local result identity mismatch")
    collective = {}
    for split_index, split in enumerate(spec["fixed_evaluation"]["splits"]):
        ref_values = np.asarray(reference["splits"][split]["sequence_nlls"], dtype=float)
        all_local = np.asarray(exp9["splits"][split]["sequence_nlls"], dtype=float)
        arrays = [all_local] + [
            np.asarray(results[f"remove-local-g{group}"]["splits"][split]["sequence_nlls"], dtype=float)
            for group in range(8)
        ] + [ref_values]
        coefficients = [1.0] + [-1.0] * 8 + [7.0]
        collective[split] = {
            "all_local_population_delta_nll": float(exp9["splits"][split]["nll"] - reference["splits"][split]["nll"]),
            "sum_single_local_delta_nll": float(sum(vectors[split]["D_L"])),
            **bootstrap_linear_combination(
                arrays, coefficients,
                samples=samples,
                seed=base_seed + 2000 + split_index,
            ),
            "interpretation": "positive means all-local damage exceeds the sum of single-local damages",
        }

    gate = len(useful) >= spec["phase_10abc"]["phase_10d_gate"]["minimum_useful_local_groups"]
    summary = {
        "format_version": 1,
        "created_at": utc_now(),
        "experiment": "10-per-group-query-contribution",
        "phase": "10ABC-contribution",
        "reference_reproduction": reproduction,
        "conditions": rows,
        "vectors": vectors,
        "interactions": interactions,
        "local_global_rank_correlation": correlations,
        "local_distribution": {
            "classification": classification,
            "useful_groups": useful,
            "useful_group_count": len(useful),
            "useful_group_checks": useful_checks,
            "positive_top2_share": top2_share,
            "rule": useful_rule["rule"],
        },
        "collective_dependence": collective,
        "phase_10d_gate": {
            "passed": gate,
            "rule": spec["phase_10abc"]["phase_10d_gate"]["rule"],
        },
        "interpretation_limits": spec["interpretation_limits"],
    }
    return write_analysis(args, summary, "phase-10abc")


def analyze_10d(args) -> dict:
    spec = load_spec()
    phase_10abc = read_json(Path(args.phase_10abc_summary))
    if not phase_10abc["phase_10d_gate"]["passed"]:
        raise ValueError("Experiment 10D is not authorized by the frozen 10ABC gate")
    conditions = phase_10d_conditions()
    results = load_phase_results(Path(args.results_root), "phase-10d", conditions)
    reference = load_phase_results(
        Path(args.results_root), "phase-10abc", [phase_10abc_conditions()[0]]
    )["hq8-unchanged"]
    samples = spec["phase_10d"]["bootstrap_samples"]
    base_seed = spec["phase_10d"]["bootstrap_seed"]
    rows = []
    for condition_index, condition in enumerate(conditions):
        result = results[condition["id"]]
        row = {"condition": condition, "splits": {}}
        for split_index, split in enumerate(spec["fixed_evaluation"]["splits"]):
            row["splits"][split] = condition_metrics(
                result, reference, split,
                samples=samples,
                seed=base_seed + condition_index * 2 + split_index,
            )
        rows.append(row)
    by_id = {row["condition"]["id"]: row for row in rows}
    group_alignment = {}
    for group in range(8):
        group_alignment[str(group)] = {}
        group_conditions = [row for row in conditions if row["target_group"] == group]
        for split_index, split in enumerate(spec["fixed_evaluation"]["splits"]):
            baseline = np.asarray(reference["splits"][split]["sequence_nlls"], dtype=float)
            values = np.stack([
                np.asarray(results[row["id"]]["splits"][split]["sequence_nlls"], dtype=float) - baseline
                for row in group_conditions
            ])
            means = values.mean(axis=1)
            group_alignment[str(group)][split] = {
                **hierarchical_bootstrap(
                    values,
                    samples=samples,
                    seed=base_seed + 1000 + group * 2 + split_index,
                ),
                "positive_fraction": float(np.mean(means > 0)),
                "minimum_substitution_delta_nll": float(means.min()),
                "maximum_substitution_delta_nll": float(means.max()),
            }
    summary = {
        "format_version": 1,
        "created_at": utc_now(),
        "experiment": "10-per-group-query-contribution",
        "phase": "10D-one-group-alignment",
        "conditions": rows,
        "group_alignment": group_alignment,
        "interpretation_limits": spec["interpretation_limits"],
    }
    return write_analysis(args, summary, "phase-10d")


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
            job_type=f"experiment10-{directory_name}-analysis",
            tags=["experiment-10", directory_name, "frozen-ablation", "single-seed"],
            config={
                "source_commit": git_commit(),
                "checkpoint_sha256": load_spec()["checkpoint"]["sha256"],
                "artifact_sha256": load_spec()["fixed_evaluation"]["sha256"],
            },
        )
        summary["wandb"] = {"run_id": run.id, "run_url": run.url}
    atomic_write_json(output / "summary.json", summary)
    with (output / "conditions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "kind", "split", "nll", "delta_nll", "ci95_low", "ci95_high"])
        for row in summary["conditions"]:
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
        for row in summary["conditions"]:
            for split, metrics in row["splits"].items():
                table.add_data(
                    row["condition"]["id"], row["condition"]["kind"], split,
                    metrics["nll"], metrics["aggregate_delta_nll"],
                    metrics["ci95_low"], metrics["ci95_high"],
                )
        run.log({f"{directory_name}/condition_table": table})
        if directory_name == "phase-10abc":
            for split, vectors in summary["vectors"].items():
                for name, values in vectors.items():
                    for group, value in enumerate(values):
                        run.log({f"{split}/{name}/group_{group}": value})
            run.summary["local_distribution"] = summary["local_distribution"]["classification"]
            run.summary["phase_10d_gate"] = summary["phase_10d_gate"]["passed"]
        else:
            for group, splits in summary["group_alignment"].items():
                for split, metrics in splits.items():
                    run.log({
                        f"{split}/alignment/group_{group}/mean_delta_nll": metrics["mean_delta_nll"],
                        f"{split}/alignment/group_{group}/ci95_low": metrics["ci95_low"],
                        f"{split}/alignment/group_{group}/ci95_high": metrics["ci95_high"],
                    })
        run.finish()
        atomic_write_json(output / "summary.json", summary)
    return summary


def signed(value: float) -> str:
    return f"{value:+.6f}"


def finalize_command(args) -> None:
    primary = read_json(Path(args.phase_10abc_summary))
    alignment = read_json(Path(args.phase_10d_summary)) if args.phase_10d_summary else None
    if primary["phase_10d_gate"]["passed"] != (alignment is not None):
        raise ValueError("final report does not match the frozen 10D gate")
    lines = [
        "# Experiment 10 — Per-Group Local/Global Query Contribution",
        "",
        "## Frozen result",
        "",
        "The unchanged condition exactly reproduced the accepted Experiment 8 HQ8 result.",
        "Positive delta NLL means the removed head or group helped the trained model.",
        f"Local contribution classification: `{primary['local_distribution']['classification']}`.",
        f"Useful local groups: {primary['local_distribution']['useful_groups']}.",
        "",
        "## Per-group contribution map",
        "",
        "| Group | Split | D_L | D_G | D_GL | Interaction |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    interaction_map = {
        (row["group"], row["split"]): row for row in primary["interactions"]
    }
    for group in range(8):
        for split in ("discovery", "confirmation"):
            vector = primary["vectors"][split]
            lines.append(
                f"| {group} | {split} | {signed(vector['D_L'][group])} | "
                f"{signed(vector['D_G'][group])} | {signed(vector['D_GL'][group])} | "
                f"{signed(interaction_map[(group, split)]['mean_delta_nll'])} |"
            )
    lines.extend([
        "",
        "## Distribution diagnostics",
        "",
        f"Confirmation positive top-two local-damage share: "
        f"{primary['local_distribution']['positive_top2_share']:.1%}.",
        f"Discovery D_L/D_G Spearman: "
        f"{primary['local_global_rank_correlation']['discovery']['spearman_D_L_vs_D_G']:+.3f}.",
        f"Confirmation D_L/D_G Spearman: "
        f"{primary['local_global_rank_correlation']['confirmation']['spearman_D_L_vs_D_G']:+.3f}.",
        "",
        "## Collective local-head diagnostic",
        "",
        "| Split | All-local damage | Sum of single-local damages | Collective gap | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ])
    for split in ("discovery", "confirmation"):
        metric = primary["collective_dependence"][split]
        lines.append(
            f"| {split} | {signed(metric['all_local_population_delta_nll'])} | "
            f"{signed(metric['sum_single_local_delta_nll'])} | "
            f"{signed(metric['mean_delta_nll'])} | "
            f"[{signed(metric['ci95_low'])}, {signed(metric['ci95_high'])}] |"
        )
    if alignment is None:
        lines.extend([
            "",
            "## One-group alignment",
            "",
            "Experiment 10D was not run because no local group passed the frozen usefulness rule.",
        ])
    else:
        lines.extend([
            "",
            "## One-group alignment",
            "",
            "| Target group | Split | Mean wrong-chunk delta NLL | Two-stage 95% CI | Positive fraction |",
            "|---:|---|---:|---:|---:|",
        ])
        for group in range(8):
            for split in ("discovery", "confirmation"):
                metric = alignment["group_alignment"][str(group)][split]
                lines.append(
                    f"| {group} | {split} | {signed(metric['mean_delta_nll'])} | "
                    f"[{signed(metric['ci95_low'])}, {signed(metric['ci95_high'])}] | "
                    f"{metric['positive_fraction']:.1%} |"
                )
    lines.extend([
        "",
        "## Figures",
        "",
        "- [Per-group local/global/whole-group contribution](fig_group_contribution.pdf)",
        "- [Local-head distribution and interaction](fig_local_distribution.pdf)",
    ])
    if alignment is not None:
        lines.append("- [One-group local-chunk alignment map](fig_group_alignment.pdf)")
    links = []
    for label, summary in (("10ABC analysis", primary), ("10D analysis", alignment)):
        if summary and summary.get("wandb"):
            links.append(f"- {label}: {summary['wandb']['run_url']}")
    lines.extend(["", "## W&B", "", *(links or ["- W&B disabled for this analysis run."])])
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend(f"- {limit}" for limit in primary["interpretation_limits"])
    lines.extend([
        "",
        "Figures are generated reproducibly by `figures/gen_fig_experiment10_group_contribution.py`.",
        "",
    ])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "format_version": 1,
        "created_at": utc_now(),
        "experiment": "10-per-group-query-contribution",
        "classification": primary["local_distribution"]["classification"],
        "useful_local_groups": primary["local_distribution"]["useful_groups"],
        "phase_10d_ran": alignment is not None,
        "phase_10abc_summary_sha256": sha256_path(Path(args.phase_10abc_summary)),
        "phase_10d_summary_sha256": sha256_path(Path(args.phase_10d_summary)) if alignment else None,
    }
    atomic_write_json(output / "final_summary.json", summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--phase", choices=("phase-10abc", "phase-10d"), required=True)
    evaluate.add_argument("--worker-index", type=int, required=True)
    evaluate.add_argument("--worker-count", type=int, required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--artifact", required=True)
    evaluate.add_argument("--output-root", required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--dtype", default="bf16")
    evaluate.add_argument("--batch-size", type=int, default=1)
    for name in ("analyze-10abc", "analyze-10d"):
        analyze = subparsers.add_parser(name)
        analyze.add_argument("--results-root", required=True)
        analyze.add_argument("--output-dir", required=True)
        analyze.add_argument("--phase-10abc-summary")
        analyze.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
        analyze.add_argument("--wandb-project", default="MHAR Stuff")
        analyze.add_argument("--wandb-entity")
        analyze.add_argument("--wandb-group", default="mhar-exp10-per-group-contribution-seed42-step2000")
        analyze.add_argument("--wandb-run-name", default=name)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--phase-10abc-summary", required=True)
    finalize.add_argument("--phase-10d-summary")
    finalize.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "analyze-10abc":
        analyze_10abc(args)
    elif args.command == "analyze-10d":
        analyze_10d(args)
    elif args.command == "finalize":
        finalize_command(args)


if __name__ == "__main__":
    main()
