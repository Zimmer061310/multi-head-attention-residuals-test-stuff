#!/usr/bin/env python3
"""Fail-closed measurement, selection, and analysis for Experiment 11."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

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
from src.experiments.experiment11_soft_specialization import (
    Experiment11MHARForCausalLM,
    RUN_BY_ID,
    RUNS,
    SoftSpecializedQueryLinear,
    experiment11_parameter_report,
    run_spec,
    weight_specialization_metrics,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiment11/protocol.json"
SPLITS = ("discovery", "confirmation")
EPSILON = 1e-12
RUN_MATRIX_SHA256 = "10f6d106d577b918f98c70fd0b3ff838ead9042c7c03978ce61f5ab96f7eb654"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash(payload: dict) -> str:
    value = {key: item for key, item in payload.items() if key != "content_sha256"}
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def immutable_json(path: Path, payload: dict) -> None:
    """Write once, or accept an exactly matching canonical payload."""

    payload = dict(payload)
    payload["content_sha256"] = canonical_hash(payload)
    if path.exists():
        existing = read_json(path)
        if existing != payload:
            raise FileExistsError(f"refusing to overwrite mismatched result: {path}")
        return
    atomic_write_json(path, payload)


def load_spec() -> dict:
    spec = read_json(CONFIG)
    rows = spec.get("runs", [])
    expected = [
        {"id": row.run_id, "family": row.family, "lambda": row.lambda_value}
        for row in RUNS
    ]
    if rows != expected:
        raise ValueError("Experiment 11 protocol does not match the frozen nine-run matrix")
    observed_matrix_hash = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if observed_matrix_hash != RUN_MATRIX_SHA256 or spec.get("run_matrix_sha256") != RUN_MATRIX_SHA256:
        raise ValueError("Experiment 11 run-matrix hash changed")
    if spec.get("seed") != 42 or spec.get("screening_milestone") != 2000:
        raise ValueError("Experiment 11 must remain a seed-42 step-2000 screen")
    if spec.get("probe_milestones") != [0, 500, 1000, 1500, 2000]:
        raise ValueError("Experiment 11 probe milestones changed")
    if spec["fixed_evaluation"].get("discovery_probe_indices") != list(range(32)):
        raise ValueError("Experiment 11 discovery probe indices changed")
    return spec


def protocol_sha256() -> str:
    return hashlib.sha256(CONFIG.read_bytes()).hexdigest()


def checkpoint_files_complete(checkpoint: Path) -> bool:
    return (
        (checkpoint / "training_manifest.json").is_file()
        and (checkpoint / "training_state.pt").is_file()
        and bool(
            list(checkpoint.glob("model*.safetensors"))
            or (checkpoint / "pytorch_model.bin").is_file()
        )
    )


def validate_checkpoint(checkpoint: Path, run_id: str, milestone: int) -> dict:
    spec = load_spec()
    if milestone not in spec["probe_milestones"][1:]:
        raise ValueError("checkpoint milestone is not preregistered")
    if not checkpoint_files_complete(checkpoint):
        raise FileNotFoundError(f"incomplete checkpoint: {checkpoint}")
    manifest = read_json(checkpoint / "training_manifest.json")
    identity = manifest.get("run_identity", {})
    row = run_spec(run_id)
    expected = {
        "global_step": milestone,
        "chunks_consumed": milestone * spec["backbone"]["global_batch_size"],
        "mode": "full_mh",
        "attnres_heads": 8,
        "hidden_size": 1280,
        "num_layers": 36,
        "num_heads": 16,
        "num_kv_heads": 8,
        "intermediate_size": 5120,
        "seq_len": 1024,
        "steps": spec["full_schedule_steps"],
        "global_batch_size": 32,
        "lr": 5e-4,
        "lr_min": 5e-5,
        "warmup": 1000,
        "seed": spec["seed"],
        "experiment11_run_id": row.run_id,
        "experiment11_family": row.family,
        "experiment11_lambda": row.lambda_value,
        "experiment11_soft_q_groups": 8,
    }
    observed = {
        "global_step": manifest.get("global_step"),
        "chunks_consumed": manifest.get("chunks_consumed"),
        **{key: identity.get(key) for key in expected if key not in {
            "global_step", "chunks_consumed"}},
    }
    mismatch = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items() if observed.get(key) != value
    }
    if mismatch:
        raise ValueError(f"checkpoint identity mismatch: {json.dumps(mismatch)}")
    return observed


def validate_model(model, run_id: str) -> dict:
    row = run_spec(run_id)
    expected = {
        "hidden_size": 1280,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 5120,
        "attnres_mode": "full_mh",
        "attnres_num_heads": 8,
        "experiment11_run_id": row.run_id,
        "experiment11_family": row.family,
        "experiment11_lambda": row.lambda_value,
        "experiment11_soft_q_groups": 8,
    }
    observed = {key: getattr(model.config, key, None) for key in expected}
    mismatch = {
        key: {"expected": value, "observed": observed[key]}
        for key, value in expected.items() if observed[key] != value
    }
    if mismatch:
        raise ValueError(f"model architecture mismatch: {json.dumps(mismatch)}")
    if not all(isinstance(layer.self_attn.q_proj, SoftSpecializedQueryLinear)
               for layer in model.model.layers):
        raise TypeError("every query projection must use the Experiment 11 wrapper")
    return observed


def verify_artifact(path: Path) -> tuple[dict, str]:
    spec = load_spec()
    payload, digest = load_fixed_eval_artifact(path)
    if digest != spec["fixed_evaluation"]["sha256"]:
        raise ValueError("fixed evaluation artifact hash differs from preregistration")
    return payload, digest


def load_model(checkpoint: Path, run_id: str, *, device: torch.device, dtype: str):
    model = Experiment11MHARForCausalLM.from_pretrained(
        str(checkpoint), dtype=dtype_from_name(dtype)
    ).to(device=device).eval()
    validate_model(model, run_id)
    return model


def _means(array: np.ndarray) -> np.ndarray:
    return np.nanmean(array, axis=0)


@torch.inference_mode()
def probe_model(model, input_ids: torch.Tensor, *, device: torch.device) -> dict:
    """Measure effective softness with the model's actual per-head QNorm."""

    layers = len(model.model.layers)
    per_sequence = {name: [] for name in (
        "local_norm", "cross_norm", "r_act", "theta_radians", "component_cosine",
        "component_cosine_valid_fraction",
    )}
    current: dict[int, dict[str, np.ndarray]] = {}
    handles = []

    def hook_for(layer_index: int, query: SoftSpecializedQueryLinear, q_norm):
        def hook(module, args):
            inputs = args[0]
            local, cross = query.local_cross_components(inputs, fp32=True)
            local_norm = local.norm(dim=-1)
            cross_norm = cross.norm(dim=-1)
            full_q = q_norm((local + cross).float()).float()
            local_q = q_norm(local.float()).float()
            direction_cos = torch.nn.functional.cosine_similarity(
                full_q, local_q, dim=-1, eps=EPSILON
            ).clamp(-1.0, 1.0)
            theta = direction_cos.acos()
            component_cos = torch.nn.functional.cosine_similarity(
                local, cross, dim=-1, eps=EPSILON
            )
            valid = cross_norm > EPSILON
            component_cos = torch.where(
                valid, component_cos, torch.zeros_like(component_cos)
            )
            current[layer_index] = {
                "local_norm": local_norm.mean(dim=(0, 1)).cpu().numpy(),
                "cross_norm": cross_norm.mean(dim=(0, 1)).cpu().numpy(),
                "r_act": (
                    cross_norm.mean(dim=(0, 1))
                    / local_norm.mean(dim=(0, 1)).clamp_min(EPSILON)
                ).cpu().numpy(),
                "theta_radians": theta.mean(dim=(0, 1)).cpu().numpy(),
                "component_cosine": component_cos.mean(dim=(0, 1)).cpu().numpy(),
                "component_cosine_valid_fraction": valid.float().mean(
                    dim=(0, 1)
                ).cpu().numpy(),
            }
        return hook

    for index, layer in enumerate(model.model.layers):
        query = layer.self_attn.q_proj
        handles.append(query.register_forward_pre_hook(
            hook_for(index, query, layer.self_attn.q_norm)
        ))
    try:
        for sequence in input_ids:
            current.clear()
            batch = sequence.unsqueeze(0).to(device=device, dtype=torch.long)
            model.model(input_ids=batch, use_cache=False)
            if set(current) != set(range(layers)):
                raise RuntimeError("probe hooks did not observe every transformer layer")
            for name in per_sequence:
                per_sequence[name].append(np.stack([
                    current[index][name] for index in range(layers)
                ]))
    finally:
        for handle in handles:
            handle.remove()

    arrays = {key: np.stack(value) for key, value in per_sequence.items()}
    diagnostic_positions = model.model.layers[0].self_attn.q_proj.diagnostic_head_positions()
    selector = np.asarray(diagnostic_positions, dtype=int)
    aggregate = {}
    for name, value in arrays.items():
        selected = value[..., selector]
        aggregate[name] = {
            "mean": float(np.nanmean(selected)),
            "per_sequence": np.nanmean(selected, axis=(1, 2, 3)).tolist(),
            "per_sequence_head": np.nanmean(value, axis=(1, 2)).tolist(),
            "layer_group_head": _means(value).tolist(),
        }
    aggregate["r_act"]["mean"] = (
        aggregate["cross_norm"]["mean"]
        / max(aggregate["local_norm"]["mean"], EPSILON)
    )
    selected_cross = arrays["cross_norm"][..., selector]
    selected_local = arrays["local_norm"][..., selector]
    aggregate["r_act"]["per_sequence"] = (
        selected_cross.mean(axis=(1, 2, 3))
        / np.maximum(selected_local.mean(axis=(1, 2, 3)), EPSILON)
    ).tolist()
    aggregate["r_act"]["per_sequence_head"] = (
        arrays["cross_norm"].mean(axis=(1, 2))
        / np.maximum(arrays["local_norm"].mean(axis=(1, 2)), EPSILON)
    ).tolist()
    return {
        "epsilon": EPSILON,
        "sequences": int(input_ids.shape[0]),
        "tokens_per_sequence": int(input_ids.shape[1]),
        "diagnostic_head_positions": list(diagnostic_positions),
        "metrics": aggregate,
    }


def write_step0_probe(
    model,
    *,
    run_id: str,
    artifact: Path,
    output: Path,
    training_identity: dict,
    device: torch.device,
) -> dict:
    """Measure the exact initialized training model before optimizer step one."""

    spec = load_spec()
    payload, artifact_hash = verify_artifact(Path(artifact).resolve())
    input_ids = payload["discovery_input_ids"][
        spec["fixed_evaluation"]["discovery_probe_indices"]
    ]
    was_training = model.training
    model.eval()
    result = {
        "format_version": 1,
        "created_at": utc_now(),
        "experiment": "11-soft-query-specialization",
        "run_id": run_id,
        "family": run_spec(run_id).family,
        "lambda": run_spec(run_id).lambda_value,
        "milestone": 0,
        "split": "discovery",
        "checkpoint_sha256": None,
        "artifact_sha256": artifact_hash,
        "protocol_sha256": protocol_sha256(),
        "source_commit": git_commit(),
        "training_identity": training_identity,
        "parameters": experiment11_parameter_report(model),
        "weight_metrics": weight_specialization_metrics(model),
        "activation_metrics": probe_model(model, input_ids, device=device),
    }
    immutable_json(Path(output).resolve(), result)
    model.train(was_training)
    return result


def probe_command(args) -> None:
    spec = load_spec()
    if args.split not in SPLITS:
        raise ValueError("unknown fixed split")
    milestone = int(args.milestone)
    if milestone not in spec["probe_milestones"]:
        raise ValueError("probe milestone is not frozen")
    if milestone == 0:
        raise ValueError("step-0 is emitted by the training entrypoint, not checkpoint probe")
    if args.split == "confirmation":
        if milestone != 2000 or args.selection_manifest is None:
            raise ValueError("confirmation probes require step 2000 and a frozen selection")
        selection = validate_selection_manifest(Path(args.selection_manifest), args.results_root)
        allowed = {selection["selected"][family]["run_id"] for family in ("s2q8", "gslq8")}
        allowed.add("m8-l100")
        if args.run_id not in allowed:
            raise ValueError("confirmation activation probe is limited to frozen selections and M8")
    checkpoint = Path(args.checkpoint).resolve()
    artifact = Path(args.artifact).resolve()
    output = Path(args.output).resolve()
    identity = validate_checkpoint(checkpoint, args.run_id, milestone)
    payload, artifact_hash = verify_artifact(artifact)
    input_ids = payload[f"{args.split}_input_ids"]
    if args.split == "discovery":
        input_ids = input_ids[spec["fixed_evaluation"]["discovery_probe_indices"]]
    device = torch.device(args.device)
    model = load_model(checkpoint, args.run_id, device=device, dtype=args.dtype)
    payload = {
        "format_version": 1,
        "created_at": utc_now(),
        "experiment": "11-soft-query-specialization",
        "run_id": args.run_id,
        "family": run_spec(args.run_id).family,
        "lambda": run_spec(args.run_id).lambda_value,
        "milestone": milestone,
        "split": args.split,
        "checkpoint_sha256": sha256_path(checkpoint),
        "artifact_sha256": artifact_hash,
        "protocol_sha256": protocol_sha256(),
        "source_commit": git_commit(),
        "training_identity": identity,
        "parameters": experiment11_parameter_report(model),
        "weight_metrics": weight_specialization_metrics(model),
        "activation_metrics": probe_model(model, input_ids, device=device),
    }
    immutable_json(output, payload)


def evaluate_command(args) -> None:
    milestone = load_spec()["screening_milestone"]
    if args.split not in SPLITS:
        raise ValueError("unknown fixed split")
    if args.split == "confirmation":
        if args.selection_manifest is None:
            raise ValueError("confirmation evaluation requires a frozen selection manifest")
        validate_selection_manifest(Path(args.selection_manifest), args.results_root)
    checkpoint = Path(args.checkpoint).resolve()
    identity = validate_checkpoint(checkpoint, args.run_id, milestone)
    payload, artifact_hash = verify_artifact(Path(args.artifact).resolve())
    device = torch.device(args.device)
    model = load_model(checkpoint, args.run_id, device=device, dtype=args.dtype)
    result = evaluate_split(
        model,
        payload[f"{args.split}_input_ids"],
        batch_size=args.batch_size,
        device=device,
    )
    immutable_json(Path(args.output).resolve(), {
        "format_version": 1,
        "created_at": utc_now(),
        "experiment": "11-soft-query-specialization",
        "run_id": args.run_id,
        "family": run_spec(args.run_id).family,
        "lambda": run_spec(args.run_id).lambda_value,
        "milestone": milestone,
        "split": args.split,
        "checkpoint_sha256": sha256_path(checkpoint),
        "artifact_sha256": artifact_hash,
        "protocol_sha256": protocol_sha256(),
        "source_commit": git_commit(),
        "training_identity": identity,
        "parameters": experiment11_parameter_report(model),
        "metrics": result,
    })


def evaluation_path(results_root: Path, run_id: str, split: str) -> Path:
    return results_root / "evaluations" / run_id / f"{split}.json"


def probe_path(results_root: Path, run_id: str, milestone: int, split="discovery") -> Path:
    return results_root / "probes" / run_id / f"step-{milestone}-{split}.json"


def load_evaluation(results_root: Path, run_id: str, split: str) -> dict:
    path = evaluation_path(results_root, run_id, split)
    if not path.is_file():
        raise FileNotFoundError(f"missing {split} evaluation for {run_id}")
    result = read_json(path)
    expected = {
        "run_id": run_id,
        "split": split,
        "artifact_sha256": load_spec()["fixed_evaluation"]["sha256"],
        "protocol_sha256": protocol_sha256(),
        "content_sha256": canonical_hash(result),
    }
    mismatch = {key: [value, result.get(key)] for key, value in expected.items()
                if result.get(key) != value}
    if mismatch:
        raise ValueError(f"evaluation identity mismatch: {mismatch}")
    values = np.asarray(result["metrics"]["sequence_nlls"], dtype=float)
    if not math.isfinite(float(result["metrics"]["nll"])) or not np.isfinite(values).all():
        raise ValueError(f"non-finite evaluation for {run_id}:{split}")
    return result


def select_command(args) -> None:
    root = Path(args.results_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        validate_selection_manifest(output, root)
        return
    spec = load_spec()
    discovery = {row.run_id: load_evaluation(root, row.run_id, "discovery") for row in RUNS}
    selected = {}
    for family, candidates in spec["selection"]["candidates"].items():
        winner = min(
            candidates,
            key=lambda run_id: (
                discovery[run_id]["metrics"]["nll"], RUN_BY_ID[run_id].lambda_value
            ),
        )
        selected[family] = {
            "run_id": winner,
            "lambda": RUN_BY_ID[winner].lambda_value,
            "discovery_nll": discovery[winner]["metrics"]["nll"],
            "checkpoint_sha256": discovery[winner]["checkpoint_sha256"],
            "discovery_result_sha256": discovery[winner]["content_sha256"],
        }
    immutable_json(output, {
        "format_version": 1,
        "created_at": utc_now(),
        "experiment": "11-soft-query-specialization",
        "seed": spec["seed"],
        "milestone": spec["screening_milestone"],
        "protocol_sha256": protocol_sha256(),
        "source_commit": git_commit(),
        "selection_rule": spec["selection"],
        "selected": selected,
        "all_discovery_results": {
            run_id: {
                "nll": result["metrics"]["nll"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "result_sha256": result["content_sha256"],
            }
            for run_id, result in discovery.items()
        },
    })


def validate_selection_manifest(path: Path, results_root) -> dict:
    if not path.is_file():
        raise FileNotFoundError("selection manifest does not exist")
    payload = read_json(path)
    if payload.get("content_sha256") != canonical_hash(payload):
        raise ValueError("selection manifest content hash mismatch")
    if payload.get("protocol_sha256") != protocol_sha256():
        raise ValueError("selection manifest protocol hash mismatch")
    root = Path(results_root).resolve()
    for run_id, recorded in payload.get("all_discovery_results", {}).items():
        current = load_evaluation(root, run_id, "discovery")
        expected = {
            "nll": current["metrics"]["nll"],
            "checkpoint_sha256": current["checkpoint_sha256"],
            "result_sha256": current["content_sha256"],
        }
        if recorded != expected:
            raise ValueError(f"selection input changed after freezing: {run_id}")
    expected_ids = set(RUN_BY_ID)
    if set(payload.get("all_discovery_results", {})) != expected_ids:
        raise ValueError("selection manifest does not cover all nine runs")
    for family, candidates in load_spec()["selection"]["candidates"].items():
        winner = min(candidates, key=lambda run_id: (
            payload["all_discovery_results"][run_id]["nll"],
            RUN_BY_ID[run_id].lambda_value,
        ))
        if payload.get("selected", {}).get(family, {}).get("run_id") != winner:
            raise ValueError(f"selection winner mismatch for {family}")
    return payload


def paired_bootstrap(values: np.ndarray, *, samples: int, seed: int) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("paired bootstrap requires a finite one-dimensional array")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        draws[start:start + count] = values[
            rng.integers(0, len(values), size=(count, len(values)))
        ].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(draws, .025)),
        "ci95_high": float(np.quantile(draws, .975)),
        "samples": samples,
        "seed": seed,
    }


def contrast(first: dict, second: dict, *, samples: int, seed: int) -> dict:
    first_values = np.asarray(first["metrics"]["sequence_nlls"], dtype=float)
    second_values = np.asarray(second["metrics"]["sequence_nlls"], dtype=float)
    if first_values.shape != second_values.shape:
        raise ValueError("paired evaluation arrays have different shapes")
    inference = paired_bootstrap(first_values - second_values, samples=samples, seed=seed)
    inference["aggregate_delta_nll"] = (
        first["metrics"]["nll"] - second["metrics"]["nll"]
    )
    return inference


def load_probe(root: Path, run_id: str, milestone: int, split: str) -> dict:
    path = probe_path(root, run_id, milestone, split)
    if not path.is_file():
        raise FileNotFoundError(f"missing {split} probe for {run_id} at {milestone}")
    payload = read_json(path)
    if payload.get("content_sha256") != canonical_hash(payload):
        raise ValueError(f"probe content hash mismatch: {path}")
    expected = {
        "run_id": run_id,
        "family": RUN_BY_ID[run_id].family,
        "lambda": RUN_BY_ID[run_id].lambda_value,
        "milestone": milestone,
        "split": split,
        "artifact_sha256": load_spec()["fixed_evaluation"]["sha256"],
        "protocol_sha256": protocol_sha256(),
    }
    mismatch = {key: [value, payload.get(key)] for key, value in expected.items()
                if payload.get(key) != value}
    if mismatch:
        raise ValueError(f"probe identity mismatch: {path}")
    metrics = payload.get("activation_metrics", {}).get("metrics", {})
    for name in ("local_norm", "cross_norm", "r_act", "theta_radians",
                 "component_cosine", "component_cosine_valid_fraction"):
        if name not in metrics:
            raise ValueError(f"probe missing metric {name}: {path}")
        values = np.asarray(metrics[name]["per_sequence"], dtype=float)
        if not np.isfinite(values).all() or not math.isfinite(float(metrics[name]["mean"])):
            raise ValueError(f"non-finite probe metric {name}: {path}")
    return payload


def metric_contrast(first: dict, second: dict, name: str, *, samples: int, seed: int):
    positions = first["activation_metrics"]["diagnostic_head_positions"]
    one_by_head = np.asarray(
        first["activation_metrics"]["metrics"][name]["per_sequence_head"], dtype=float
    )
    two_by_head = np.asarray(
        second["activation_metrics"]["metrics"][name]["per_sequence_head"], dtype=float
    )
    one = one_by_head[:, positions].mean(axis=1)
    two = two_by_head[:, positions].mean(axis=1)
    if one.shape != two.shape:
        raise ValueError("paired probe arrays have different shapes")
    return paired_bootstrap(one - two, samples=samples, seed=seed)


def _family_classification(nll_rows, softness) -> str:
    nll_pass = all(row["ci95_high"] < 0 for row in nll_rows)
    soft_pass = (
        softness["r_act_point"] > 0
        and softness["theta_point"] > 0
        and softness["r_act_minus_m8"]["ci95_high"] < 0
        and softness["theta_minus_m8"]["ci95_high"] < 0
    )
    if nll_pass and soft_pass:
        return "soft_specialization_supported"
    if nll_pass:
        return "performance_gain_without_demonstrated_softness"
    return "no_confirmed_soft_specialization_advantage"


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary: dict) -> str:
    lines = [
        "# Experiment 11 — Soft MHAR Query Specialization",
        "",
        "## Result",
        "",
        "This report is generated after the frozen run completes. It is a "
        "single-seed step-2,000 screen, not a convergence claim.",
        "",
        "| Family | Discovery-selected run | Lambda | Frozen classification |",
        "|---|---|---:|---|",
    ]
    for family in ("s2q8", "gslq8"):
        row = summary["families"][family]
        lines.append(
            f"| {family.upper()} | `{row['selected_run_id']}` | "
            f"{row['selected_lambda']:.2f} | `{row['classification']}` |"
        )
    lines.extend([
        "",
        "## Confirmation comparisons",
        "",
        "Negative delta NLL favors the selected intermediate model.",
        "",
        "| Family | Contrast | Delta NLL | Paired 95% CI |",
        "|---|---|---:|---:|",
    ])
    for family in ("s2q8", "gslq8"):
        for row in summary["families"][family]["confirmation_nll"]:
            lines.append(
                f"| {family.upper()} | {row['id']} | "
                f"{row['aggregate_delta_nll']:+.6f} | "
                f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] |"
            )
    between = summary["selected_s2q8_minus_selected_gslq8"]
    lines.append(
        f"| S2Q8 vs GSLQ8 | selected-minus-selected | "
        f"{between['aggregate_delta_nll']:+.6f} | "
        f"[{between['ci95_low']:+.6f}, {between['ci95_high']:+.6f}] |"
    )
    lines.extend([
        "",
        "## Complete NLL curves",
        "",
        "| Split | Run | Family | Lambda | NLL |",
        "|---|---|---|---:|---:|",
    ])
    for row in summary["curves"]:
        lines.append(
            f"| {row['split']} | `{row['run_id']}` | {row['family']} | "
            f"{row['lambda']:.2f} | {row['nll']:.6f} |"
        )
    lines.extend([
        "",
        "## Effective-softness confirmation",
        "",
        "The M8 comparison uses the same query-head slots as the selected family: "
        "both slots for S2Q8 and the even slot for GSLQ8.",
        "",
        "| Family | R_act | Theta (rad) | R_act minus M8 95% CI | Theta minus M8 95% CI |",
        "|---|---:|---:|---:|---:|",
    ])
    for family in ("s2q8", "gslq8"):
        softness = summary["families"][family]["effective_softness"]
        r_delta = softness["r_act_minus_m8"]
        t_delta = softness["theta_minus_m8"]
        lines.append(
            f"| {family.upper()} | {softness['r_act_point']:.6f} | "
            f"{softness['theta_point']:.6f} | "
            f"[{r_delta['ci95_low']:+.6f}, {r_delta['ci95_high']:+.6f}] | "
            f"[{t_delta['ci95_low']:+.6f}, {t_delta['ci95_high']:+.6f}] |"
        )
    lines.extend([
        "",
        "## Matched systems accounting",
        "",
        "All rows retain the same dense trainable Q/K/V parameters and dense "
        "projection-MAC proxy. Throughput and memory are measured diagnostics, not "
        "claimed speedups.",
        "",
        "| Run | Trainable params | QKV params | QKV MAC/token | Train hours | Eval tok/s | Peak GiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["systems"]:
        hours = None if row["training_seconds"] is None else row["training_seconds"] / 3600
        gib = None if row["confirmation_peak_memory_bytes"] is None else row["confirmation_peak_memory_bytes"] / 2**30
        lines.append(
            f"| `{row['run_id']}` | {row['trainable_parameters']:,} | "
            f"{row['qkv_parameters']:,} | {row['qkv_macs_per_token']:,} | "
            f"{hours:.3f} | {row['confirmation_tokens_per_second']:.1f} | {gib:.2f} |"
            if hours is not None and gib is not None else
            f"| `{row['run_id']}` | {row['trainable_parameters']:,} | "
            f"{row['qkv_parameters']:,} | {row['qkv_macs_per_token']:,} | n/a | "
            f"{row['confirmation_tokens_per_second']:.1f} | n/a |"
        )
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "- Lambda is a fixed training bias, not a final information percentage.",
        "- Effective softness is measured on fixed held-out examples after training.",
        "- All dense Q parameters and dense projection MACs are retained; this is not a speedup experiment.",
        "- Paired intervals quantify sequence sampling uncertainty, not seed uncertainty.",
        "- No 20,000-step or multi-seed continuation is authorized by this screen.",
        "",
    ])
    wandb = summary.get("wandb") or {}
    lines.extend(["## W&B", ""])
    for run_id, link in summary.get("training_wandb", {}).items():
        lines.append(f"- {run_id}: {link or 'unavailable'}")
    if wandb:
        lines.append(f"- Analysis: {wandb.get('run_url')}")
    lines.append("")
    return "\n".join(lines)


def analyze_command(args) -> None:
    root = Path(args.results_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection = validate_selection_manifest(Path(args.selection_manifest), root)
    spec = load_spec()
    samples = spec["inference"]["bootstrap_samples"]
    seed = spec["inference"]["bootstrap_seed"]
    evaluations = {
        split: {run_id: load_evaluation(root, run_id, split) for run_id in RUN_BY_ID}
        for split in SPLITS
    }
    curves = []
    for split in SPLITS:
        for row in RUNS:
            curves.append({
                "split": split,
                "run_id": row.run_id,
                "family": row.family,
                "lambda": row.lambda_value,
                "nll": evaluations[split][row.run_id]["metrics"]["nll"],
                "ppl": evaluations[split][row.run_id]["metrics"]["ppl"],
                "throughput": evaluations[split][row.run_id]["metrics"]["tokens_per_second"],
                "peak_memory_bytes": evaluations[split][row.run_id]["metrics"]["peak_cuda_allocated_bytes"],
            })
    family_summary = {}
    for family in ("s2q8", "gslq8"):
        selected_id = selection["selected"][family]["run_id"]
        endpoint = f"{family}-l000"
        comparisons = []
        for comparison_id, other in (("selected-minus-lambda0", endpoint),
                                      ("selected-minus-m8", "m8-l100")):
            comparisons.append({
                "id": comparison_id,
                "first": selected_id,
                "second": other,
                **contrast(
                    evaluations["confirmation"][selected_id],
                    evaluations["confirmation"][other],
                    samples=samples,
                    seed=seed + len(comparisons),
                ),
            })
        selected_probe = load_probe(root, selected_id, 2000, "confirmation")
        m8_probe = load_probe(root, "m8-l100", 2000, "confirmation")
        softness = {
            "r_act_point": selected_probe["activation_metrics"]["metrics"]["r_act"]["mean"],
            "theta_point": selected_probe["activation_metrics"]["metrics"]["theta_radians"]["mean"],
            "r_act_minus_m8": metric_contrast(
                selected_probe, m8_probe, "r_act", samples=samples, seed=seed + 10
            ),
            "theta_minus_m8": metric_contrast(
                selected_probe, m8_probe, "theta_radians", samples=samples, seed=seed + 11
            ),
        }
        family_summary[family] = {
            "selected_run_id": selected_id,
            "selected_lambda": RUN_BY_ID[selected_id].lambda_value,
            "confirmation_nll": comparisons,
            "effective_softness": softness,
            "classification": _family_classification(comparisons, softness),
        }
    between = contrast(
        evaluations["confirmation"][family_summary["s2q8"]["selected_run_id"]],
        evaluations["confirmation"][family_summary["gslq8"]["selected_run_id"]],
        samples=samples,
        seed=seed + 20,
    )
    systems = []
    training_wandb = {}
    for row in RUNS:
        training_manifest = read_json(
            root / "training" / row.run_id / "training_run_manifest.json"
        )
        checkpoint_manifest = read_json(
            root / "training" / row.run_id / "step-2000/training_manifest.json"
        )
        wandb_metadata = training_manifest.get("wandb") or {}
        run_url = wandb_metadata.get("run_url")
        if run_url is None and wandb_metadata.get("run_id"):
            entity = wandb_metadata.get("entity") or args.wandb_entity
            project = wandb_metadata.get("project", args.wandb_project).replace(" ", "%20")
            if entity:
                run_url = f"https://wandb.ai/{entity}/{project}/runs/{wandb_metadata['run_id']}"
        training_wandb[row.run_id] = run_url
        evaluation = evaluations["confirmation"][row.run_id]
        parameters = evaluation["parameters"]
        systems.append({
            "run_id": row.run_id,
            "trainable_parameters": parameters["trainable_parameters"],
            "qkv_parameters": parameters["qkv_parameters"],
            "qkv_macs_per_token": parameters["qkv_macs_per_token"],
            "training_seconds": checkpoint_manifest.get("elapsed_training_seconds"),
            "confirmation_tokens_per_second": evaluation["metrics"]["tokens_per_second"],
            "confirmation_peak_memory_bytes": evaluation["metrics"]["peak_cuda_allocated_bytes"],
        })
    trajectories = []
    for row in RUNS:
        for milestone in spec["probe_milestones"]:
            probe = load_probe(root, row.run_id, milestone, "discovery")
            metrics = probe["activation_metrics"]["metrics"]
            weight_rows = probe["weight_metrics"]["rows"]
            trajectories.append({
                "run_id": row.run_id,
                "family": row.family,
                "lambda": row.lambda_value,
                "milestone": milestone,
                "r_weight": float(np.mean([item["r_weight"] for item in weight_rows])),
                "r_act": metrics["r_act"]["mean"],
                "theta_radians": metrics["theta_radians"]["mean"],
                "component_cosine": metrics["component_cosine"]["mean"],
            })
    weight_trajectories = []
    expected_weight_steps = list(range(100, 2001, 100))
    for row in RUNS:
        path = root / "training" / row.run_id / "experiment11_weight_metrics.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing 100-step weight metrics for {row.run_id}")
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if [record.get("step") for record in records] != expected_weight_steps:
            raise ValueError(f"weight metric schedule mismatch for {row.run_id}")
        for record in records:
            values = [item["r_weight"] for item in record["rows"]]
            if not values or not np.isfinite(values).all():
                raise ValueError(f"non-finite weight metrics for {row.run_id}")
            weight_trajectories.append({
                "run_id": row.run_id,
                "family": row.family,
                "lambda": row.lambda_value,
                "milestone": record["step"],
                "r_weight": float(np.mean(values)),
            })
    summary = {
        "format_version": 1,
        "created_at": utc_now(),
        "experiment": "11-soft-query-specialization",
        "seed": spec["seed"],
        "milestone": 2000,
        "single_seed_screen_only": True,
        "artifact_sha256": spec["fixed_evaluation"]["sha256"],
        "protocol_sha256": protocol_sha256(),
        "selection_manifest_sha256": read_json(Path(args.selection_manifest))["content_sha256"],
        "families": family_summary,
        "selected_s2q8_minus_selected_gslq8": between,
        "curves": curves,
        "trajectories": trajectories,
        "weight_trajectories": weight_trajectories,
        "systems": systems,
        "training_wandb": training_wandb,
    }
    run = None
    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=spec["wandb_group"],
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            job_type="experiment11-analysis",
            tags=["experiment-11", "soft-query", "analysis", "seed42", "step2000"],
            config={"protocol_sha256": protocol_sha256(), "selection": selection},
        )
        run.log({
            f"{split}/{row.run_id}/nll": evaluations[split][row.run_id]["metrics"]["nll"]
            for split in SPLITS for row in RUNS
        })
        summary["wandb"] = {"run_id": run.id, "run_url": run.url}
    immutable_json(output / "summary.json", summary)
    _write_csv(
        output / "nll_curves.csv",
        ["split", "run_id", "family", "lambda", "nll", "ppl", "throughput", "peak_memory_bytes"],
        curves,
    )
    _write_csv(
        output / "softness_trajectories.csv",
        ["run_id", "family", "lambda", "milestone", "r_weight", "r_act", "theta_radians", "component_cosine"],
        trajectories,
    )
    _write_csv(
        output / "weight_trajectories.csv",
        ["run_id", "family", "lambda", "milestone", "r_weight"],
        weight_trajectories,
    )
    report = render_report(summary)
    (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    if run:
        run.summary.update({
            f"{family}/classification": family_summary[family]["classification"]
            for family in family_summary
        })
        run.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("--run-id", choices=RUN_BY_ID, required=True)
    probe.add_argument("--milestone", type=int, required=True)
    probe.add_argument("--split", choices=SPLITS, default="discovery")
    probe.add_argument("--checkpoint", required=True)
    probe.add_argument("--artifact", required=True)
    probe.add_argument("--output", required=True)
    probe.add_argument("--results-root", default=".")
    probe.add_argument("--selection-manifest")
    probe.add_argument("--device", default="cuda")
    probe.add_argument("--dtype", default="bf16")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run-id", choices=RUN_BY_ID, required=True)
    evaluate.add_argument("--split", choices=SPLITS, required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--artifact", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--results-root", default=".")
    evaluate.add_argument("--selection-manifest")
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--dtype", default="bf16")
    evaluate.add_argument("--batch-size", type=int, default=1)
    select = commands.add_parser("select")
    select.add_argument("--results-root", required=True)
    select.add_argument("--output", required=True)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--results-root", required=True)
    analyze.add_argument("--selection-manifest", required=True)
    analyze.add_argument("--output-dir", required=True)
    analyze.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    analyze.add_argument("--wandb-project", default="MHAR Stuff")
    analyze.add_argument("--wandb-entity")
    analyze.add_argument("--wandb-run-name", default="mhar-exp11-soft-query-analysis-seed42-step2000")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "probe":
        probe_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "select":
        select_command(args)
    elif args.command == "analyze":
        analyze_command(args)


if __name__ == "__main__":
    main()
