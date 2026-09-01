#!/usr/bin/env python3
"""Fixed-data evaluation and descriptive analysis for Experiment 6 screening."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.experiments.experiment1_partition_compatibility import (
    atomic_write_json,
    dtype_from_name,
    git_commit,
    load_fixed_eval_artifact,
    sha256_path,
    utc_now,
)
from src.experiments.experiment6_coupled_qkv import (
    Experiment6BaselineForCausalLM,
    Experiment6MHARForCausalLM,
    experiment6_parameter_report,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiment6/screening.json"


def load_spec():
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in payload["runs"]}
    if len(rows) != 7:
        raise ValueError("Experiment 6 must contain exactly seven unique runs")
    return payload, rows


def validate_architecture(model, row):
    expected = {
        "hidden_size": 1280,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 5120,
        "experiment6_qkv_groups": row["qkv_groups"],
        "experiment6_variant": row["id"],
    }
    if row["mode"] == "full_mh":
        expected.update({
            "attnres_mode": "full_mh",
            "attnres_num_heads": row["attnres_heads"],
        })
    observed = {key: getattr(model.config, key, None) for key in expected}
    mismatch = {
        key: {"expected": value, "observed": observed[key]}
        for key, value in expected.items() if observed[key] != value
    }
    if mismatch:
        raise ValueError(f"checkpoint architecture mismatch: {json.dumps(mismatch)}")
    return observed


def validate_training_manifest(manifest, row, spec):
    """Reject checkpoints that do not match the frozen screening recipe."""

    identity = manifest.get("run_identity", {})
    expected = {
        "global_step": spec["screening_milestone"],
        "mode": row["mode"],
        "attnres_heads": row["attnres_heads"] if row["mode"] == "full_mh" else 8,
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
        "experiment6_variant": row["id"],
        "experiment6_qkv_groups": row["qkv_groups"],
    }
    observed = {"global_step": manifest.get("global_step")}
    observed.update({key: identity.get(key) for key in expected if key != "global_step"})
    mismatch = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items() if observed.get(key) != value
    }
    if mismatch:
        raise ValueError(f"checkpoint training manifest mismatch: {json.dumps(mismatch)}")
    return observed


def load_model(checkpoint, row, *, device, dtype):
    if row["mode"] == "baseline":
        cls = Experiment6BaselineForCausalLM if row["qkv_groups"] else Qwen3ForCausalLM
    elif row["mode"] == "full_mh":
        if row["qkv_groups"]:
            cls = Experiment6MHARForCausalLM
        else:
            from src.attention_residuals.modeling_qwen3_attnres import Qwen3AttnResForCausalLM
            cls = Qwen3AttnResForCausalLM
    else:
        raise ValueError(f"unsupported Experiment 6 mode: {row['mode']}")
    model = cls.from_pretrained(str(checkpoint), dtype=dtype).to(device=device).eval()
    return model, validate_architecture(model, row)


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


def evaluate_command(args):
    spec, rows = load_spec()
    if args.variant not in rows:
        raise ValueError(f"unknown variant {args.variant!r}")
    row = rows[args.variant]
    if not row["train"]:
        raise ValueError(f"{args.variant} reuses an accepted result and needs no new evaluation")
    checkpoint = Path(args.checkpoint).resolve()
    artifact = Path(args.artifact).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload, artifact_hash = load_fixed_eval_artifact(artifact)
    if artifact_hash != spec["fixed_evaluation"]["sha256"]:
        raise ValueError("fixed evaluation artifact hash differs from preregistration")
    checkpoint_manifest = json.loads(
        (checkpoint / "training_manifest.json").read_text(encoding="utf-8"))
    training_recipe = validate_training_manifest(checkpoint_manifest, row, spec)
    identity = {
        "experiment": "6-coupled-qkv-screening",
        "variant": args.variant,
        "seed": spec["seed"],
        "milestone": spec["screening_milestone"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_path(checkpoint),
        "artifact": str(artifact),
        "artifact_sha256": artifact_hash,
        "source_commit": git_commit(),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "training_recipe": training_recipe,
    }
    manifest_path = output / "run_manifest.json"
    result_path = output / "result.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["run_identity"] != identity:
            raise RuntimeError("existing evaluation identity differs")
        if result_path.exists():
            return
    else:
        manifest = {"created_at": utc_now(), "run_identity": identity}

    device = torch.device(args.device)
    model, architecture = load_model(
        checkpoint, row, device=device, dtype=dtype_from_name(args.dtype))
    manifest["architecture"] = architecture
    manifest["parameters"] = experiment6_parameter_report(model)
    run = None
    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            job_type="experiment6-fixed-evaluation",
            tags=["experiment-6", "fixed-eval", args.variant],
            config={**identity, **architecture, **manifest["parameters"]},
        )
        manifest["wandb"] = {"run_id": run.id, "run_url": run.url}
    atomic_write_json(manifest_path, manifest)

    splits = {}
    for split in ("discovery", "confirmation"):
        metrics = evaluate_split(
            model, payload[f"{split}_input_ids"],
            batch_size=args.batch_size, device=device)
        splits[split] = metrics
        if run is not None:
            run.log({f"{split}/{key}": value for key, value in metrics.items()
                     if key != "sequence_nlls"})
    result = {
        "format_version": 1,
        "created_at": utc_now(),
        "variant": args.variant,
        "milestone": spec["screening_milestone"],
        "artifact_sha256": artifact_hash,
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "architecture": architecture,
        "parameters": manifest["parameters"],
        "splits": splits,
    }
    atomic_write_json(result_path, result)
    if run is not None:
        run.summary.update({
            "discovery_nll": splits["discovery"]["nll"],
            "confirmation_nll": splits["confirmation"]["nll"],
        })
        run.finish()


def bootstrap_contrast(results, terms, split, *, samples=10_000, seed=20260902):
    arrays = {
        variant: np.asarray(results[variant]["splits"][split]["sequence_nlls"], dtype=float)
        for variant in terms
    }
    lengths = {len(array) for array in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("paired sequence arrays have different lengths")
    values = sum(coefficient * arrays[variant] for variant, coefficient in terms.items())
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = np.empty(samples, dtype=float)
    for offset in range(0, samples, 1000):
        count = min(1000, samples - offset)
        indices = rng.integers(0, n, size=(count, n))
        draws[offset:offset + count] = values[indices].mean(axis=1)
    return {
        "mean_delta_nll": float(values.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def load_results(results_root):
    spec, rows = load_spec()
    results = {}
    artifact_hashes = set()
    for variant, row in rows.items():
        path = (ROOT / row["existing_result"] if not row["train"]
                else Path(results_root) / variant / "result.json")
        result = json.loads(path.read_text(encoding="utf-8"))
        artifact_hashes.add(result["artifact_sha256"])
        results[variant] = result
    if artifact_hashes != {spec["fixed_evaluation"]["sha256"]}:
        raise ValueError("not all results use the preregistered fixed artifact")
    return spec, results


def nonfinite_results(results):
    failures = []
    for variant, result in results.items():
        for split in ("discovery", "confirmation"):
            metrics = result["splits"][split]
            values = np.asarray(metrics["sequence_nlls"], dtype=float)
            if not math.isfinite(float(metrics["nll"])) or not np.isfinite(values).all():
                failures.append(f"{variant}:{split}")
    return failures


def analyze_command(args):
    spec, results = load_results(args.results_root)
    nonfinite = nonfinite_results(results)
    contrasts = []
    for contrast in ([] if nonfinite else spec["primary_contrasts"]):
        row = {"id": contrast["id"], "terms": contrast["terms"], "splits": {}}
        for split in ("discovery", "confirmation"):
            aggregate = sum(
                coefficient * results[variant]["splits"][split]["nll"]
                for variant, coefficient in contrast["terms"].items())
            row["splits"][split] = {
                "aggregate_delta_nll": aggregate,
                **bootstrap_contrast(results, contrast["terms"], split),
            }
        contrasts.append(row)
    catastrophic_ids = set(spec["catastrophic_rule"]["comparisons"])
    threshold = spec["catastrophic_rule"]["maximum_delta_nll"]
    catastrophic = [
        row["id"] for row in contrasts
        if row["id"] in catastrophic_ids
        and row["splits"]["confirmation"]["aggregate_delta_nll"] > threshold
    ]
    catastrophic.extend(f"nonfinite:{item}" for item in nonfinite)
    summary = {
        "format_version": 1,
        "created_at": utc_now(),
        "experiment": "6-coupled-qkv-screening",
        "seed": spec["seed"],
        "milestone": spec["screening_milestone"],
        "single_seed_screen_only": True,
        "contrasts": contrasts,
        "catastrophic_designs": catastrophic,
        "decision": "reject_catastrophic_designs" if catastrophic else "eligible_for_review",
        "interpretation_limit": spec["screening_rule"],
    }
    run = None
    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            job_type="experiment6-screening-analysis",
            tags=["experiment-6", "screening-analysis", "single-seed"],
            config={
                "seed": spec["seed"],
                "milestone": spec["screening_milestone"],
                "single_seed_screen_only": True,
            },
        )
        summary["wandb"] = {"run_id": run.id, "run_url": run.url}
        for row in contrasts:
            for split, metrics in row["splits"].items():
                run.log({
                    f"{split}/{row['id']}/delta_nll": metrics["aggregate_delta_nll"],
                    f"{split}/{row['id']}/ci95_low": metrics["ci95_low"],
                    f"{split}/{row['id']}/ci95_high": metrics["ci95_high"],
                })
        run.summary["decision"] = summary["decision"]
        run.summary["catastrophic_designs"] = catastrophic
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", summary)
    with (output / "contrasts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["contrast", "split", "delta_nll", "ci95_low", "ci95_high"])
        for row in contrasts:
            for split, metrics in row["splits"].items():
                writer.writerow([
                    row["id"], split, metrics["aggregate_delta_nll"],
                    metrics["ci95_low"], metrics["ci95_high"],
                ])
    if run is not None:
        run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--variant", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--artifact", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    evaluate.add_argument("--wandb-project", default="MHAR Stuff")
    evaluate.add_argument("--wandb-entity", default=None)
    evaluate.add_argument("--wandb-group", default="mhar-exp6-coupled-qkv-screen-seed42-eval")
    evaluate.add_argument("--wandb-run-name", default=None)
    evaluate.set_defaults(func=evaluate_command)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--results-root", required=True)
    analyze.add_argument("--output-dir", required=True)
    analyze.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    analyze.add_argument("--wandb-project", default="MHAR Stuff")
    analyze.add_argument("--wandb-entity", default=None)
    analyze.add_argument("--wandb-group", default="mhar-exp6-coupled-qkv-screen-seed42-eval")
    analyze.add_argument("--wandb-run-name", default="mhar-exp6-screening-analysis-seed42")
    analyze.set_defaults(func=analyze_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
