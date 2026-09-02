#!/usr/bin/env python3
"""Fixed-data evaluation and frozen single-seed analysis for Experiment 7."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.experiments.experiment1_partition_compatibility import (
    atomic_write_json, dtype_from_name, git_commit, load_fixed_eval_artifact,
    sha256_path, utc_now,
)
from src.experiments.experiment6_screening import evaluate_split
from src.experiments.experiment7_local_q import (
    Experiment7BaselineForCausalLM, Experiment7MHARForCausalLM,
    experiment7_parameter_report,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiment7/screening.json"
NEW_VARIANTS = ("lq4", "lq8", "blq4", "blq8")


def load_spec():
    spec = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in spec["runs"]}
    if len(rows) != 7 or set(NEW_VARIANTS) - set(rows):
        raise ValueError("Experiment 7 config must contain seven unique frozen rows")
    return spec, rows


def validate_architecture(model, row):
    expected = {
        "hidden_size": 1280, "num_hidden_layers": 36,
        "num_attention_heads": 16, "num_key_value_heads": 8,
        "intermediate_size": 5120,
        "experiment7_local_q_groups": row["local_q_groups"],
        "experiment7_variant": row["id"],
    }
    if row["mode"] == "full_mh":
        expected.update(attnres_mode="full_mh", attnres_num_heads=row["attnres_heads"])
    observed = {key: getattr(model.config, key, None) for key in expected}
    mismatch = {key: {"expected": value, "observed": observed[key]}
                for key, value in expected.items() if observed[key] != value}
    if mismatch:
        raise ValueError(f"checkpoint architecture mismatch: {json.dumps(mismatch)}")
    return observed


def validate_training_manifest(manifest, row, spec):
    identity = manifest.get("run_identity", {})
    expected = {
        "global_step": 2000, "mode": row["mode"],
        "attnres_heads": row["attnres_heads"] if row["mode"] == "full_mh" else 8,
        "hidden_size": 1280, "num_layers": 36, "num_heads": 16,
        "num_kv_heads": 8, "intermediate_size": 5120, "seq_len": 1024,
        "steps": 20000, "global_batch_size": 32, "lr": 5e-4,
        "lr_min": 5e-5, "warmup": 1000, "seed": spec["seed"],
        "experiment7_variant": row["id"],
        "experiment7_local_q_groups": row["local_q_groups"],
    }
    observed = {"global_step": manifest.get("global_step")}
    observed.update({key: identity.get(key) for key in expected if key != "global_step"})
    mismatch = {key: {"expected": value, "observed": observed.get(key)}
                for key, value in expected.items() if observed.get(key) != value}
    if mismatch:
        raise ValueError(f"checkpoint training manifest mismatch: {json.dumps(mismatch)}")
    return observed


def evaluate_command(args):
    spec, rows = load_spec()
    row = rows.get(args.variant)
    if row is None or not row["train"]:
        raise ValueError(f"{args.variant!r} is not a new Experiment 7 run")
    checkpoint, artifact, output = map(
        lambda value: Path(value).resolve(), (args.checkpoint, args.artifact, args.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    payload, artifact_hash = load_fixed_eval_artifact(artifact)
    if artifact_hash != spec["fixed_evaluation"]["sha256"]:
        raise ValueError("fixed evaluation artifact hash differs from preregistration")
    manifest = json.loads((checkpoint / "training_manifest.json").read_text())
    recipe = validate_training_manifest(manifest, row, spec)
    cls = Experiment7MHARForCausalLM if row["mode"] == "full_mh" else Experiment7BaselineForCausalLM
    device = torch.device(args.device)
    model = cls.from_pretrained(
        str(checkpoint), dtype=dtype_from_name(args.dtype)).to(device=device).eval()
    architecture = validate_architecture(model, row)
    parameters = experiment7_parameter_report(model)
    identity = {
        "experiment": "7-local-q-global-kv", "variant": args.variant,
        "seed": spec["seed"], "milestone": 2000,
        "checkpoint_sha256": sha256_path(checkpoint),
        "artifact_sha256": artifact_hash, "source_commit": git_commit(),
        "dtype": args.dtype, "batch_size": args.batch_size, "training_recipe": recipe,
    }
    run = None
    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            group=args.wandb_group, name=args.wandb_run_name, mode=args.wandb_mode,
            job_type="experiment7-fixed-evaluation",
            tags=["experiment-7", "fixed-eval", args.variant],
            config={**identity, **architecture, **parameters},
        )
    splits = {}
    for split in ("discovery", "confirmation"):
        splits[split] = evaluate_split(
            model, payload[f"{split}_input_ids"], batch_size=args.batch_size, device=device)
        if run:
            run.log({f"{split}/{key}": value for key, value in splits[split].items()
                     if key != "sequence_nlls"})
    result = {
        "format_version": 1, "created_at": utc_now(), "variant": args.variant,
        "milestone": 2000, "artifact_sha256": artifact_hash,
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "architecture": architecture, "parameters": parameters, "splits": splits,
    }
    atomic_write_json(output / "run_manifest.json", {
        "created_at": utc_now(), "run_identity": identity,
        "wandb": ({"run_id": run.id, "run_url": run.url} if run else None),
    })
    atomic_write_json(output / "result.json", result)
    if run:
        run.summary.update({f"{split}_nll": splits[split]["nll"]
                            for split in ("discovery", "confirmation")})
        run.finish()


def bootstrap(results, terms, split, *, samples=10_000, seed=20260903):
    arrays = {key: np.asarray(results[key]["splits"][split]["sequence_nlls"], dtype=float)
              for key in terms}
    if len({len(value) for value in arrays.values()}) != 1:
        raise ValueError("paired sequence arrays differ in length")
    values = sum(coefficient * arrays[key] for key, coefficient in terms.items())
    rng, draws, n = np.random.default_rng(seed), np.empty(samples), len(values)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        draws[start:start + count] = values[
            rng.integers(0, n, size=(count, n))].mean(axis=1)
    return {"mean_delta_nll": float(values.mean()),
            "ci95_low": float(np.quantile(draws, .025)),
            "ci95_high": float(np.quantile(draws, .975)),
            "bootstrap_samples": samples, "bootstrap_seed": seed}


def load_results(results_root):
    spec, rows = load_spec()
    results = {}
    for variant, row in rows.items():
        path = (Path(results_root) / variant / "result.json" if row["train"]
                else ROOT / row["existing_result"])
        results[variant] = json.loads(path.read_text(encoding="utf-8"))
    for variant, relative in spec["diagnostic_results"].items():
        results[variant] = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    hashes = {result["artifact_sha256"] for result in results.values()}
    if hashes != {spec["fixed_evaluation"]["sha256"]}:
        raise ValueError("results do not share the frozen artifact")
    return spec, results


def analyze_command(args):
    spec, results = load_results(args.results_root)
    nonfinite = [f"{variant}:{split}" for variant, result in results.items()
                 for split in ("discovery", "confirmation")
                 if not math.isfinite(float(result["splits"][split]["nll"])) or
                 not np.isfinite(result["splits"][split]["sequence_nlls"]).all()]
    contrasts = []
    for contrast in ([] if nonfinite else spec["contrasts"]):
        row = {"id": contrast["id"], "terms": contrast["terms"],
               "diagnostic": contrast.get("diagnostic", False), "splits": {}}
        for split in ("discovery", "confirmation"):
            aggregate = sum(coef * results[key]["splits"][split]["nll"]
                            for key, coef in contrast["terms"].items())
            row["splits"][split] = {"aggregate_delta_nll": aggregate,
                                     **bootstrap(results, contrast["terms"], split)}
        contrasts.append(row)
    threshold = spec["catastrophic_rule"]["maximum_delta_nll"]
    catastrophic_ids = set(spec["catastrophic_rule"]["comparisons"])
    catastrophic = [row["id"] for row in contrasts if row["id"] in catastrophic_ids
                    and row["splits"]["confirmation"]["aggregate_delta_nll"] > threshold]
    catastrophic += [f"nonfinite:{value}" for value in nonfinite]
    summary = {
        "format_version": 1, "created_at": utc_now(),
        "experiment": "7-local-q-global-kv", "seed": 42, "milestone": 2000,
        "single_seed_screen_only": True, "contrasts": contrasts,
        "catastrophic_designs": catastrophic,
        "decision": "reject_catastrophic_designs" if catastrophic else "eligible_for_review",
        "interpretation_limit": spec["screening_rule"],
    }
    run = None
    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         group=args.wandb_group, name=args.wandb_run_name,
                         mode=args.wandb_mode, job_type="experiment7-screening-analysis",
                         tags=["experiment-7", "screening-analysis", "single-seed"])
        summary["wandb"] = {"run_id": run.id, "run_url": run.url}
        for row in contrasts:
            for split, metrics in row["splits"].items():
                run.log({f"{split}/{row['id']}/delta_nll": metrics["aggregate_delta_nll"],
                         f"{split}/{row['id']}/ci95_low": metrics["ci95_low"],
                         f"{split}/{row['id']}/ci95_high": metrics["ci95_high"]})
        run.summary.update(decision=summary["decision"], catastrophic_designs=catastrophic)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", summary)
    with (output / "contrasts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(
            ["contrast", "split", "delta_nll", "ci95_low", "ci95_high"])
        for row in contrasts:
            for split, metrics in row["splits"].items():
                writer.writerow([row["id"], split, metrics["aggregate_delta_nll"],
                                 metrics["ci95_low"], metrics["ci95_high"]])
    if run: run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--variant", required=True); evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--artifact", required=True); evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--device", default="cuda"); evaluate.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    evaluate.add_argument("--batch-size", type=int, default=1); evaluate.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    evaluate.add_argument("--wandb-project", default="MHAR Stuff"); evaluate.add_argument("--wandb-entity", default=None)
    evaluate.add_argument("--wandb-group", default="mhar-exp7-local-q-global-kv-screen-seed42-eval"); evaluate.add_argument("--wandb-run-name", default=None)
    evaluate.set_defaults(func=evaluate_command)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--results-root", required=True); analyze.add_argument("--output-dir", required=True)
    analyze.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled")); analyze.add_argument("--wandb-project", default="MHAR Stuff")
    analyze.add_argument("--wandb-entity", default=None); analyze.add_argument("--wandb-group", default="mhar-exp7-local-q-global-kv-screen-seed42-eval")
    analyze.add_argument("--wandb-run-name", default="mhar-exp7-screening-analysis-seed42"); analyze.set_defaults(func=analyze_command)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
