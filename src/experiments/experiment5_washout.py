"""Fixed-held-out washout measurements; never infer validation NLL from training logs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from src.experiments.experiment3_actionability import apply_branch
from src.experiments.experiment3_common import evaluate_tokens, load_artifact_split, load_mhar_model, paired_bootstrap
from src.experiments.experiment4_short_horizon import PARTITIONS, ROLES
from src.training.train_scratch import atomic_write_json, sha256_file, sha256_tree

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/experiment5/protocol.json"
REFERENCE = ROOT / "results/experiment3/seed-43/signal/signal_results.jsonl"
GROUP = "mhar-exp5-fixed-validation-washout-seed43-step1500"
CANDIDATES = dict(zip(ROLES, ("remove-03", "remove-13", "native-h16")))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to replace accepted output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def source_commit():
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def spec():
    return read_json(PROTOCOL)


def load_manifest(path):
    manifest = read_json(path)
    if manifest["protocol"] != spec() or manifest["protocol_sha256"] != sha256_file(PROTOCOL):
        raise RuntimeError("protocol changed after manifest freeze")
    if manifest["source_commit"] != source_commit():
        raise RuntimeError("code must stay at the frozen clean source commit")
    if manifest["branches"] != {r: {"candidate_id": CANDIDATES[r], "partition_id": PARTITIONS[r]} for r in ROLES}:
        raise RuntimeError("branch definitions changed")
    if manifest["parent_checkpoint_sha256"] != spec()["parent_checkpoint_sha256"]:
        raise RuntimeError("wrong frozen parent identity")
    return manifest


def prepare(args):
    protocol = spec()
    parent, artifact = Path(args.parent).resolve(), Path(args.artifact).resolve()
    if sha256_tree(parent) != protocol["parent_checkpoint_sha256"]:
        raise RuntimeError("wrong parent checkpoint content hash")
    if sha256_file(artifact) != protocol["artifact_sha256"] or sha256_file(REFERENCE) != protocol["reference_results_sha256"]:
        raise RuntimeError("fixed artifact or original reference hash mismatch")
    training = read_json(parent / "training_manifest.json")
    if (training["global_step"], training["run_identity"]["seed"], training["chunks_consumed"]) != (1500, 43, 48000):
        raise RuntimeError("parent step/seed/data position mismatch")
    if not (parent / "training_state.pt").is_file():
        raise RuntimeError("parent lacks full optimizer/scheduler/RNG state")
    write_new(args.manifest, {
        "format_version": 1, "experiment": protocol["experiment"],
        "source_commit": source_commit(), "protocol": protocol,
        "protocol_sha256": sha256_file(PROTOCOL), "seed": 43,
        "source_step": 1500, "endpoint_step": 1600,
        "parent_checkpoint": str(parent), "artifact": str(artifact),
        "parent_checkpoint_sha256": protocol["parent_checkpoint_sha256"],
        "branches": {r: {"candidate_id": CANDIDATES[r], "partition_id": PARTITIONS[r]} for r in ROLES},
        "parent_training_manifest": training,
    })


def validate_checkpoint(manifest, manifest_path, path, role, offset):
    protocol = manifest["protocol"]
    if offset not in protocol["offsets"]:
        raise ValueError("unregistered offset")
    path = Path(path)
    training = read_json(path / "training_manifest.json")
    if training["global_step"] != 1500 + offset or training["run_identity"]["seed"] != 43:
        raise RuntimeError("checkpoint step or seed mismatch")
    if training["chunks_consumed"] != (1500 + offset) * 32:
        raise RuntimeError("checkpoint data position mismatch")
    if not (path / "training_state.pt").is_file() or not (path / "model.safetensors").is_file():
        raise RuntimeError("incomplete checkpoint")
    if offset:
        identity = training["run_identity"]
        branch = identity.get("branch", {})
        if branch.get("role") != role or branch.get("selection_manifest_sha256") != sha256_file(manifest_path):
            raise RuntimeError("checkpoint does not match frozen branch manifest")
        if identity["mixed_partition"] != PARTITIONS[role] or identity["source_commit"] != manifest["source_commit"]:
            raise RuntimeError("checkpoint partition or source mismatch")
        if branch.get("parent_checkpoint_sha256") != protocol["parent_checkpoint_sha256"]:
            raise RuntimeError("checkpoint has the wrong branch parent")
    elif sha256_tree(path) != protocol["parent_checkpoint_sha256"]:
        raise RuntimeError("step0 parent content changed")
    return training


def validate_metrics(metrics, protocol):
    values = np.asarray(metrics["sequence_nlls"], dtype=float)
    count = protocol["sequences_per_split"]
    if values.shape != (count,) or not np.isfinite(values).all() or not np.isfinite(metrics["nll"]):
        raise RuntimeError("invalid or incomplete fixed-eval measurements")
    if metrics["valid_tokens"] != count * (protocol["sequence_length"] - 1):
        raise RuntimeError("wrong fixed-eval token count")
    if abs(metrics["nll"] - metrics["total_nll"] / metrics["valid_tokens"]) > 1e-12:
        raise RuntimeError("inconsistent token-weighted NLL")
    if abs(metrics["nll"] - values.mean()) > 1e-5:
        raise RuntimeError("sequence NLLs do not agree with token-weighted NLL")


def evaluate(args):
    import torch
    manifest = load_manifest(args.manifest)
    protocol = manifest["protocol"]
    checkpoint = Path(manifest["parent_checkpoint"]) if args.offset == 0 else Path(args.root) / "branches" / args.role / f"step-{1500 + args.offset}"
    target = Path(args.root) / "measurements" / f"{args.role}-{args.offset:03d}.json"
    if target.exists():
        raise FileExistsError(f"refusing duplicate evaluation: {target}")
    training = validate_checkpoint(manifest, args.manifest, checkpoint, args.role, args.offset)
    payload, digest, _ = load_artifact_split(Path(manifest["artifact"]), "confirmation")
    if digest != protocol["artifact_sha256"]:
        raise RuntimeError("fixed evaluation artifact changed")
    for split in protocol["splits"]:
        if tuple(payload[f"{split}_input_ids"].shape) != (512, 1024):
            raise RuntimeError("wrong fixed split dimensions")
    device = torch.device("cuda:0")
    model, _ = load_mhar_model(checkpoint, device=device, dtype=protocol["dtype"], required_heads=16)
    apply_branch(model, manifest["branches"][args.role])
    result = {
        "role": args.role, "offset": args.offset, "absolute_step": 1500 + args.offset,
        "candidate_id": CANDIDATES[args.role], "partition_id": PARTITIONS[args.role],
        "branch_manifest_sha256": sha256_file(args.manifest),
        "artifact_sha256": digest, "source_commit": manifest["source_commit"],
        "checkpoint_sha256": sha256_tree(checkpoint), "checkpoint_manifest": training,
        "dtype": protocol["dtype"], "eval_batch_size": protocol["eval_batch_size"], "splits": {},
    }
    for split in protocol["splits"]:
        metrics = evaluate_tokens(model, payload[f"{split}_input_ids"], batch_size=protocol["eval_batch_size"], device=device)
        validate_metrics(metrics, protocol)
        result["splits"][split] = metrics
    write_new(target, result)


def collect(root, manifest, manifest_path, offsets):
    rows = {}
    for role in ROLES:
        for offset in offsets:
            row = read_json(Path(root) / "measurements" / f"{role}-{offset:03d}.json")
            expected = {"role": role, "offset": offset, "absolute_step": 1500 + offset,
                        "candidate_id": CANDIDATES[role], "partition_id": PARTITIONS[role],
                        "branch_manifest_sha256": sha256_file(manifest_path),
                        "artifact_sha256": manifest["protocol"]["artifact_sha256"],
                        "source_commit": manifest["source_commit"],
                        "dtype": "bf16", "eval_batch_size": 1}
            if any(row.get(k) != v for k, v in expected.items()):
                raise RuntimeError("measurement identity mismatch")
            for split in manifest["protocol"]["splits"]:
                validate_metrics(row["splits"][split], manifest["protocol"])
            rows[role, offset] = row
    return rows


def check_step0(rows, reference_rows, protocol):
    details = []
    for role in ROLES:
        for split in protocol["splits"]:
            candidates = [r for r in reference_rows if r["candidate_id"] == CANDIDATES[role] and r["split"] == split]
            if len(candidates) != 1:
                raise RuntimeError("reference must have one row per branch and split")
            reference, observed = candidates[0], rows[role, 0]["splits"][split]
            validate_metrics(reference, protocol)
            validate_metrics(observed, protocol)
            nll_error = abs(observed["nll"] - reference["nll"])
            seq_error = float(np.max(np.abs(np.array(observed["sequence_nlls"]) - reference["sequence_nlls"])))
            details.append({"role": role, "split": split, "nll_error": nll_error, "max_sequence_error": seq_error})
            if nll_error > protocol["step0_nll_atol"] or seq_error > protocol["step0_sequence_atol"]:
                raise RuntimeError(f"step0 failed Exp3 reproduction: {details[-1]}")
    return details


def step0_gate(args):
    manifest = load_manifest(args.manifest)
    if sha256_file(REFERENCE) != manifest["protocol"]["reference_results_sha256"]:
        raise RuntimeError("original reference changed")
    rows = collect(args.root, manifest, args.manifest, [0])
    reference = [json.loads(line) for line in REFERENCE.read_text().splitlines() if line]
    details = check_step0(rows, reference, manifest["protocol"])
    write_new(Path(args.root) / "step0_gate.json", {
        "passed": True, "branch_manifest_sha256": sha256_file(args.manifest), "details": details,
        "measurement_hashes": {f"{r}-000.json": sha256_file(Path(args.root) / "measurements" / f"{r}-000.json") for r in ROLES},
    })


def verify_step0(root, manifest_path):
    gate = read_json(Path(root) / "step0_gate.json")
    if gate.get("passed") is not True or gate["branch_manifest_sha256"] != sha256_file(manifest_path):
        raise RuntimeError("training requires a passed step0 reproduction gate")
    for name, digest in gate["measurement_hashes"].items():
        if sha256_file(Path(root) / "measurements" / name) != digest:
            raise RuntimeError("accepted step0 results changed")
    return gate


def analyze_rows(rows, protocol):
    table = []
    for split in protocol["splits"]:
        baseline_gap = rows[ROLES[0], 0]["splits"][split]["nll"] - rows[ROLES[1], 0]["splits"][split]["nll"]
        for offset in protocol["offsets"]:
            a, b, c = [rows[r, offset]["splits"][split] for r in ROLES]
            entry = {"split": split, "offset": offset, "step": 1500 + offset,
                     "A_nll": a["nll"], "B_nll": b["nll"], "C_nll": c["nll"],
                     "A_minus_B": a["nll"] - b["nll"], "A_minus_C": a["nll"] - c["nll"]}
            for name, ref in (("AB", b), ("AC", c)):
                ci = paired_bootstrap(a["sequence_nlls"], ref["sequence_nlls"],
                                      samples=protocol["bootstrap_samples"], seed=protocol["bootstrap_seed"])
                entry[f"{name}_ci95_low"], entry[f"{name}_ci95_high"] = ci["ci95_low"], ci["ci95_high"]
            margin = protocol["practical_equivalence_margin_nll"]
            entry["AB_within_practical_margin"] = entry["AB_ci95_low"] > -margin and entry["AB_ci95_high"] < margin
            entry["signed_gap_fraction_of_step0"] = entry["A_minus_B"] / baseline_gap if baseline_gap else None
            table.append(entry)
    primary = [r for r in table if r["split"] == protocol["primary_split"]]
    sustained = next((r["offset"] for i, r in enumerate(primary) if r["offset"] > 0 and all(x["AB_within_practical_margin"] for x in primary[i:])), None)
    late = [r for r in primary if r["offset"] in (50, 100)]
    persists = all(r["AB_ci95_high"] < 0 for r in late)
    # Practical negligibility and sign are separate: a tiny negative gap can satisfy both.
    interpretation = "rapid_practical_washout" if sustained is not None and sustained <= 20 else (
        "late_practical_washout" if sustained is not None else (
            "negative_gap_persists_at_50_and_100" if persists else "inconclusive_or_nonmonotonic"))
    return table, {"interpretation": interpretation, "first_sampled_sustained_practical_equivalence": sustained,
                   "negative_gap_at_both_50_and_100": persists,
                   "caveat": "Single seed; reused held-out data; pointwise paired-sequence bootstrap, not training-seed uncertainty. CI crossing zero is not equivalence. No new experiment is automatically authorized."}


def analyze(args):
    manifest = load_manifest(args.manifest)
    verify_step0(args.root, args.manifest)
    rows = collect(args.root, manifest, args.manifest, manifest["protocol"]["offsets"])
    table, decision = analyze_rows(rows, manifest["protocol"])
    out = Path(args.root) / "analysis"
    out.mkdir(exist_ok=False)
    with (out / "fixed_eval_losses.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    from figures.gen_fig_experiment5_washout import plot
    plot(table, out)
    summary = {"experiment": manifest["experiment"], "decision": decision, "rows": table,
               "branch_manifest_sha256": sha256_file(args.manifest), "source_commit": manifest["source_commit"],
               "measurement_hashes": {f"{r}-{t:03d}.json": sha256_file(Path(args.root) / "measurements" / f"{r}-{t:03d}.json") for r, t in rows}}
    if args.wandb:
        import wandb
        run = wandb.init(project="MHAR Stuff", group=GROUP, job_type="experiment5-fixed-evaluation", name="mhar-exp5-washout-analysis", config=manifest)
        summary["wandb_url"] = run.url
        summary["training_wandb_urls"] = {
            role: run.url.rsplit("/runs/", 1)[0] + "/runs/" + rows[role, 100]["checkpoint_manifest"]["wandb_run_id"]
            for role in ROLES
        }
        run.log({"fixed_eval/paired_table": wandb.Table(columns=list(table[0]), data=[list(r.values()) for r in table]),
                 "fixed_eval/gap_curve": wandb.Image(str(out / "fig_washout.png"))})
        for row in table:
            run.log({f"{row['split']}/{key}": value for key, value in row.items() if isinstance(value, (float, int))})
        run.summary.update(decision)
    write_new(out / "washout_summary.json", summary)
    report = ["# Experiment 5: fixed-validation washout", "", decision["interpretation"], "", decision["caveat"], "",
              "All NLLs below use the same 512 confirmation sequences. Negative differences favor A.", "",
              "| Added steps | A | B | C | A−B | A−B 95% CI | A−C |", "|---:|---:|---:|---:|---:|:---|---:|"]
    for row in table:
        if row["split"] == "confirmation":
            report.append(f"| {row['offset']} | {row['A_nll']:.6f} | {row['B_nll']:.6f} | {row['C_nll']:.6f} | {row['A_minus_B']:+.6f} | [{row['AB_ci95_low']:+.6f}, {row['AB_ci95_high']:+.6f}] | {row['A_minus_C']:+.6f} |")
    report.extend(["", "A/B use 15 groups; C uses 16. All use eager routing; parent training used fused routing.",
                   "Snapshots were evaluated in separate processes after uninterrupted training to step1600.",
                   "Original optimizer/scheduler/RNG/data position and the 20,000-step LR schedule were restored.",
                   "No Experiment 3 gate or accepted earlier result was changed.", ""])
    if args.wandb:
        report.append(f"[W&B fixed-evaluation analysis]({summary['wandb_url']})")
        report.append("")
        for role, url in summary["training_wandb_urls"].items():
            report.append(f"- [{role} training]({url})")
    (out / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if args.wandb:
        artifact = wandb.Artifact("exp5-seed43-fixed-eval-washout", type="experiment5-results")
        artifact.add_dir(str(out), name="analysis")
        artifact.add_dir(str(Path(args.root) / "measurements"), name="measurements")
        artifact.add_file(args.manifest, name="branch_manifest.json")
        artifact.add_file(str(Path(args.root) / "step0_gate.json"), name="step0_gate.json")
        run.log_artifact(artifact).wait()
        run.finish()
        write_new(out / "WANDB_UPLOAD_COMPLETE.json", {"wandb_url": summary["wandb_url"], "summary_sha256": sha256_file(out / "washout_summary.json")})
    print(json.dumps(summary["decision"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("prepare", prepare), ("evaluate", evaluate), ("gate", step0_gate), ("analyze", analyze)):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.set_defaults(func=fn)
        if name == "prepare":
            p.add_argument("--parent", required=True); p.add_argument("--artifact", required=True)
        else:
            p.add_argument("--root", required=True)
        if name == "evaluate":
            p.add_argument("--role", choices=ROLES, required=True); p.add_argument("--offset", type=int, required=True)
        if name == "analyze":
            p.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
