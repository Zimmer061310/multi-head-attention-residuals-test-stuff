#!/usr/bin/env python3
"""Fail-closed two-GPU Experiment 8 evaluation, backup, and shutdown controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


VARIANTS = ("hq8", "bhq8")
GPU = {"hq8": "0", "bhq8": "1"}
ARTIFACT_SHA256 = "29e545dd9399e9eaea6f5abf38f20ef76ba232466a01b6ddf13c3a6a287a3691"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path):
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def alive(name):
    result = subprocess.run(
        ["screen", "-ls"], text=True, capture_output=True, check=False
    )
    return f".{name}\t" in result.stdout


def fail(status, message):
    write(status / "FAILED.json", {"failed_at_unix": time.time(), "message": message})
    raise RuntimeError(message)


def checkpoint_complete(root, variant, training_commit):
    checkpoint = root / variant / "step-2000"
    path = checkpoint / "training_manifest.json"
    if not path.is_file() or not (checkpoint / "training_state.pt").is_file():
        return False
    if not (
        list(checkpoint.glob("model*.safetensors"))
        or (checkpoint / "pytorch_model.bin").is_file()
    ):
        return False
    manifest = read(path)
    identity = manifest.get("run_identity", {})
    expected_mode = "full_mh" if variant == "hq8" else "baseline"
    return (
        manifest.get("global_step") == 2000
        and identity.get("seed") == 42
        and identity.get("steps") == 20000
        and identity.get("global_batch_size") == 32
        and identity.get("mode") == expected_mode
        and identity.get("attnres_heads") == 8
        and identity.get("num_heads") == 16
        and identity.get("num_kv_heads") == 8
        and identity.get("experiment8_variant") == variant
        and identity.get("experiment8_hybrid_q_groups") == 8
        and identity.get("experiment8_local_head_position") == "even"
        and identity.get("experiment8_global_head_position") == "odd"
        and identity.get("source_commit") == training_commit
    )


def wait_training(args, status):
    while True:
        complete = {
            variant: checkpoint_complete(args.output_root, variant, args.training_commit)
            for variant in VARIANTS
        }
        if all(complete.values()):
            if any(alive(f"mhar-exp8-{variant}") for variant in VARIANTS):
                time.sleep(args.poll_seconds)
                continue
            write(status / "TRAINING_COMPLETE.json", {
                "completed_at_unix": time.time(), "variants": list(VARIANTS),
            })
            return
        for variant in VARIANTS:
            if not complete[variant] and not alive(f"mhar-exp8-{variant}"):
                fail(status, f"{variant.upper()} exited without a valid step-2000 checkpoint")
        time.sleep(args.poll_seconds)


def launch_eval(args, variant):
    name = f"mhar-exp8-eval-{variant}"
    result = args.output_root / "results" / variant / "result.json"
    if result.is_file() and not alive(name):
        return
    if alive(name):
        return
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": GPU[variant],
        "MHAR_REPO_DIR": str(args.repo),
        "MHAR_PYTHON_BIN": str(args.python),
        "MHAR_ROOT": str(args.output_root),
        "MHAR_ARTIFACT": str(args.artifact),
    })
    log = args.output_root / "evaluation-logs"
    log.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "screen", "-L", "-Logfile", str(log / f"{variant}.log"),
            "-dmS", name,
            str(args.repo / "scripts/evaluate/run_experiment8_screen.sh"), variant,
        ],
        cwd=args.repo,
        env=environment,
        check=True,
    )


def evaluate(args, status):
    for variant in VARIANTS:
        launch_eval(args, variant)
    pending = set(VARIANTS)
    while pending:
        for variant in tuple(pending):
            name = f"mhar-exp8-eval-{variant}"
            result = args.output_root / "results" / variant / "result.json"
            if result.is_file() and not alive(name):
                pending.remove(variant)
            elif not alive(name) and not result.is_file():
                fail(status, f"evaluation {variant} exited without result.json")
        if pending:
            time.sleep(args.poll_seconds)
    write(status / "EVALUATION_COMPLETE.json", {"completed_at_unix": time.time()})


def analyze_backup_push(args, status):
    analysis = args.output_root / "analysis"
    subprocess.run(
        [
            str(args.python), "-m", "src.experiments.experiment8_screening", "analyze",
            "--results-root", str(args.output_root / "results"),
            "--output-dir", str(analysis), "--wandb-mode", "online",
        ],
        cwd=args.repo,
        check=True,
    )
    summary_path = analysis / "summary.json"
    report_path = analysis / "FINAL_REPORT.md"
    if not summary_path.is_file() or not report_path.is_file():
        fail(status, "analysis is missing summary.json or FINAL_REPORT.md")
    summary = read(summary_path)
    decision = summary.get("decision")
    if decision not in {"eligible_for_review", "reject_catastrophic_designs"}:
        fail(status, "invalid frozen decision")
    destination = args.repo / "results/experiment8/step-2000"
    if destination.exists():
        fail(status, f"refusing to overwrite {destination}")
    destination.mkdir(parents=True)
    for variant in VARIANTS:
        target = destination / variant
        target.mkdir()
        shutil.copy2(
            args.output_root / variant / "training_run_manifest.json",
            target / "training_run_manifest.json",
        )
        shutil.copy2(
            args.output_root / variant / "step-2000/training_manifest.json",
            target / "checkpoint_training_manifest.json",
        )
        shutil.copytree(args.output_root / "results" / variant, target / "evaluation")
        if tree_sha256(args.output_root / "results" / variant) != tree_sha256(
            target / "evaluation"
        ):
            fail(status, f"backup hash mismatch for {variant} evaluation")
    shutil.copytree(analysis, destination / "analysis")
    if tree_sha256(analysis) != tree_sha256(destination / "analysis"):
        fail(status, "backup hash mismatch for analysis")
    shutil.copy2(report_path, args.repo / "results/experiment8/FINAL_REPORT.md")
    if sha256(report_path) != sha256(args.repo / "results/experiment8/FINAL_REPORT.md"):
        fail(status, "backup hash mismatch for final report")
    subprocess.run(["git", "add", "results/experiment8"], cwd=args.repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "results: record Experiment 8 seed-42 screening"],
        cwd=args.repo,
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True
    ).strip()
    subprocess.run(
        ["git", "push", "origin", f"HEAD:{args.result_branch}"],
        cwd=args.repo,
        check=True,
    )
    remote = subprocess.check_output(
        ["git", "ls-remote", "--heads", "origin", args.result_branch],
        cwd=args.repo,
        text=True,
    ).split()[0]
    if remote != commit:
        fail(status, "remote backup commit mismatch")
    write(status / "BACKUP_PUSH_COMPLETE.json", {
        "completed_at_unix": time.time(),
        "commit": commit,
        "decision": decision,
        "summary_sha256": sha256(summary_path),
    })
    return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--controller-commit", required=True)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--result-branch", default="codex/experiment-8-hybrid-q")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--shutdown-grace-seconds", type=int, default=600)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.python = args.python.absolute()
    args.output_root = args.output_root.resolve()
    args.artifact = args.artifact.resolve()
    status = args.output_root / "controller"
    status.mkdir(parents=True, exist_ok=True)
    if (status / "FAILED.json").exists():
        raise RuntimeError("existing FAILED.json requires review")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True
    ).strip()
    if head != args.controller_commit:
        fail(status, f"controller commit mismatch: {head}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=args.repo, text=True
    ).strip()
    if dirty:
        fail(status, "controller checkout must be clean before launch")
    if sha256(args.artifact) != ARTIFACT_SHA256:
        fail(status, "artifact hash mismatch")
    wait_training(args, status)
    evaluate(args, status)
    commit = analyze_backup_push(args, status)
    write(status / "SHUTDOWN_SCHEDULED.json", {
        "scheduled_at_unix": time.time(),
        "grace_seconds": args.shutdown_grace_seconds,
        "result_commit": commit,
    })
    time.sleep(args.shutdown_grace_seconds)
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        text=True,
    ).strip()
    if processes:
        fail(status, f"refusing shutdown while GPU processes remain: {processes}")
    write(status / "SHUTDOWN_STARTED.json", {"started_at_unix": time.time()})
    subprocess.run(["/bin/bash", "/usr/bin/shutdown"], check=True)


if __name__ == "__main__":
    main()
