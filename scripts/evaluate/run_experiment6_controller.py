#!/usr/bin/env python3
"""Success-only Experiment 6 evaluation, backup, and shutdown controller."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


VARIANTS = ("b", "c4", "g4", "c8", "g8")
GPU_BY_VARIANT = {variant: index for index, variant in enumerate(VARIANTS)}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def screen_alive(name: str) -> bool:
    result = subprocess.run(
        ["screen", "-ls"], text=True, capture_output=True, check=False)
    return f".{name}\t" in result.stdout


def checkpoint_complete(path: Path, variant: str, training_commit: str) -> bool:
    manifest_path = path / "training_manifest.json"
    if not manifest_path.is_file() or not (path / "training_state.pt").is_file():
        return False
    if not (list(path.glob("model*.safetensors")) or (path / "pytorch_model.bin").is_file()):
        return False
    manifest = read_json(manifest_path)
    identity = manifest.get("run_identity", {})
    expected_groups = {"b": None, "c4": 4, "g4": 4, "c8": 8, "g8": 8}[variant]
    return (
        manifest.get("global_step") == 2000
        and identity.get("seed") == 42
        and identity.get("steps") == 20000
        and identity.get("global_batch_size") == 32
        and identity.get("experiment6_variant") == variant
        and identity.get("experiment6_qkv_groups") == expected_groups
        and identity.get("source_commit") == training_commit
    )


def fail(status_root: Path, message: str) -> None:
    write_json(status_root / "FAILED.json", {
        "failed_at_unix": time.time(),
        "message": message,
    })
    raise RuntimeError(message)


def launch_evaluation(args, variant: str) -> None:
    name = f"mhar-exp6-eval-{variant}"
    if screen_alive(name):
        return
    log_dir = args.output_root / "evaluation-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(GPU_BY_VARIANT[variant]),
        "MHAR_REPO_DIR": str(args.repo),
        "MHAR_PYTHON_BIN": str(args.python),
        "MHAR_ROOT": str(args.output_root),
        "MHAR_CHECKPOINT": str(args.output_root / variant / "step-2000"),
        "MHAR_ARTIFACT": str(args.artifact),
        "MHAR_RESULTS_ROOT": str(args.output_root / "results"),
    })
    subprocess.run([
        "screen", "-L", "-Logfile", str(log_dir / f"{variant}.log"),
        "-dmS", name,
        str(args.repo / "scripts/evaluate/run_experiment6_screen.sh"), variant,
    ], cwd=args.repo, env=environment, check=True)


def wait_for_training(args, status_root: Path) -> None:
    while True:
        complete = []
        for variant in VARIANTS:
            checkpoint = args.output_root / variant / "step-2000"
            done = checkpoint_complete(checkpoint, variant, args.training_commit)
            alive = screen_alive(f"mhar-exp6-{variant}")
            complete.append(done and not alive)
            if not alive and not done:
                fail(status_root, f"training {variant} exited without a valid step-2000 checkpoint")
        if all(complete):
            write_json(status_root / "TRAINING_COMPLETE.json", {
                "completed_at_unix": time.time(),
                "variants": list(VARIANTS),
            })
            return
        time.sleep(args.poll_seconds)


def wait_for_evaluation(args, status_root: Path) -> None:
    for variant in VARIANTS:
        launch_evaluation(args, variant)
    while True:
        complete = []
        for variant in VARIANTS:
            result = args.output_root / "results" / variant / "result.json"
            done = result.is_file()
            alive = screen_alive(f"mhar-exp6-eval-{variant}")
            complete.append(done and not alive)
            if not alive and not done:
                fail(status_root, f"evaluation {variant} exited without result.json")
        if all(complete):
            write_json(status_root / "EVALUATION_COMPLETE.json", {
                "completed_at_unix": time.time(),
                "variants": list(VARIANTS),
            })
            return
        time.sleep(args.poll_seconds)


def run_analysis(args, status_root: Path) -> None:
    analysis = args.output_root / "analysis"
    subprocess.run([
        str(args.python), "-m", "src.experiments.experiment6_screening", "analyze",
        "--results-root", str(args.output_root / "results"),
        "--output-dir", str(analysis),
        "--wandb-mode", "online",
    ], cwd=args.repo, check=True)
    summary = analysis / "summary.json"
    if not summary.is_file():
        fail(status_root, "analysis exited without summary.json")
    payload = read_json(summary)
    if payload.get("decision") not in {"eligible_for_review", "reject_catastrophic_designs"}:
        fail(status_root, "analysis summary has no valid frozen screening decision")
    write_json(status_root / "ANALYSIS_COMPLETE.json", {
        "completed_at_unix": time.time(),
        "decision": payload["decision"],
        "wandb": payload.get("wandb"),
    })


def copy_results_and_push(args, status_root: Path) -> str:
    destination = args.repo / "results/experiment6/step-2000"
    if destination.exists():
        fail(status_root, f"refusing to overwrite existing result backup: {destination}")
    destination.mkdir(parents=True)
    for variant in VARIANTS:
        target = destination / variant
        target.mkdir()
        shutil.copy2(
            args.output_root / variant / "training_run_manifest.json",
            target / "training_run_manifest.json")
        shutil.copy2(
            args.output_root / variant / "step-2000/training_manifest.json",
            target / "checkpoint_training_manifest.json")
        shutil.copytree(
            args.output_root / "results" / variant,
            target / "evaluation")
    shutil.copytree(args.output_root / "analysis", destination / "analysis")
    subprocess.run(["git", "add", "results/experiment6/step-2000"], cwd=args.repo, check=True)
    subprocess.run([
        "git", "commit", "-m", "results: record Experiment 6 seed-42 screening"
    ], cwd=args.repo, check=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    subprocess.run([
        "git", "push", "origin", f"HEAD:{args.result_branch}"
    ], cwd=args.repo, check=True)
    remote = subprocess.check_output([
        "git", "ls-remote", "--heads", "origin", args.result_branch
    ], cwd=args.repo, text=True).split()[0]
    if remote != commit:
        fail(status_root, f"remote result commit mismatch: local={commit} remote={remote}")
    write_json(status_root / "BACKUP_PUSH_COMPLETE.json", {
        "completed_at_unix": time.time(),
        "commit": commit,
        "branch": args.result_branch,
    })
    return commit


def gpu_processes() -> list[str]:
    result = subprocess.run([
        "nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"
    ], text=True, capture_output=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--controller-commit", required=True)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--result-branch", default="codex/experiment-6-coupled-qkv")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--shutdown-grace-seconds", type=int, default=600)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.python = args.python.resolve()
    args.output_root = args.output_root.resolve()
    args.artifact = args.artifact.resolve()
    status_root = args.output_root / "controller"
    status_root.mkdir(parents=True, exist_ok=True)
    if (status_root / "FAILED.json").exists():
        raise RuntimeError("existing FAILED.json requires human review")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    if head != args.controller_commit:
        fail(status_root, f"controller source commit mismatch: {head}")
    if subprocess.check_output(
            ["sha256sum", str(args.artifact)], text=True).split()[0] != (
            "29e545dd9399e9eaea6f5abf38f20ef76ba232466a01b6ddf13c3a6a287a3691"):
        fail(status_root, "fixed evaluation artifact hash mismatch")
    wait_for_training(args, status_root)
    wait_for_evaluation(args, status_root)
    run_analysis(args, status_root)
    commit = copy_results_and_push(args, status_root)
    write_json(status_root / "SHUTDOWN_SCHEDULED.json", {
        "scheduled_at_unix": time.time(),
        "grace_seconds": args.shutdown_grace_seconds,
        "result_commit": commit,
    })
    time.sleep(args.shutdown_grace_seconds)
    processes = gpu_processes()
    if processes:
        fail(status_root, f"refusing shutdown while GPU processes remain: {processes}")
    write_json(status_root / "SHUTDOWN_STARTED.json", {"started_at_unix": time.time()})
    subprocess.run(["/bin/bash", "/usr/bin/shutdown"], check=True)


if __name__ == "__main__":
    main()
