#!/usr/bin/env python3
"""Fail-closed three-GPU Experiment 7 evaluation, backup, and shutdown controller."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

VARIANTS = ("lq4", "lq8", "blq4", "blq8")
GPU = {"lq4": "0", "lq8": "1", "blq4": "2", "blq8": "2"}


def read(path): return json.loads(path.read_text(encoding="utf-8"))


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def alive(name):
    result = subprocess.run(["screen", "-ls"], text=True, capture_output=True, check=False)
    return f".{name}\t" in result.stdout


def fail(status, message):
    write(status / "FAILED.json", {"failed_at_unix": time.time(), "message": message})
    raise RuntimeError(message)


def checkpoint_complete(root, variant, training_commit):
    checkpoint = root / variant / "step-2000"
    path = checkpoint / "training_manifest.json"
    if not path.is_file() or not (checkpoint / "training_state.pt").is_file(): return False
    if not (list(checkpoint.glob("model*.safetensors")) or (checkpoint / "pytorch_model.bin").is_file()): return False
    manifest = read(path); identity = manifest.get("run_identity", {})
    return (manifest.get("global_step") == 2000 and identity.get("seed") == 42
            and identity.get("steps") == 20000 and identity.get("global_batch_size") == 32
            and identity.get("experiment7_variant") == variant
            and identity.get("experiment7_local_q_groups") == (4 if variant.endswith("4") else 8)
            and identity.get("source_commit") == training_commit)


def wait_training(args, status):
    while True:
        complete = {variant: checkpoint_complete(args.output_root, variant, args.training_commit)
                    for variant in VARIANTS}
        if all(complete.values()):
            if any(alive(name) for name in ("mhar-exp7-lq4", "mhar-exp7-lq8", "mhar-exp7-blq-queue")):
                time.sleep(args.poll_seconds); continue
            write(status / "TRAINING_COMPLETE.json", {"completed_at_unix": time.time(), "variants": list(VARIANTS)})
            return
        if not complete["lq4"] and not alive("mhar-exp7-lq4"): fail(status, "LQ4 exited without valid checkpoint")
        if not complete["lq8"] and not alive("mhar-exp7-lq8"): fail(status, "LQ8 exited without valid checkpoint")
        if (not complete["blq4"] or not complete["blq8"]) and not alive("mhar-exp7-blq-queue"):
            fail(status, "baseline local-Q queue exited before both checkpoints were valid")
        time.sleep(args.poll_seconds)


def launch_eval(args, variant):
    name = f"mhar-exp7-eval-{variant}"
    if alive(name): return
    environment = os.environ.copy(); environment.update({
        "CUDA_VISIBLE_DEVICES": GPU[variant], "MHAR_REPO_DIR": str(args.repo),
        "MHAR_PYTHON_BIN": str(args.python), "MHAR_ROOT": str(args.output_root),
        "MHAR_ARTIFACT": str(args.artifact),
    })
    log = args.output_root / "evaluation-logs"; log.mkdir(parents=True, exist_ok=True)
    subprocess.run(["screen", "-L", "-Logfile", str(log / f"{variant}.log"), "-dmS", name,
                    str(args.repo / "scripts/evaluate/run_experiment7_screen.sh"), variant],
                   cwd=args.repo, env=environment, check=True)


def wait_one_eval(args, status, variant):
    result = args.output_root / "results" / variant / "result.json"
    if result.is_file() and not alive(f"mhar-exp7-eval-{variant}"):
        return
    launch_eval(args, variant)
    while alive(f"mhar-exp7-eval-{variant}") or not result.is_file():
        if not alive(f"mhar-exp7-eval-{variant}") and not result.is_file():
            fail(status, f"evaluation {variant} exited without result.json")
        time.sleep(args.poll_seconds)


def evaluate(args, status):
    launch_eval(args, "lq4"); launch_eval(args, "lq8")
    wait_one_eval(args, status, "blq4"); wait_one_eval(args, status, "blq8")
    for variant in ("lq4", "lq8"):
        result = args.output_root / "results" / variant / "result.json"
        while alive(f"mhar-exp7-eval-{variant}") or not result.is_file():
            if not alive(f"mhar-exp7-eval-{variant}") and not result.is_file():
                fail(status, f"evaluation {variant} exited without result.json")
            time.sleep(args.poll_seconds)
    write(status / "EVALUATION_COMPLETE.json", {"completed_at_unix": time.time()})


def analyze_backup_push(args, status):
    analysis = args.output_root / "analysis"
    subprocess.run([str(args.python), "-m", "src.experiments.experiment7_screening", "analyze",
                    "--results-root", str(args.output_root / "results"), "--output-dir", str(analysis),
                    "--wandb-mode", "online"], cwd=args.repo, check=True)
    summary = analysis / "summary.json"
    if not summary.is_file(): fail(status, "analysis missing summary.json")
    decision = read(summary).get("decision")
    if decision not in {"eligible_for_review", "reject_catastrophic_designs"}: fail(status, "invalid decision")
    destination = args.repo / "results/experiment7/step-2000"
    if destination.exists(): fail(status, f"refusing to overwrite {destination}")
    destination.mkdir(parents=True)
    for variant in VARIANTS:
        target = destination / variant; target.mkdir()
        shutil.copy2(args.output_root / variant / "training_run_manifest.json", target / "training_run_manifest.json")
        shutil.copy2(args.output_root / variant / "step-2000/training_manifest.json", target / "checkpoint_training_manifest.json")
        shutil.copytree(args.output_root / "results" / variant, target / "evaluation")
    shutil.copytree(analysis, destination / "analysis")
    subprocess.run(["git", "add", "results/experiment7/step-2000"], cwd=args.repo, check=True)
    subprocess.run(["git", "commit", "-m", "results: record Experiment 7 seed-42 screening"], cwd=args.repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    subprocess.run(["git", "push", "origin", f"HEAD:{args.result_branch}"], cwd=args.repo, check=True)
    remote = subprocess.check_output(["git", "ls-remote", "--heads", "origin", args.result_branch], cwd=args.repo, text=True).split()[0]
    if remote != commit: fail(status, "remote backup commit mismatch")
    write(status / "BACKUP_PUSH_COMPLETE.json", {"completed_at_unix": time.time(), "commit": commit, "decision": decision})
    return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True); parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True); parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--controller-commit", required=True); parser.add_argument("--training-commit", required=True)
    parser.add_argument("--result-branch", default="codex/experiment-7-local-q-global-kv")
    parser.add_argument("--poll-seconds", type=int, default=60); parser.add_argument("--shutdown-grace-seconds", type=int, default=600)
    args = parser.parse_args()
    args.repo = args.repo.resolve(); args.python = args.python.absolute()
    args.output_root = args.output_root.resolve(); args.artifact = args.artifact.resolve()
    status = args.output_root / "controller"; status.mkdir(parents=True, exist_ok=True)
    if (status / "FAILED.json").exists(): raise RuntimeError("existing FAILED.json requires review")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    if head != args.controller_commit: fail(status, f"controller commit mismatch: {head}")
    digest = subprocess.check_output(["sha256sum", str(args.artifact)], text=True).split()[0]
    if digest != "29e545dd9399e9eaea6f5abf38f20ef76ba232466a01b6ddf13c3a6a287a3691": fail(status, "artifact hash mismatch")
    wait_training(args, status); evaluate(args, status); commit = analyze_backup_push(args, status)
    write(status / "SHUTDOWN_SCHEDULED.json", {"scheduled_at_unix": time.time(), "grace_seconds": args.shutdown_grace_seconds, "result_commit": commit})
    time.sleep(args.shutdown_grace_seconds)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True).strip()
    if processes: fail(status, f"refusing shutdown while GPU processes remain: {processes}")
    write(status / "SHUTDOWN_STARTED.json", {"started_at_unix": time.time()})
    subprocess.run(["/bin/bash", "/usr/bin/shutdown"], check=True)


if __name__ == "__main__": main()
