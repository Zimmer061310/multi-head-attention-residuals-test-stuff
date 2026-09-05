#!/usr/bin/env python3
"""Fail-closed multi-GPU controller for Experiment 11 screening."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


RUNS = (
    "s2q8-l000", "s2q8-l010", "s2q8-l025", "s2q8-l050",
    "gslq8-l000", "gslq8-l010", "gslq8-l025", "gslq8-l050",
    "m8-l100",
)
MILESTONES = (500, 1000, 1500, 2000)
ARTIFACT_SHA256 = "29e545dd9399e9eaea6f5abf38f20ef76ba232466a01b6ddf13c3a6a287a3691"


@dataclass
class Job:
    name: str
    command: list[str]
    expected: Path
    log: Path
    environment: dict[str, str]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256(item).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def alive(name: str) -> bool:
    result = subprocess.run(["screen", "-ls"], text=True, capture_output=True, check=False)
    return f".{name}\t" in result.stdout


def fail(status: Path, message: str) -> None:
    write(status / "FAILED.json", {"failed_at_unix": time.time(), "message": message})
    raise RuntimeError(message)


def checkpoint_complete(checkpoint: Path, run_id: str, milestone: int) -> bool:
    manifest_path = checkpoint / "training_manifest.json"
    if not manifest_path.is_file() or not (checkpoint / "training_state.pt").is_file():
        return False
    if not (list(checkpoint.glob("model*.safetensors")) or (checkpoint / "pytorch_model.bin").is_file()):
        return False
    manifest = read(manifest_path)
    identity = manifest.get("run_identity", {})
    return (
        manifest.get("global_step") == milestone
        and manifest.get("chunks_consumed") == milestone * 32
        and identity.get("seed") == 42
        and identity.get("mode") == "full_mh"
        and identity.get("attnres_heads") == 8
        and identity.get("num_heads") == 16
        and identity.get("num_kv_heads") == 8
        and identity.get("experiment11_run_id") == run_id
        and identity.get("experiment11_soft_q_groups") == 8
    )


def launch(job: Job, gpu: str, repo: Path) -> None:
    job.log.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, **job.environment, "CUDA_VISIBLE_DEVICES": gpu}
    subprocess.run(
        ["screen", "-L", "-Logfile", str(job.log), "-dmS", job.name, *job.command],
        cwd=repo,
        env=environment,
        check=True,
    )


def run_pool(args, status: Path, jobs: list[Job], phase: str) -> None:
    pending = [job for job in jobs if not job.expected.exists()]
    if not pending:
        write(status / f"{phase}_COMPLETE.json", {"completed_at_unix": time.time(), "jobs": len(jobs)})
        return
    if any(alive(job.name) for job in jobs):
        fail(status, f"{phase}: pre-existing worker screen requires review")
    free = list(args.gpu_ids)
    active: dict[str, Job] = {}
    while pending or active:
        while pending and free:
            gpu = free.pop(0)
            job = pending.pop(0)
            launch(job, gpu, args.repo)
            active[gpu] = job
        for gpu, job in tuple(active.items()):
            if not alive(job.name):
                if not job.expected.exists():
                    fail(status, f"{phase}: {job.name} exited without {job.expected}")
                del active[gpu]
                free.append(gpu)
                free.sort(key=args.gpu_ids.index)
        if pending or active:
            time.sleep(args.poll_seconds)
    write(status / f"{phase}_COMPLETE.json", {"completed_at_unix": time.time(), "jobs": len(jobs)})


def base_environment(args, index: int) -> dict[str, str]:
    return {
        "MHAR_REPO_DIR": str(args.repo),
        "MHAR_PYTHON_BIN": str(args.python),
        "MHAR_EXP11_ROOT": str(args.output_root),
        "MHAR_EXP11_ARTIFACT": str(args.artifact),
        "MHAR_DATA_FILES": args.data_files,
        "MHAR_MASTER_PORT": str(args.master_port_base + index),
    }


def training_jobs(args, milestone: int) -> list[Job]:
    previous = None if milestone == MILESTONES[0] else MILESTONES[MILESTONES.index(milestone) - 1]
    jobs = []
    for index, run_id in enumerate(RUNS):
        checkpoint = args.output_root / "training" / run_id / f"step-{milestone}"
        command = [
            str(args.repo / "scripts/train/run_experiment11_screen.sh"),
            run_id, str(milestone),
        ]
        if previous is not None:
            source = args.output_root / "training" / run_id / f"step-{previous}"
            if not checkpoint.exists() and not checkpoint_complete(source, run_id, previous):
                raise RuntimeError(f"missing valid resume checkpoint for {run_id}: {source}")
            command.append(str(source))
        jobs.append(Job(
            name=f"mhar-exp11-train-{run_id}-to{milestone}",
            command=command,
            expected=checkpoint / "training_manifest.json",
            log=args.output_root / "logs" / f"train-{run_id}-to{milestone}.log",
            environment=base_environment(args, index),
        ))
    return jobs


def probe_jobs(args, milestone: int, run_ids=RUNS, split="discovery", selection=None) -> list[Job]:
    jobs = []
    for index, run_id in enumerate(run_ids):
        output = args.output_root / "probes" / run_id / f"step-{milestone}-{split}.json"
        command = [
            str(args.python), "-m", "src.experiments.experiment11_workflow", "probe",
            "--run-id", run_id, "--milestone", str(milestone), "--split", split,
            "--checkpoint", str(args.output_root / "training" / run_id / f"step-{milestone}"),
            "--artifact", str(args.artifact), "--output", str(output),
            "--results-root", str(args.output_root), "--device", "cuda", "--dtype", "bf16",
        ]
        if selection is not None:
            command.extend(["--selection-manifest", str(selection)])
        jobs.append(Job(
            name=f"mhar-exp11-probe-{run_id}-{milestone}-{split}",
            command=command,
            expected=output,
            log=args.output_root / "logs" / f"probe-{run_id}-{milestone}-{split}.log",
            environment={},
        ))
    return jobs


def evaluation_jobs(args, split: str, selection=None) -> list[Job]:
    jobs = []
    for run_id in RUNS:
        output = args.output_root / "evaluations" / run_id / f"{split}.json"
        command = [
            str(args.python), "-m", "src.experiments.experiment11_workflow", "evaluate",
            "--run-id", run_id, "--split", split,
            "--checkpoint", str(args.output_root / "training" / run_id / "step-2000"),
            "--artifact", str(args.artifact), "--output", str(output),
            "--results-root", str(args.output_root), "--device", "cuda", "--dtype", "bf16",
            "--batch-size", str(args.evaluation_batch_size),
        ]
        if selection is not None:
            command.extend(["--selection-manifest", str(selection)])
        jobs.append(Job(
            name=f"mhar-exp11-eval-{run_id}-{split}",
            command=command,
            expected=output,
            log=args.output_root / "logs" / f"eval-{run_id}-{split}.log",
            environment={},
        ))
    return jobs


def validate_all_checkpoints(args, status: Path) -> None:
    invalid = [
        run_id for run_id in RUNS
        if not checkpoint_complete(args.output_root / "training" / run_id / "step-2000", run_id, 2000)
    ]
    if invalid:
        fail(status, f"invalid final checkpoints: {invalid}")
    write(status / "TRAINING_COMPLETE.json", {"completed_at_unix": time.time(), "runs": list(RUNS)})


def analyze_and_figure(args, status: Path, selection: Path) -> Path:
    analysis = args.output_root / "analysis"
    subprocess.run([
        str(args.python), "-m", "src.experiments.experiment11_workflow", "analyze",
        "--results-root", str(args.output_root), "--selection-manifest", str(selection),
        "--output-dir", str(analysis), "--wandb-mode", args.wandb_mode,
    ], cwd=args.repo, check=True)
    subprocess.run([
        str(args.python), str(args.repo / "figures/gen_fig_experiment11_soft_specialization.py"),
        "--summary", str(analysis / "summary.json"),
        "--results-root", str(args.output_root), "--output-dir", str(analysis),
    ], cwd=args.repo, check=True, env={
        **os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": "/tmp/mhar-exp11-mpl"
    })
    required = [
        "summary.json", "FINAL_REPORT.md", "nll_curves.csv",
        "softness_trajectories.csv", "weight_trajectories.csv", "systems_table.csv",
    ]
    required += [
        f"fig_{stem}.{suffix}"
        for stem in ("nll_curves", "selected_contrasts", "softness_curves",
                     "softness_trajectories", "s2q8_heatmap", "gslq8_heatmap")
        for suffix in ("png", "pdf")
    ]
    missing = [name for name in required if not (analysis / name).is_file()]
    if missing:
        fail(status, f"analysis missing required outputs: {missing}")
    write(status / "ANALYSIS_COMPLETE.json", {
        "completed_at_unix": time.time(), "summary_sha256": sha256(analysis / "summary.json")
    })
    return analysis


def backup_push(args, status: Path, analysis: Path, selection: Path) -> str:
    destination = args.repo / "results/experiment11"
    if destination.exists():
        fail(status, f"refusing to overwrite existing result directory: {destination}")
    destination.mkdir(parents=True)
    for name in ("evaluations", "probes"):
        shutil.copytree(args.output_root / name, destination / name)
    shutil.copy2(selection, destination / "selection_manifest.json")
    shutil.copytree(analysis, destination / "analysis")
    shutil.copy2(analysis / "FINAL_REPORT.md", destination / "FINAL_REPORT.md")
    manifests = destination / "training_manifests"
    manifests.mkdir()
    for run_id in RUNS:
        target = manifests / run_id
        target.mkdir()
        for source, name in (
            (args.output_root / "training" / run_id / "training_run_manifest.json", "training_run_manifest.json"),
            (args.output_root / "training" / run_id / "step-2000/training_manifest.json", "checkpoint_training_manifest.json"),
            (args.output_root / "training" / run_id / "experiment11_weight_metrics.jsonl", "weight_metrics.jsonl"),
        ):
            shutil.copy2(source, target / name)
    for name in ("evaluations", "probes", "analysis"):
        if tree_sha256(args.output_root / name) != tree_sha256(destination / name):
            fail(status, f"backup tree hash mismatch: {name}")
    subprocess.run(["git", "add", "results/experiment11"], cwd=args.repo, check=True)
    subprocess.run([
        "git", "commit", "-m", "results: record Experiment 11 seed-42 screening"
    ], cwd=args.repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    subprocess.run(["git", "push", "origin", f"HEAD:{args.result_branch}"], cwd=args.repo, check=True)
    remote = subprocess.check_output(
        ["git", "ls-remote", "--heads", "origin", args.result_branch],
        cwd=args.repo, text=True,
    ).split()[0]
    if remote != commit:
        fail(status, "remote result commit mismatch")
    write(status / "BACKUP_PUSH_COMPLETE.json", {
        "completed_at_unix": time.time(), "commit": commit,
        "summary_sha256": sha256(analysis / "summary.json"),
    })
    return commit


def preflight(args, status: Path) -> None:
    if status.joinpath("FAILED.json").exists():
        raise RuntimeError("existing FAILED.json requires review")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    if head != args.controller_commit:
        fail(status, f"controller commit mismatch: expected {args.controller_commit}, got {head}")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=args.repo, text=True).strip()
    if dirty:
        fail(status, "controller checkout must be clean")
    if sha256(args.artifact) != ARTIFACT_SHA256:
        fail(status, "fixed evaluation artifact hash mismatch")
    if len(set(args.gpu_ids)) != len(args.gpu_ids) or not args.gpu_ids:
        fail(status, "GPU list must be non-empty and unique")
    observed = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
    ).split()
    if any(gpu not in observed for gpu in args.gpu_ids):
        fail(status, f"requested GPU absent; available={observed}")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        fail(status, f"assigned server must be GPU-idle before launch: {active}")
    usage = shutil.disk_usage(args.output_root.parent)
    if usage.free < args.minimum_free_bytes:
        fail(status, f"insufficient free disk: {usage.free} < {args.minimum_free_bytes}")
    freeze = subprocess.check_output([str(args.python), "-m", "pip", "freeze"], text=True)
    (status / "dependency-freeze.txt").write_text(freeze, encoding="utf-8")
    write(status / "PREFLIGHT_COMPLETE.json", {
        "completed_at_unix": time.time(), "controller_commit": head,
        "artifact_sha256": ARTIFACT_SHA256, "gpu_ids": args.gpu_ids,
        "free_disk_bytes": usage.free,
    })


def execute(args, status: Path) -> None:
    preflight(args, status)
    for milestone in MILESTONES:
        run_pool(args, status, training_jobs(args, milestone), f"TRAIN_TO_{milestone}")
        invalid = [
            run_id for run_id in RUNS
            if not checkpoint_complete(args.output_root / "training" / run_id / f"step-{milestone}", run_id, milestone)
        ]
        if invalid:
            fail(status, f"invalid milestone-{milestone} checkpoints: {invalid}")
        run_pool(args, status, probe_jobs(args, milestone), f"PROBE_{milestone}")
    validate_all_checkpoints(args, status)
    run_pool(args, status, evaluation_jobs(args, "discovery"), "DISCOVERY_EVALUATION")
    selection = args.output_root / "selection_manifest.json"
    subprocess.run([
        str(args.python), "-m", "src.experiments.experiment11_workflow", "select",
        "--results-root", str(args.output_root), "--output", str(selection),
    ], cwd=args.repo, check=True)
    selected = read(selection)["selected"]
    run_pool(args, status, evaluation_jobs(args, "confirmation", selection), "CONFIRMATION_EVALUATION")
    confirmation_probes = [selected["s2q8"]["run_id"], selected["gslq8"]["run_id"], "m8-l100"]
    run_pool(
        args, status,
        probe_jobs(args, 2000, confirmation_probes, "confirmation", selection),
        "CONFIRMATION_PROBES",
    )
    analysis = analyze_and_figure(args, status, selection)
    commit = backup_push(args, status, analysis, selection)
    write(status / "WORKFLOW_COMPLETE.json", {
        "completed_at_unix": time.time(), "result_commit": commit,
        "shutdown_authorized": args.shutdown_on_success,
    })
    if not args.shutdown_on_success:
        return
    write(status / "SHUTDOWN_SCHEDULED.json", {
        "scheduled_at_unix": time.time(), "grace_seconds": args.shutdown_grace_seconds,
    })
    time.sleep(args.shutdown_grace_seconds)
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        text=True,
    ).strip()
    if active:
        fail(status, f"refusing shutdown while GPU processes remain: {active}")
    write(status / "SHUTDOWN_STARTED.json", {"started_at_unix": time.time()})
    subprocess.run(["/bin/bash", "/usr/bin/shutdown"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--controller-commit", required=True)
    parser.add_argument("--result-branch", default="codex/experiment-11-soft-specialization")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--data-files", default="/root/autodl-tmp/datasets/fineweb-edu-sample-10BT/train/*.parquet")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--master-port-base", type=int, default=30110)
    parser.add_argument("--evaluation-batch-size", type=int, default=1)
    parser.add_argument("--minimum-free-bytes", type=int, default=110_000_000_000)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--shutdown-on-success", action="store_true")
    parser.add_argument("--shutdown-grace-seconds", type=int, default=600)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.python = args.python.absolute()
    args.output_root = args.output_root.resolve()
    args.artifact = args.artifact.resolve()
    args.gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    status = args.output_root / "controller"
    status.mkdir(parents=True, exist_ok=True)
    try:
        execute(args, status)
    except Exception as error:
        if not (status / "FAILED.json").exists():
            write(status / "FAILED.json", {
                "failed_at_unix": time.time(),
                "error_type": type(error).__name__,
                "message": str(error),
            })
        raise


if __name__ == "__main__":
    main()
