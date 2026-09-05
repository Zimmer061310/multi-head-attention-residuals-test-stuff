#!/usr/bin/env python3
"""Fail-closed rotating worker for one host of a split Experiment 11 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ALL_RUNS = (
    "s2q8-l000", "s2q8-l010", "s2q8-l025", "s2q8-l050",
    "gslq8-l000", "gslq8-l010", "gslq8-l025", "gslq8-l050",
    "m8-l100",
)
MILESTONES = (500, 1000, 1500, 2000)
ARTIFACT_SHA256 = "29e545dd9399e9eaea6f5abf38f20ef76ba232466a01b6ddf13c3a6a287a3691"


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def alive(name: str) -> bool:
    result = subprocess.run(["screen", "-ls"], text=True, capture_output=True, check=False)
    return f".{name}\t" in result.stdout


def checkpoint_complete(root: Path, run_id: str, step: int) -> bool:
    checkpoint = root / "training" / run_id / f"step-{step}"
    manifest_path = checkpoint / "training_manifest.json"
    if not manifest_path.is_file() or not (checkpoint / "training_state.pt").is_file():
        return False
    if not (list(checkpoint.glob("model*.safetensors")) or (checkpoint / "pytorch_model.bin").is_file()):
        return False
    manifest = read(manifest_path)
    identity = manifest.get("run_identity", {})
    return (
        manifest.get("global_step") == step
        and manifest.get("chunks_consumed") == step * 32
        and identity.get("seed") == 42
        and identity.get("mode") == "full_mh"
        and identity.get("attnres_heads") == 8
        and identity.get("num_heads") == 16
        and identity.get("num_kv_heads") == 8
        and identity.get("experiment11_run_id") == run_id
        and identity.get("experiment11_soft_q_groups") == 8
    )


def current_step(root: Path, run_id: str) -> int:
    run_root = root / "training" / run_id
    if not run_root.exists():
        return 0
    steps = []
    for path in run_root.glob("step-*"):
        try:
            step = int(path.name.split("-", 1)[1])
        except ValueError:
            continue
        if checkpoint_complete(root, run_id, step):
            steps.append(step)
    if len(steps) > 1:
        raise RuntimeError(f"rotation invariant failed: multiple checkpoints for {run_id}: {steps}")
    return steps[0] if steps else 0


def next_target(step: int, quantum: int) -> int:
    later_milestones = [value for value in MILESTONES if value > step]
    return min(2000, step + quantum, later_milestones[0])


@dataclass
class Task:
    run_id: str
    kind: str
    target: int
    screen: str
    expected: Path
    log: Path


def launch_training(args, run_id: str, target: int, gpu: str) -> Task:
    step = current_step(args.output_root, run_id)
    command = [str(args.repo / "scripts/train/run_experiment11_screen.sh"), run_id, str(target)]
    if step:
        command.append(str(args.output_root / "training" / run_id / f"step-{step}"))
    name = f"mhar-exp11-{args.worker_id}-{run_id}-to{target}"
    log = args.output_root / "logs" / f"{name}.log"
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpu,
        "MHAR_REPO_DIR": str(args.repo),
        "MHAR_PYTHON_BIN": str(args.python),
        "MHAR_EXP11_ROOT": str(args.output_root),
        "MHAR_EXP11_ARTIFACT": str(args.artifact),
        "MHAR_DATA_FILES": args.data_files,
        "MHAR_MASTER_PORT": str(args.master_port_base + int(gpu)),
        "MHAR_CHECKPOINT_LOCK": str(args.checkpoint_lock),
        "MHAR_EXP11_TRAIN_EVAL_EVERY": "0",
    }
    log.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["screen", "-L", "-Logfile", str(log), "-dmS", name, *command],
        cwd=args.repo, env=environment, check=True,
    )
    return Task(
        run_id=run_id, kind="train", target=target, screen=name,
        expected=args.output_root / "training" / run_id / f"step-{target}" / "training_manifest.json",
        log=log,
    )


def launch_probe(args, run_id: str, milestone: int, gpu: str) -> Task:
    output = args.output_root / "probes" / run_id / f"step-{milestone}-discovery.json"
    name = f"mhar-exp11-{args.worker_id}-{run_id}-probe{milestone}"
    log = args.output_root / "logs" / f"{name}.log"
    command = [
        str(args.python), "-m", "src.experiments.experiment11_workflow", "probe",
        "--run-id", run_id, "--milestone", str(milestone), "--split", "discovery",
        "--checkpoint", str(args.output_root / "training" / run_id / f"step-{milestone}"),
        "--artifact", str(args.artifact), "--output", str(output),
        "--results-root", str(args.output_root), "--device", "cuda", "--dtype", "bf16",
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["screen", "-L", "-Logfile", str(log), "-dmS", name, *command],
        cwd=args.repo, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, check=True,
    )
    return Task(run_id, "probe", milestone, name, output, log)


def preflight(args, status: Path) -> None:
    if (status / "FAILED.json").exists():
        raise RuntimeError("existing FAILED.json requires review")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    if head != args.controller_commit:
        raise RuntimeError(f"commit mismatch: {head} != {args.controller_commit}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=args.repo, text=True).strip():
        raise RuntimeError("worker checkout must be clean")
    if sha256(args.artifact) != ARTIFACT_SHA256:
        raise RuntimeError("fixed artifact hash mismatch")
    if len(set(args.run_ids)) != len(args.run_ids) or not set(args.run_ids) <= set(ALL_RUNS):
        raise RuntimeError("run IDs must be a unique Experiment 11 subset")
    if len(set(args.gpu_ids)) != len(args.gpu_ids) or not args.gpu_ids:
        raise RuntimeError("GPU IDs must be non-empty and unique")
    observed = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
    ).split()
    if any(gpu not in observed for gpu in args.gpu_ids):
        raise RuntimeError(f"requested GPU absent; available={observed}")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        raise RuntimeError(f"GPUs must be idle before launch: {active}")
    write(status / "PREFLIGHT_COMPLETE.json", {
        "completed_at_unix": time.time(), "commit": head,
        "artifact_sha256": ARTIFACT_SHA256, "run_ids": args.run_ids,
        "gpu_ids": args.gpu_ids, "quantum": args.quantum,
    })


def execute(args, status: Path) -> None:
    preflight(args, status)
    free = list(args.gpu_ids)
    active: dict[str, Task] = {}
    probe_pending: list[tuple[str, int]] = []
    while True:
        for gpu, task in tuple(active.items()):
            if alive(task.screen):
                continue
            if not task.expected.is_file():
                raise RuntimeError(f"{task.screen} exited without {task.expected}; see {task.log}")
            if task.kind == "train":
                if not checkpoint_complete(args.output_root, task.run_id, task.target):
                    raise RuntimeError(f"invalid checkpoint for {task.run_id} step {task.target}")
                if task.target in MILESTONES:
                    probe_pending.append((task.run_id, task.target))
            del active[gpu]
            free.append(gpu)

        free.sort(key=args.gpu_ids.index)
        while free and probe_pending:
            run_id, milestone = probe_pending.pop(0)
            gpu = free.pop(0)
            active[gpu] = launch_probe(args, run_id, milestone, gpu)

        incomplete = [run_id for run_id in args.run_ids if current_step(args.output_root, run_id) < 2000]
        busy_runs = {task.run_id for task in active.values()}
        while free:
            candidates = [run_id for run_id in incomplete if run_id not in busy_runs]
            if not candidates:
                break
            run_id = min(
                candidates,
                key=lambda value: (current_step(args.output_root, value), args.run_ids.index(value)),
            )
            step = current_step(args.output_root, run_id)
            gpu = free.pop(0)
            task = launch_training(args, run_id, next_target(step, args.quantum), gpu)
            active[gpu] = task
            busy_runs.add(run_id)

        if not incomplete and not active and not probe_pending:
            break
        time.sleep(args.poll_seconds)

    missing = [
        (run_id, milestone) for run_id in args.run_ids for milestone in MILESTONES
        if not (args.output_root / "probes" / run_id / f"step-{milestone}-discovery.json").is_file()
    ]
    if missing:
        raise RuntimeError(f"missing milestone probes: {missing}")
    write(status / "TRAINING_AND_PROBES_COMPLETE.json", {
        "completed_at_unix": time.time(), "run_ids": args.run_ids,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--controller-commit", required=True)
    parser.add_argument("--run-ids", required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--quantum", type=int, default=200)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--master-port-base", type=int, default=31100)
    parser.add_argument("--checkpoint-lock", type=Path, default=Path("/tmp/mhar-exp11-checkpoint.lock"))
    parser.add_argument(
        "--data-files",
        default="/root/autodl-tmp/datasets/fineweb-edu-sample-10BT/train/*.parquet",
    )
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.python = args.python.absolute()
    args.output_root = args.output_root.resolve()
    args.artifact = args.artifact.resolve()
    args.run_ids = [value.strip() for value in args.run_ids.split(",") if value.strip()]
    args.gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if args.quantum < 100 or args.quantum % 100:
        raise ValueError("quantum must be a positive multiple of 100")
    status = args.output_root / "controller" / args.worker_id
    status.mkdir(parents=True, exist_ok=True)
    try:
        execute(args, status)
    except Exception as error:
        write(status / "FAILED.json", {
            "failed_at_unix": time.time(), "error_type": type(error).__name__,
            "message": str(error),
        })
        raise


if __name__ == "__main__":
    main()
