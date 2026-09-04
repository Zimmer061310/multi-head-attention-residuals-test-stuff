#!/usr/bin/env python3
"""Fail-closed two-GPU controller for frozen Experiment 10."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.experiment10_per_group_contribution import (
    condition_manifest,
    phase_10abc_conditions,
    phase_10d_conditions,
)


CHECKPOINT_SHA256 = "74cff0ab19409dac9f6104e8986e4890c0837bc730d8ac233ae18011dbc58333"
ARTIFACT_SHA256 = "29e545dd9399e9eaea6f5abf38f20ef76ba232466a01b6ddf13c3a6a287a3691"


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
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
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


def checkpoint_complete(checkpoint: Path) -> bool:
    manifest_path = checkpoint / "training_manifest.json"
    if not manifest_path.is_file() or not (checkpoint / "training_state.pt").is_file():
        return False
    if not (list(checkpoint.glob("model*.safetensors")) or (checkpoint / "pytorch_model.bin").is_file()):
        return False
    manifest = read(manifest_path)
    identity = manifest.get("run_identity", {})
    return (
        manifest.get("global_step") == 2000
        and identity.get("seed") == 42
        and identity.get("mode") == "full_mh"
        and identity.get("attnres_heads") == 8
        and identity.get("num_heads") == 16
        and identity.get("num_kv_heads") == 8
        and identity.get("experiment8_variant") == "hq8"
        and identity.get("experiment8_hybrid_q_groups") == 8
        and identity.get("experiment8_local_head_position") == "even"
        and identity.get("experiment8_global_head_position") == "odd"
    )


def phase_conditions(phase: str) -> list[dict]:
    return phase_10abc_conditions() if phase == "phase-10abc" else phase_10d_conditions()


def launch_phase(args, phase: str, status: Path) -> None:
    logs = args.output_root / "worker-logs"
    logs.mkdir(parents=True, exist_ok=True)
    for worker in range(2):
        name = f"mhar-exp10-{phase}-gpu{worker}"
        if alive(name):
            continue
        marker = args.output_root / phase / f"WORKER_{worker}_COMPLETE.json"
        if marker.is_file():
            continue
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(worker),
            "MHAR_REPO_DIR": str(args.repo),
            "MHAR_PYTHON_BIN": str(args.python),
            "MHAR_EXP10_ROOT": str(args.output_root),
            "MHAR_EXP10_CHECKPOINT": str(args.checkpoint),
            "MHAR_EXP10_ARTIFACT": str(args.artifact),
        })
        subprocess.run(
            [
                "screen", "-L", "-Logfile", str(logs / f"{phase}-gpu{worker}.log"),
                "-dmS", name,
                str(args.repo / "scripts/evaluate/run_experiment10_worker.sh"),
                phase, str(worker),
            ],
            cwd=args.repo,
            env=environment,
            check=True,
        )
    expected = phase_conditions(phase)
    while True:
        missing = [
            row["id"] for row in expected
            if not (args.output_root / phase / "conditions" / row["id"] / "result.json").is_file()
        ]
        active = [alive(f"mhar-exp10-{phase}-gpu{worker}") for worker in range(2)]
        markers = [
            (args.output_root / phase / f"WORKER_{worker}_COMPLETE.json").is_file()
            for worker in range(2)
        ]
        if not missing and all(markers) and not any(active):
            write(status / f"{phase.upper()}_EVALUATION_COMPLETE.json", {
                "completed_at_unix": time.time(),
                "conditions": len(expected),
            })
            return
        if missing and not any(active):
            fail(status, f"{phase} workers exited with {len(missing)} missing conditions")
        time.sleep(args.poll_seconds)


def run_analysis(args, phase: str) -> Path:
    output = args.output_root / "analysis" / phase
    command = [
        str(args.python), "-m", "src.experiments.experiment10_per_group_contribution",
        f"analyze-{phase.removeprefix('phase-')}",
        "--results-root", str(args.output_root),
        "--output-dir", str(output),
        "--wandb-mode", "online",
        "--wandb-group", "mhar-exp10-per-group-contribution-seed42-step2000",
        "--wandb-run-name", f"mhar-exp10-{phase}-analysis-seed42",
    ]
    if phase == "phase-10d":
        command.extend([
            "--phase-10abc-summary",
            str(args.output_root / "analysis/phase-10abc/summary.json"),
        ])
    subprocess.run(command, cwd=args.repo, check=True)
    summary = output / "summary.json"
    if not summary.is_file():
        raise RuntimeError(f"{phase} analysis did not write summary.json")
    return summary


def finalize(args, primary: Path, alignment: Path | None) -> Path:
    final = args.output_root / "analysis" / "final"
    command = [
        str(args.python), "-m", "src.experiments.experiment10_per_group_contribution",
        "finalize", "--phase-10abc-summary", str(primary), "--output-dir", str(final),
    ]
    if alignment is not None:
        command.extend(["--phase-10d-summary", str(alignment)])
    subprocess.run(command, cwd=args.repo, check=True)
    subprocess.run(
        [
            str(args.python),
            str(args.repo / "figures/gen_fig_experiment10_group_contribution.py"),
            "--analysis-root", str(args.output_root / "analysis"),
            "--output-dir", str(final),
        ],
        cwd=args.repo,
        check=True,
        env={**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": "/tmp/mhar-exp10-mpl"},
    )
    for required in (
        "FINAL_REPORT.md", "final_summary.json", "fig_group_contribution.pdf",
        "fig_group_contribution.png", "fig_local_distribution.pdf",
        "fig_local_distribution.png",
    ):
        if not (final / required).is_file():
            raise RuntimeError(f"finalization is missing {required}")
    if alignment is not None:
        for required in ("fig_group_alignment.pdf", "fig_group_alignment.png"):
            if not (final / required).is_file():
                raise RuntimeError(f"finalization is missing {required}")
    return final


def backup_push(args, status: Path, final: Path) -> str:
    destination = args.repo / "results/experiment10"
    if destination.exists():
        fail(status, f"refusing to overwrite {destination}")
    destination.mkdir(parents=True)
    shutil.copy2(args.output_root / "condition_manifest.json", destination / "condition_manifest.json")
    shutil.copytree(args.output_root / "phase-10abc", destination / "phase-10abc")
    if (args.output_root / "phase-10d").exists():
        shutil.copytree(args.output_root / "phase-10d", destination / "phase-10d")
    shutil.copytree(args.output_root / "analysis", destination / "analysis")
    shutil.copy2(final / "FINAL_REPORT.md", destination / "FINAL_REPORT.md")
    for figure in final.glob("fig_*.pdf"):
        shutil.copy2(figure, destination / figure.name)
    for figure in final.glob("fig_*.png"):
        shutil.copy2(figure, destination / figure.name)
    for source, target in (
        (args.output_root / "phase-10abc", destination / "phase-10abc"),
        (args.output_root / "analysis", destination / "analysis"),
    ):
        if tree_sha256(source) != tree_sha256(target):
            fail(status, f"backup hash mismatch for {source.name}")
    subprocess.run(["git", "add", "results/experiment10"], cwd=args.repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "results: record Experiment 10 group contributions"],
        cwd=args.repo,
        check=True,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    subprocess.run(["git", "push", "origin", f"HEAD:{args.result_branch}"], cwd=args.repo, check=True)
    remote = subprocess.check_output(
        ["git", "ls-remote", "--heads", "origin", args.result_branch],
        cwd=args.repo,
        text=True,
    ).split()[0]
    if remote != commit:
        fail(status, "remote Experiment 10 result commit mismatch")
    write(status / "BACKUP_PUSH_COMPLETE.json", {
        "completed_at_unix": time.time(),
        "commit": commit,
        "final_summary_sha256": sha256(final / "final_summary.json"),
    })
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--controller-commit", required=True)
    parser.add_argument("--result-branch", default="codex/experiment-10-per-group-contribution")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--shutdown-grace-seconds", type=int, default=600)
    args = parser.parse_args()
    for name in ("repo", "output_root", "checkpoint", "artifact"):
        setattr(args, name, getattr(args, name).resolve())
    args.python = args.python.absolute()
    status = args.output_root / "controller"
    status.mkdir(parents=True, exist_ok=True)
    if (status / "FAILED.json").exists():
        raise RuntimeError("existing Experiment 10 FAILED.json requires review")
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
        if head != args.controller_commit:
            fail(status, f"controller commit mismatch: {head}")
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=args.repo, text=True).strip():
            fail(status, "controller checkout must be clean")
        if sha256(args.artifact) != ARTIFACT_SHA256:
            fail(status, "fixed evaluation artifact hash mismatch")
        if not checkpoint_complete(args.checkpoint):
            fail(status, "accepted HQ8 checkpoint is structurally incomplete")
        from src.experiments.experiment1_partition_compatibility import sha256_path

        if sha256_path(args.checkpoint) != CHECKPOINT_SHA256:
            fail(status, "accepted HQ8 checkpoint content hash mismatch")
        manifest_path = args.output_root / "condition_manifest.json"
        expected_manifest = condition_manifest()
        if manifest_path.exists() and read(manifest_path) != expected_manifest:
            fail(status, "existing condition manifest differs from reviewed manifest")
        if not manifest_path.exists():
            write(manifest_path, expected_manifest)
        launch_phase(args, "phase-10abc", status)
        primary = run_analysis(args, "phase-10abc")
        primary_payload = read(primary)
        alignment = None
        if primary_payload["phase_10d_gate"]["passed"]:
            write(status / "PHASE_10D_AUTHORIZED.json", {"authorized_at_unix": time.time()})
            launch_phase(args, "phase-10d", status)
            alignment = run_analysis(args, "phase-10d")
        else:
            write(status / "PHASE_10D_SKIPPED.json", {
                "skipped_at_unix": time.time(),
                "reason": "no local group passed the frozen usefulness rule",
            })
        final = finalize(args, primary, alignment)
        commit = backup_push(args, status, final)
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
    except Exception as error:
        if not (status / "FAILED.json").exists():
            write(status / "FAILED.json", {"failed_at_unix": time.time(), "message": str(error)})
        raise


if __name__ == "__main__":
    main()
