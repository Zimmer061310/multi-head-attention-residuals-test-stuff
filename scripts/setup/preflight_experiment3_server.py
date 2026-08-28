#!/usr/bin/env python3
"""Fail fast before any Experiment 3 GPU process is allowed to start."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / "configs/environment/stage-b-server.json"
PROTOCOL_PATH = ROOT / "configs/experiment3/protocol.json"
ENVIRONMENT = json.loads(ENV_PATH.read_text(encoding="utf-8"))
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
OLD_ARTIFACT_HASHES = {
    "29e545dd9399e9eaea6f5abf38f20ef76ba232466a01b6ddf13c3a6a287a3691",
    "80276b413b27c2da2ed7bc3b1121536f9bc7763b3cc99c3f86e49a83e5705cb3",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def check_source_shard(directory: Path) -> None:
    spec = ENVIRONMENT["experiment3_evaluation_dataset"]["files"][0]
    path = directory / spec["name"]
    if not path.is_file() or path.stat().st_size != spec["bytes"]:
        raise RuntimeError(f"Experiment 3 source shard size check failed: {path}")
    observed = sha256_file(path)
    if observed != spec["sha256"]:
        raise RuntimeError(f"Experiment 3 source shard SHA-256 mismatch: {observed}")
    used_hashes = {
        row["sha256"]
        for key in ("dataset", "evaluation_dataset")
        for row in ENVIRONMENT[key]["files"]
    }
    if observed in used_hashes:
        raise RuntimeError("Experiment 3 source shard overlaps training or Experiment 2 eval")


def check_artifact(path: Path) -> str:
    sidecar = Path(str(path) + ".manifest.json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"artifact and sidecar are required: {path}")
    digest = sha256_file(path)
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    if manifest["artifact_sha256"] != digest:
        raise RuntimeError("Experiment 3 artifact sidecar hash mismatch")
    expected = PROTOCOL["evaluation"]
    if manifest["discovery_shape"] != [expected["discovery_sequences"], expected["sequence_length"]]:
        raise RuntimeError("unexpected Experiment 3 discovery tensor shape")
    if manifest["confirmation_shape"] != [expected["confirmation_sequences"], expected["sequence_length"]]:
        raise RuntimeError("unexpected Experiment 3 confirmation tensor shape")
    source = manifest["metadata"]["dataset"]["matched_files"]
    expected_source = ENVIRONMENT["experiment3_evaluation_dataset"]["files"]
    if [row["sha256"] for row in source] != [row["sha256"] for row in expected_source]:
        raise RuntimeError("artifact was not materialized from the locked Experiment 3 shard")
    if digest in OLD_ARTIFACT_HASHES:
        raise RuntimeError("Experiment 3 artifact reuses a prior experiment artifact")
    return digest


def check_packages() -> None:
    if f"{sys.version_info.major}.{sys.version_info.minor}" != ENVIRONMENT["python"]:
        raise RuntimeError(f"Python {ENVIRONMENT['python']} required")
    for name, expected in ENVIRONMENT["packages"].items():
        observed = importlib.import_module(name).__version__
        if observed != expected:
            raise RuntimeError(f"{name}=={expected} required, found {observed}")


def check_gpus() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")
    required = int(os.environ.get("MHAR_MIN_GPUS", "1"))
    if torch.cuda.device_count() < required:
        raise RuntimeError(f"at least {required} visible GPUs required")
    minimum = ENVIRONMENT["minimum_gpu_memory_gib"] * 2**30
    for index in range(required):
        properties = torch.cuda.get_device_properties(index)
        if properties.total_memory < minimum or not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"GPU {index} fails memory/bf16 requirements")
        print(f"GPU {index}: {properties.name}, {properties.total_memory / 2**30:.1f} GiB")


def check_external_state(output_root: Path) -> None:
    free_gib = shutil.disk_usage(output_root).free / 2**30
    required = float(os.environ.get("MHAR_MIN_FREE_DISK_GIB", "250"))
    if free_gib < required:
        raise RuntimeError(f"only {free_gib:.1f} GiB free; require {required:.0f} GiB")
    subprocess.run(["gh", "auth", "status", "--hostname", "github.com"], check=True)
    import wandb
    if not wandb.Api(timeout=30).viewer:
        raise RuntimeError("W&B authentication returned no viewer")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if status:
        raise RuntimeError(f"repository is not clean:\n{status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, default=Path(
        "/root/autodl-tmp/datasets/fineweb-edu-sample-10BT/experiment3-eval"))
    parser.add_argument("--output-root", type=Path, default=Path(
        "/root/autodl-tmp/experiment3"))
    parser.add_argument("--artifact-only", action="store_true")
    args = parser.parse_args()
    if not args.artifact_only:
        check_source_shard(args.eval_dir)
    digest = check_artifact(args.artifact)
    if not args.artifact_only:
        check_packages()
        check_gpus()
        args.output_root.mkdir(parents=True, exist_ok=True)
        check_external_state(args.output_root)
    print(f"Experiment 3 preflight passed; artifact SHA-256={digest}")


if __name__ == "__main__":
    main()
