"""Fail-fast validation for a fresh seven-GPU Stage B training server."""

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT = json.loads(
    (ROOT / "configs/environment/stage-b-server.json").read_text(encoding="utf-8"))


def fail(message):
    raise RuntimeError(message)


def wandb_viewer_name(viewer):
    return (
        getattr(viewer, "username", None)
        or getattr(viewer, "entity", None)
        or str(viewer)
    )


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def check_packages():
    if f"{sys.version_info.major}.{sys.version_info.minor}" != ENVIRONMENT["python"]:
        fail(f"Python {ENVIRONMENT['python']} required, found {sys.version.split()[0]}")
    for name, expected in ENVIRONMENT["packages"].items():
        module = importlib.import_module(name)
        observed = module.__version__
        if observed != expected:
            fail(f"{name}=={expected} required, found {observed}")


def check_gpus():
    import torch

    if not torch.cuda.is_available():
        fail("PyTorch cannot access CUDA")
    required = int(os.environ.get("MHAR_MIN_GPUS", ENVIRONMENT["minimum_gpu_count"]))
    if torch.cuda.device_count() < required:
        fail(f"at least {required} visible GPUs required, found {torch.cuda.device_count()}")
    minimum_bytes = ENVIRONMENT["minimum_gpu_memory_gib"] * 2**30
    for index in range(required):
        properties = torch.cuda.get_device_properties(index)
        if properties.total_memory < minimum_bytes:
            fail(
                f"GPU {index} has {properties.total_memory / 2**30:.1f} GiB; "
                f"at least {ENVIRONMENT['minimum_gpu_memory_gib']} GiB required")
        with torch.cuda.device(index):
            if not torch.cuda.is_bf16_supported():
                fail(f"GPU {index} does not report bf16 support")
        print(
            f"GPU {index}: {properties.name}, {properties.total_memory / 2**30:.1f} GiB",
            flush=True,
        )


def check_dataset():
    dataset = ENVIRONMENT["dataset"]
    directory = Path(os.environ.get("MHAR_DATA_DIR", dataset["directory"]))
    for spec in dataset["files"]:
        path = directory / spec["name"]
        if not path.is_file() or path.stat().st_size != spec["bytes"]:
            fail(f"dataset size check failed: {path}")
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            fail(f"dataset SHA-256 check failed: {path}: {digest}")
        print(f"dataset verified: {path}", flush=True)


def check_disk():
    output_root = Path(os.environ.get(
        "MHAR_OUTPUT_ROOT", "/root/autodl-tmp/experiment2/stage-b-screening"))
    output_root.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output_root).free / 2**30
    recommended = float(os.environ.get(
        "MHAR_MIN_FREE_DISK_GIB", ENVIRONMENT["recommended_free_disk_gib"]))
    if free_gib < recommended:
        fail(f"{free_gib:.1f} GiB free at {output_root}; {recommended:.0f} GiB required")
    print(f"disk free: {free_gib:.1f} GiB at {output_root}", flush=True)


def check_authentication():
    subprocess.run(["gh", "auth", "status", "--hostname", "github.com"], check=True)
    import wandb

    viewer = wandb.Api(timeout=30).viewer
    if not viewer:
        fail("W&B authentication check returned no viewer")
    print(f"W&B authenticated as: {wandb_viewer_name(viewer)}", flush=True)


def check_repository():
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if status:
        fail(f"repository is not clean:\n{status}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    print(f"repository: branch={branch} commit={commit}", flush=True)


def main():
    check_packages()
    check_gpus()
    check_dataset()
    check_disk()
    check_authentication()
    check_repository()
    print("Stage B server preflight passed", flush=True)


if __name__ == "__main__":
    main()
