"""Download and content-verify frozen Stage B FineWeb-Edu shards."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT = ROOT / "configs/environment/stage-b-server.json"


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path, spec):
    path = Path(path)
    return (
        path.is_file()
        and path.stat().st_size == spec["bytes"]
        and sha256_file(path) == spec["sha256"]
    )


def resolve_source_url(source_url, endpoint=None):
    """Use a transport mirror without changing the content-addressed spec."""
    endpoint = endpoint or os.environ.get("MHAR_HF_ENDPOINT") or os.environ.get("HF_ENDPOINT")
    if not endpoint:
        return source_url
    source = urlparse(source_url)
    mirror = urlparse(endpoint)
    if source.hostname != "huggingface.co" or not mirror.scheme or not mirror.netloc:
        return source_url
    return f"{mirror.scheme}://{mirror.netloc}{source.path}"


def download_file(spec, destination, timeout=60):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if verify_file(destination, spec):
        print(f"verified existing dataset shard: {destination}", flush=True)
        return
    if destination.exists():
        raise RuntimeError(
            f"refusing to overwrite invalid dataset file: {destination}; move it aside first")

    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    source_url = resolve_source_url(spec["source_url"])
    if source_url != spec["source_url"]:
        print(f"using verified-content mirror: {source_url}", flush=True)
    with requests.get(source_url, headers=headers, stream=True, timeout=timeout) as response:
        if existing and response.status_code != 206:
            partial.unlink()
            existing = 0
            response.close()
            return download_file(spec, destination, timeout)
        response.raise_for_status()
        mode = "ab" if existing else "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())

    if partial.stat().st_size != spec["bytes"]:
        raise RuntimeError(
            f"downloaded size mismatch for {partial}: {partial.stat().st_size} != {spec['bytes']}")
    digest = sha256_file(partial)
    if digest != spec["sha256"]:
        raise RuntimeError(f"downloaded SHA-256 mismatch for {partial}: {digest}")
    os.replace(partial, destination)
    print(f"downloaded and verified: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--dataset", choices=("train", "evaluation", "experiment3-evaluation"),
        default="train")
    args = parser.parse_args()
    environment = json.loads(ENVIRONMENT.read_text(encoding="utf-8"))
    dataset_keys = {
        "train": "dataset",
        "evaluation": "evaluation_dataset",
        "experiment3-evaluation": "experiment3_evaluation_dataset",
    }
    dataset = environment[dataset_keys[args.dataset]]
    output_dir = Path(args.output_dir or dataset["directory"])
    for spec in dataset["files"]:
        download_file(spec, output_dir / spec["name"])


if __name__ == "__main__":
    main()
