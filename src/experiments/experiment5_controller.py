"""One persistent controller, three isolated GPU workers, no automatic retries."""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from src.experiments.experiment5_washout import (
    GROUP, PARTITIONS, ROOT, ROLES, analyze, load_manifest, prepare, read_json,
    sha256_file, source_commit, spec, step0_gate, validate_checkpoint,
    verify_step0, write_new,
)


def training_command(manifest, manifest_path, root, role, gpu):
    offsets = manifest["protocol"]["offsets"]
    command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=1",
               "--master_port", str(29850 + gpu), "--module", "src.training.train_scratch"]
    flags = {
        "mode": "full_mh", "attnres_heads": 16, "hidden_size": 1280,
        "num_layers": 36, "num_heads": 16, "num_kv_heads": 8, "intermediate_size": 5120,
        "seq_len": 1024, "steps": 20000, "stop_after_step": 1600,
        "batch_size": 4, "grad_accum": 8, "expected_global_batch": 32,
        "lr": "5e-4", "lr_min": "5e-5", "warmup": 1000, "max_norm": 1.0, "seed": 43,
        "dataset": "HuggingFaceFW/fineweb-edu", "dataset_name": "sample-10BT",
        "dataset_revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "data_files": manifest["parent_training_manifest"]["run_identity"]["data_files"]["pattern"],
        "tokenizer": "Qwen/Qwen3-0.6B", "tokenizer_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "save_every": 0, "save_steps": ",".join(str(1500 + t) for t in offsets if t),
        "keep_last": 7, "eval_every": 0, "log_every": 1,
        "branch_from": manifest["parent_checkpoint"], "branch_manifest": str(manifest_path),
        "branch_role": role, "out_dir": str(Path(root) / "branches" / role),
        "wandb_project": "MHAR Stuff", "wandb_group": GROUP,
        "run_name": f"mhar-exp5-seed43-{role}",
    }
    for key, value in flags.items():
        command.extend([f"--{key}", str(value)])
    command.extend(["--grad_ckpt", "--wandb_required"])
    if PARTITIONS[role] is not None:
        command.extend(["--mixed_partition", PARTITIONS[role]])
    return command


def eval_command(args, offset):
    return [sys.executable, "-m", "src.experiments.experiment5_washout", "evaluate",
            "--manifest", args.manifest, "--root", args.root, "--role", args.role,
            "--offset", str(offset)]


def worker(args):
    manifest = load_manifest(args.manifest)
    verify_step0(args.root, args.manifest)
    output = Path(args.root) / "branches" / args.role
    if output.exists():
        raise FileExistsError(f"no automatic restart or overwrite: {output}")
    # Training never calls the fixed evaluator and never resumes between snapshots.
    subprocess.run(training_command(manifest, args.manifest, args.root, args.role, args.gpu), cwd=ROOT, check=True)
    validate_checkpoint(manifest, args.manifest, output / "final", args.role, 100)
    for offset in manifest["protocol"]["offsets"]:
        if offset:
            validate_checkpoint(manifest, args.manifest, output / f"step-{1500 + offset}", args.role, offset)
            subprocess.run(eval_command(args, offset), cwd=ROOT, check=True)
    write_new(Path(args.root) / f"COMPLETE-{args.role}.json", {
        "role": args.role, "step": 1600, "branch_manifest_sha256": sha256_file(args.manifest),
        "final_manifest_sha256": sha256_file(output / "final/training_manifest.json"),
    })


def parallel_phase(args, phase):
    jobs = []
    for gpu, role in enumerate(ROLES):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if phase == "step0":
            child = argparse.Namespace(**vars(args)); child.role = role
            cmd = eval_command(child, 0)
        else:
            cmd = [sys.executable, "-m", "src.experiments.experiment5_controller", "worker",
                   "--manifest", args.manifest, "--root", args.root, "--role", role, "--gpu", str(gpu)]
        log = (Path(args.root) / "logs" / f"{phase}-{role}.log").open("x")
        try:
            proc = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        finally:
            log.close()
        jobs.append((role, proc))
    failures = [(role, code) for role, proc in jobs if (code := proc.wait()) != 0]
    if failures:
        raise RuntimeError(f"{phase} workers failed; no retry, analysis or shutdown: {failures}")


def verify_completion(root, manifest_path):
    root = Path(root)
    manifest = load_manifest(manifest_path)
    verify_step0(root, manifest_path)
    for role in ROLES:
        marker = read_json(root / f"COMPLETE-{role}.json")
        if marker != {"role": role, "step": 1600, "branch_manifest_sha256": sha256_file(manifest_path),
                      "final_manifest_sha256": sha256_file(root / "branches" / role / "final/training_manifest.json")}:
            raise RuntimeError("branch completion marker mismatch")
        validate_checkpoint(manifest, manifest_path, root / "branches" / role / "final", role, 100)
    summary_path = root / "analysis/washout_summary.json"
    summary = read_json(summary_path)
    upload = read_json(root / "analysis/WANDB_UPLOAD_COMPLETE.json")
    if upload["summary_sha256"] != sha256_file(summary_path) or summary["branch_manifest_sha256"] != sha256_file(manifest_path):
        raise RuntimeError("analysis/upload identity mismatch")
    for name, digest in summary["measurement_hashes"].items():
        if sha256_file(root / "measurements" / name) != digest:
            raise RuntimeError("final measurement changed")
    return summary


def acknowledge(args):
    # Called ONLY after the operator has verified the local copy and GitHub push.
    verify_completion(args.root, args.manifest)
    if not re.fullmatch(r"[0-9a-f]{40}", args.pushed_commit):
        raise ValueError("acknowledgment requires a full verified GitHub commit")
    summary_path = Path(args.root) / "analysis/washout_summary.json"
    if args.summary_sha256 != sha256_file(summary_path):
        raise RuntimeError("local backup summary hash differs from server")
    write_new(Path(args.root) / "BACKUP_PUSH_ACK.json", {
        "pushed_commit": args.pushed_commit, "summary_sha256": args.summary_sha256,
        "branch_manifest_sha256": sha256_file(args.manifest), "acknowledged_at": time.time(),
    })


def verify_ack(root, manifest_path):
    ack = read_json(Path(root) / "BACKUP_PUSH_ACK.json")
    if (not re.fullmatch(r"[0-9a-f]{40}", ack["pushed_commit"])
            or ack["summary_sha256"] != sha256_file(Path(root) / "analysis/washout_summary.json")
            or ack["branch_manifest_sha256"] != sha256_file(manifest_path)):
        raise RuntimeError("backup acknowledgment does not match completed experiment")
    return ack


def run(args):
    import torch
    root = Path(args.root)
    if root.exists():
        raise FileExistsError("use a fresh Exp5 output directory; do not relaunch accepted work")
    if torch.cuda.device_count() != 3:
        raise RuntimeError("launcher requires exactly the dedicated three-GPU server")
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if processes:
        raise RuntimeError("GPU compute processes already present; refusing competing launch")
    if shutil.disk_usage(root.parent).free < 180 * 2**30:
        raise RuntimeError("need 180 GiB free for protected full-state snapshots; never delete older checkpoints")
    source_commit()
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    args.manifest = str(root / "branch_manifest.json")
    try:
        prepare(args)
        parallel_phase(args, "step0")
        step0_gate(args)
        print("STEP0_REPRODUCTION_PASSED: launching A/B/C on GPUs 0/1/2", flush=True)
        parallel_phase(args, "train-eval")
        args.wandb = True
        analyze(args)
        verify_completion(root, args.manifest)
        write_new(root / "READY_FOR_BACKUP.json", {"summary_sha256": sha256_file(root / "analysis/washout_summary.json")})
        print("SUCCESS: W&B upload complete; awaiting verified local backup + GitHub push acknowledgment", flush=True)
        while not (root / "BACKUP_PUSH_ACK.json").exists():
            time.sleep(30)
        ack = verify_ack(root, args.manifest)
        deadline = max(time.time(), ack["acknowledged_at"]) + spec()["shutdown_grace_seconds"]
        write_new(root / "SHUTDOWN_SCHEDULED.json", {"shutdown_at_unix": deadline, "pushed_commit": ack["pushed_commit"]})
        while time.time() < deadline:
            time.sleep(min(30, deadline - time.time()))
        verify_completion(root, args.manifest)
        verify_ack(root, args.manifest)
        if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
            raise RuntimeError("unexpected live GPU workload; refusing shutdown")
        subprocess.run(["shutdown", "-h", "now"], check=True)
    except Exception as error:
        write_new(root / "FAILED.json", {"type": type(error).__name__, "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("run", run), ("worker", worker), ("acknowledge", acknowledge)):
        p = sub.add_parser(name); p.set_defaults(func=fn)
        p.add_argument("--root", required=True)
        if name == "run":
            p.add_argument("--parent", required=True); p.add_argument("--artifact", required=True)
        else:
            p.add_argument("--manifest", required=True)
        if name == "worker":
            p.add_argument("--role", choices=ROLES, required=True); p.add_argument("--gpu", type=int, required=True)
        if name == "acknowledge":
            p.add_argument("--pushed-commit", required=True); p.add_argument("--summary-sha256", required=True)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
