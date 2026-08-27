"""Run H16/H8/H4 on a three-GPU helper and return atomic checkpoints.

The five-GPU primary remains the only evaluation host.  This helper resumes
the three uniform models, pauses them at the requested atomic milestone, and
copies the completed checkpoints back.  After step 2000 it follows the
primary's already-frozen gate: stop, or continue the same three runs to 5000.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

from scripts.setup.pause_stage_b_at_milestone import (
    active_screen_names,
    checkpoint_step,
)
from scripts.setup.run_stage_b_5gpu_queue import (
    EXPECTED_TRAINING_COMMIT,
    PYTHON,
    TRAINING_ROOT,
    UNIFORM_VARIANTS,
    launch_training,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = "/root/autodl-tmp/experiment2/stage-b-screening"


def pause_status_path(output_root, milestone):
    return Path(output_root) / (
        f"milestone-{milestone}-uniform-remote-pause-status.json")


def primary_summary_path(output_root, milestone):
    return Path(output_root) / "milestones" / f"step-{milestone}" / (
        "analysis/summary.json")


def ssh_base(host, port, key):
    return [
        "ssh", "-i", key, "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes", "-p", str(port),
        f"root@{host}",
    ]


def start_pause(milestone, output_root):
    screen = f"mhar-stageb-pause-{milestone}-uniform-remote"
    status = pause_status_path(output_root, milestone)
    if screen not in active_screen_names() and not (
            status.is_file()
            and json.loads(status.read_text(encoding="utf-8")).get("complete")):
        subprocess.run([
            "screen", "-L", "-Logfile",
            str(Path(output_root) / f"pause-{milestone}-uniform-remote.log"),
            "-dmS", screen, PYTHON,
            str(ROOT / "scripts/setup/pause_stage_b_at_milestone.py"),
            "--milestone", str(milestone), "--poll-seconds", "10",
            "--output-root", output_root,
            "--variants", ",".join(UNIFORM_VARIANTS),
            "--status-path", str(status),
        ], check=True)
    return screen, status


def run_uniform_milestone(milestone, output_root, poll_seconds):
    subprocess.run(["screen", "-wipe"], check=False, capture_output=True)
    for gpu, variant in enumerate(UNIFORM_VARIANTS):
        launch_training(variant, gpu, milestone, output_root, gpu)
    pause_screen, status_path = start_pause(milestone, output_root)
    while True:
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("complete"):
                return
            if status.get("failed") and pause_screen not in active_screen_names():
                raise RuntimeError(
                    f"uniform milestone {milestone} failed: {status['failed']}")
        time.sleep(poll_seconds)


def transfer_milestone(
        milestone, output_root, primary_host, primary_port, ssh_key):
    destination = f"root@{primary_host}"
    remote_shell = (
        f"ssh -i {ssh_key} -o BatchMode=yes "
        f"-o StrictHostKeyChecking=yes -p {primary_port}")
    for variant in UNIFORM_VARIANTS:
        if checkpoint_step(output_root, variant, milestone) != milestone:
            raise RuntimeError(
                f"refusing incomplete transfer: {variant} step {milestone}")
        remote_directory = f"{output_root}/{variant}"
        subprocess.run(
            ssh_base(primary_host, primary_port, ssh_key)
            + ["mkdir", "-p", remote_directory],
            check=True,
        )
        subprocess.run([
            "rsync", "-aH", "--partial", "--info=progress2",
            "-e", remote_shell,
            str(Path(output_root) / variant / f"step-{milestone}"),
            str(Path(output_root) / variant / "training_run_manifest.json"),
            f"{destination}:{remote_directory}/",
        ], check=True)
    print(f"returned all uniform step-{milestone} checkpoints", flush=True)


def read_primary_decision(
        milestone, output_root, primary_host, primary_port, ssh_key):
    path = str(primary_summary_path(output_root, milestone))
    result = subprocess.run(
        ssh_base(primary_host, primary_port, ssh_key) + ["cat", path],
        text=True, capture_output=True, check=False,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)["decision"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return None


def verify_training_commit():
    observed = subprocess.run(
        ["git", "-C", str(TRAINING_ROOT), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    if observed != EXPECTED_TRAINING_COMMIT:
        raise RuntimeError(
            "training worktree must remain at the checkpoint source commit: "
            f"expected {EXPECTED_TRAINING_COMMIT}, observed {observed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-host", required=True)
    parser.add_argument("--primary-port", required=True, type=int)
    parser.add_argument("--ssh-key", required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--shutdown-grace-seconds", type=int, default=600)
    args = parser.parse_args()
    verify_training_commit()

    run_uniform_milestone(2000, args.output_root, args.poll_seconds)
    transfer_milestone(
        2000, args.output_root, args.primary_host, args.primary_port,
        args.ssh_key)

    while True:
        decision = read_primary_decision(
            2000, args.output_root, args.primary_host, args.primary_port,
            args.ssh_key)
        if decision in {"stop_at_2000", "resume_to_5000"}:
            break
        time.sleep(args.poll_seconds)

    if decision == "resume_to_5000":
        run_uniform_milestone(5000, args.output_root, args.poll_seconds)
        transfer_milestone(
            5000, args.output_root, args.primary_host, args.primary_port,
            args.ssh_key)

    print(
        f"uniform helper complete after decision={decision}; shutting down in "
        f"{args.shutdown_grace_seconds}s",
        flush=True,
    )
    time.sleep(args.shutdown_grace_seconds)
    subprocess.run(["/bin/bash", "-lc", "shutdown -h now"], check=True)


if __name__ == "__main__":
    main()
