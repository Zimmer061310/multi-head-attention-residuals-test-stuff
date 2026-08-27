"""Resume the eight Stage B runs in two waves on a five-GPU clone."""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from scripts.setup.pause_stage_b_at_milestone import (
    active_screen_names,
    checkpoint_step,
    load_spec,
)


ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/root/autodl-tmp/venvs/mhar-stage-b/bin/python"
DEFAULT_OUTPUT_ROOT = "/root/autodl-tmp/experiment2/stage-b-screening"
MIXED_VARIANTS = (
    "mixed-k2", "mixed-k3", "mixed-k4-best", "mixed-k5", "mixed-k4-worst")
UNIFORM_VARIANTS = ("h16", "h8", "h4")


def latest_atomic_checkpoint(output_root, variant, target_milestone):
    directory = Path(output_root) / variant
    candidates = []
    if directory.is_dir():
        for path in directory.iterdir():
            match = re.fullmatch(r"step-(\d+)", path.name)
            if not match:
                continue
            step = int(match.group(1))
            if step < target_milestone and checkpoint_step(
                    output_root, variant, step) == step:
                candidates.append((step, path))
    if not candidates:
        raise FileNotFoundError(
            f"no atomic checkpoint before step {target_milestone} for {variant}")
    return max(candidates)[1]


def launch_training(variant, gpu, target_milestone, output_root, port_index):
    screen = f"mhar-stageb-{variant}"
    if screen in active_screen_names():
        return
    if checkpoint_step(output_root, variant, target_milestone) == target_milestone:
        return
    checkpoint = latest_atomic_checkpoint(output_root, variant, target_milestone)
    log = Path(output_root) / "logs" / (
        f"{variant}-clone-resume-{checkpoint.name}-to-{target_milestone}.log")
    subprocess.run([
        "screen", "-L", "-Logfile", str(log), "-dmS", screen,
        "env", f"CUDA_VISIBLE_DEVICES={gpu}", f"MHAR_PYTHON_BIN={PYTHON}",
        f"MHAR_OUTPUT_ROOT={output_root}",
        "MHAR_WANDB_GROUP=mhar-exp2-stage-b-screening-seed42",
        f"MHAR_MASTER_PORT={29500 + port_index}",
        str(ROOT / "scripts/train/run_experiment2_stage_b_screen.sh"),
        variant, str(checkpoint),
    ], check=True)
    print(f"launched {variant} on GPU {gpu} from {checkpoint.name}", flush=True)


def start_subset_pause(target_milestone, output_root):
    screen = f"mhar-stageb-pause-{target_milestone}-mixed-wave"
    status = Path(output_root) / (
        f"milestone-{target_milestone}-mixed-wave-pause-status.json")
    if screen not in active_screen_names() and not (
            status.is_file() and json.loads(status.read_text())["complete"]):
        subprocess.run([
            "screen", "-L", "-Logfile",
            str(Path(output_root) / f"pause-{target_milestone}-mixed-wave.log"),
            "-dmS", screen, PYTHON,
            str(ROOT / "scripts/setup/pause_stage_b_at_milestone.py"),
            "--milestone", str(target_milestone), "--poll-seconds", "10",
            "--output-root", output_root,
            "--variants", ",".join(MIXED_VARIANTS),
            "--status-path", str(status),
        ], check=True)
    return status


def start_full_controllers(target_milestone, output_root, no_further_resume):
    active = active_screen_names()
    pause_screen = f"mhar-stageb-pause-{target_milestone}"
    if pause_screen not in active:
        subprocess.run([
            "screen", "-L", "-Logfile",
            str(Path(output_root) / f"pause-{target_milestone}.log"),
            "-dmS", pause_screen, PYTHON,
            str(ROOT / "scripts/setup/pause_stage_b_at_milestone.py"),
            "--milestone", str(target_milestone), "--poll-seconds", "10",
            "--output-root", output_root,
        ], check=True)
    workflow_screen = f"mhar-stageb-workflow-{target_milestone}"
    if workflow_screen not in active_screen_names():
        command = [
            "screen", "-L", "-Logfile",
            str(Path(output_root) / f"workflow-{target_milestone}.log"),
            "-dmS", workflow_screen, PYTHON, "-m",
            "scripts.setup.run_stage_b_milestone_workflow",
            "--milestone", str(target_milestone), "--output-root", output_root,
            "--poll-seconds", "30", "--gpu-count", "5",
        ]
        if no_further_resume:
            command.append("--no-resume")
        subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-milestone", type=int, required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--no-further-resume", action="store_true")
    args = parser.parse_args()
    available = set(load_spec(args.target_milestone))
    if available != set(MIXED_VARIANTS) | set(UNIFORM_VARIANTS):
        raise RuntimeError("five-GPU queue variants do not match the frozen screen")

    subprocess.run(["screen", "-wipe"], check=False, capture_output=True)
    for gpu, variant in enumerate(MIXED_VARIANTS):
        launch_training(variant, gpu, args.target_milestone, args.output_root, gpu)
    wave_status = start_subset_pause(args.target_milestone, args.output_root)
    while True:
        if wave_status.is_file():
            status = json.loads(wave_status.read_text(encoding="utf-8"))
            if status.get("failed"):
                raise RuntimeError(f"mixed wave failed: {status['failed']}")
            if status.get("complete"):
                break
        time.sleep(args.poll_seconds)

    for gpu, variant in enumerate(UNIFORM_VARIANTS):
        launch_training(variant, gpu, args.target_milestone, args.output_root, gpu)
    start_full_controllers(
        args.target_milestone, args.output_root, args.no_further_resume)
    print(
        f"uniform wave launched; full step-{args.target_milestone} controllers active",
        flush=True,
    )


if __name__ == "__main__":
    main()
