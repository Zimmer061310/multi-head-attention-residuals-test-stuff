"""Run fixed evaluation after a Stage B pause and apply the frozen chaos gate."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from scripts.setup.pause_stage_b_at_milestone import (
    active_screen_names,
    checkpoint_step,
    load_spec,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = "/root/autodl-tmp/experiment2/stage-b-screening"
PYTHON = "/root/autodl-tmp/venvs/mhar-stage-b/bin/python"
GPU_BY_VARIANT = {
    "h16": 0,
    "h8": 1,
    "mixed-k2": 2,
    "mixed-k3": 3,
    "mixed-k4-best": 4,
    "mixed-k5": 5,
    "mixed-k4-worst": 6,
    "h4": 7,
}


def load_pause_status(output_root, milestone):
    path = Path(output_root) / f"milestone-{milestone}-pause-status.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def eval_state(variants, results_root, active_screens, milestone):
    completed, running, failed = set(), set(), set()
    for variant in variants:
        result = Path(results_root) / variant / "result.json"
        screen = f"mhar-stageb-eval-{milestone}-{variant}"
        if result.is_file():
            completed.add(variant)
        elif screen in active_screens:
            running.add(variant)
        else:
            failed.add(variant)
    return completed, running, failed


def launch_evaluations(variants, milestone, output_root):
    results_root = Path(output_root) / "milestones" / f"step-{milestone}"
    log_root = results_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    active = active_screen_names()
    for variant in variants:
        result = results_root / variant / "result.json"
        screen = f"mhar-stageb-eval-{milestone}-{variant}"
        if result.is_file() or screen in active:
            continue
        subprocess.run([
            "screen", "-L", "-Logfile", str(log_root / f"{variant}.log"),
            "-dmS", screen,
            "env", f"CUDA_VISIBLE_DEVICES={GPU_BY_VARIANT[variant]}",
            f"MHAR_PYTHON_BIN={PYTHON}", f"MHAR_TRAIN_ROOT={output_root}",
            str(ROOT / "scripts/evaluate/run_experiment2_stage_b_screen.sh"),
            variant, str(milestone),
        ], check=True)
        print(f"launched fixed evaluation: {variant} on GPU {GPU_BY_VARIANT[variant]}", flush=True)
    return results_root


def launch_resume(variants, milestone, next_milestone, output_root):
    active = active_screen_names()
    for index, variant in enumerate(GPU_BY_VARIANT):
        if variant not in variants:
            continue
        screen = f"mhar-stageb-{variant}"
        if screen in active:
            raise RuntimeError(f"refusing duplicate training screen: {screen}")
        checkpoint = Path(output_root) / variant / f"step-{milestone}"
        if checkpoint_step(output_root, variant, milestone) != milestone:
            raise RuntimeError(f"invalid resume checkpoint: {checkpoint}")
        subprocess.run([
            "screen", "-L", "-Logfile",
            str(Path(output_root) / "logs" / f"{variant}-resume-{milestone}.log"),
            "-dmS", screen,
            "env", f"CUDA_VISIBLE_DEVICES={GPU_BY_VARIANT[variant]}",
            f"MHAR_PYTHON_BIN={PYTHON}", f"MHAR_OUTPUT_ROOT={output_root}",
            "MHAR_WANDB_GROUP=mhar-exp2-stage-b-screening-seed42",
            f"MHAR_MASTER_PORT={29500 + index}",
            str(ROOT / "scripts/train/run_experiment2_stage_b_screen.sh"),
            variant, str(checkpoint),
        ], check=True)
        print(f"resumed {variant} from step {milestone}", flush=True)

    pause_log = Path(output_root) / f"pause-{next_milestone}.log"
    subprocess.run([
        "screen", "-L", "-Logfile", str(pause_log), "-dmS",
        f"mhar-stageb-pause-{next_milestone}", PYTHON,
        str(ROOT / "scripts/setup/pause_stage_b_at_milestone.py"),
        "--milestone", str(next_milestone), "--poll-seconds", "10",
        "--output-root", output_root,
    ], check=True)
    subprocess.run([
        "screen", "-L", "-Logfile",
        str(Path(output_root) / f"workflow-{next_milestone}.log"), "-dmS",
        f"mhar-stageb-workflow-{next_milestone}", PYTHON,
        str(ROOT / "scripts/setup/run_stage_b_milestone_workflow.py"),
        "--milestone", str(next_milestone), "--output-root", output_root,
        "--no-resume", "--poll-seconds", "30",
    ], check=True)


def analyze(milestone, results_root):
    analysis_dir = Path(results_root) / "analysis"
    summary_path = analysis_dir / "summary.json"
    if not summary_path.is_file():
        subprocess.run([
            PYTHON, "-m", "src.experiments.experiment2_stage_b_screening", "analyze",
            "--milestone", str(milestone), "--results-root", str(results_root),
            "--output-dir", str(analysis_dir), "--wandb-mode", "online",
            "--wandb-project", "MHAR Stuff",
            "--wandb-group", f"mhar-exp2-stage-b-screening-seed42-step-{milestone}",
            "--wandb-run-name", f"mhar-exp2-stage-b-step-{milestone}-analysis",
        ], cwd=ROOT, check=True)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone", type=int, required=True)
    parser.add_argument("--next-milestone", type=int, default=5000)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--shutdown-grace-seconds", type=int, default=4200)
    args = parser.parse_args()
    variants = load_spec(args.milestone)
    artifact = Path(args.output_root) / "fixed_eval.pt"

    while True:
        status = load_pause_status(args.output_root, args.milestone)
        if status and status.get("complete"):
            break
        print(f"waiting for all runs to pause at step {args.milestone}", flush=True)
        time.sleep(args.poll_seconds)
    if not artifact.is_file():
        raise FileNotFoundError(f"fixed evaluation artifact is missing: {artifact}")

    results_root = launch_evaluations(variants, args.milestone, args.output_root)
    while True:
        completed, running, failed = eval_state(
            variants, results_root, active_screen_names(), args.milestone)
        print(
            f"evaluation completed={len(completed)}/{len(variants)} "
            f"running={sorted(running)} failed={sorted(failed)}", flush=True)
        if failed:
            raise RuntimeError(f"fixed evaluation failed: {sorted(failed)}")
        if len(completed) == len(variants):
            break
        time.sleep(args.poll_seconds)

    summary = analyze(args.milestone, results_root)
    print(
        f"milestone decision={summary['decision']} "
        f"rank_spearman={summary['rank_spearman']:+.3f}", flush=True)
    if summary["decision"] == "resume_to_5000" and not args.no_resume:
        launch_resume(variants, args.milestone, args.next_milestone, args.output_root)
        print(f"all runs resumed toward step {args.next_milestone}", flush=True)
        return

    print(
        f"final milestone result uploaded; waiting {args.shutdown_grace_seconds}s "
        "before success-only poweroff", flush=True)
    time.sleep(args.shutdown_grace_seconds)
    subprocess.run(["shutdown", "-h", "now"], check=True)


if __name__ == "__main__":
    main()
