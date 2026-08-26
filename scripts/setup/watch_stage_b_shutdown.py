"""Power off a Stage B server only after every frozen run finishes cleanly."""

import argparse
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCREENING_CONFIG = ROOT / "configs/experiment2/stage-b-screening.json"


def load_spec():
    manifest = json.loads(SCREENING_CONFIG.read_text(encoding="utf-8"))
    return [row["id"] for row in manifest["runs"]], int(manifest["training_steps"])


def active_screen_names():
    result = subprocess.run(
        ["screen", "-list"], text=True, capture_output=True, check=False
    )
    return {
        line.split(".", 1)[1].split()[0]
        for line in result.stdout.splitlines()
        if ".mhar-stageb-" in line
    }


def final_step(output_root, variant):
    manifest = Path(output_root) / variant / "final" / "training_manifest.json"
    if not manifest.is_file():
        return None
    try:
        return int(json.loads(manifest.read_text(encoding="utf-8"))["global_step"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def evaluate_state(variants, required_step, output_root, active_screens):
    completed = set()
    running = set()
    failed = set()
    for variant in variants:
        screen_name = f"mhar-stageb-{variant}"
        if final_step(output_root, variant) == required_step:
            completed.add(variant)
        elif screen_name in active_screens:
            running.add(variant)
        else:
            failed.add(variant)
    return completed, running, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="/root/autodl-tmp/experiment2/stage-b-screening",
    )
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--grace-seconds", type=int, default=600)
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.grace_seconds < 0:
        raise ValueError("poll-seconds must be positive and grace-seconds non-negative")

    variants, required_step = load_spec()
    while True:
        active = active_screen_names()
        completed, running, failed = evaluate_state(
            variants, required_step, args.output_root, active
        )
        print(
            f"completed={len(completed)}/{len(variants)} "
            f"running={sorted(running)} failed={sorted(failed)}",
            flush=True,
        )
        if failed:
            raise RuntimeError(
                "refusing automatic shutdown because runs exited without valid "
                f"step-{required_step} finals: {sorted(failed)}"
            )
        if len(completed) == len(variants) and not running:
            break
        time.sleep(args.poll_seconds)

    print(
        f"all {len(variants)} runs completed step {required_step}; "
        f"waiting {args.grace_seconds}s before shutdown",
        flush=True,
    )
    time.sleep(args.grace_seconds)
    completed, running, failed = evaluate_state(
        variants, required_step, args.output_root, active_screen_names()
    )
    if failed or running or len(completed) != len(variants):
        raise RuntimeError("completion state changed during shutdown grace period")
    print("success conditions revalidated; powering off now", flush=True)
    subprocess.run(["shutdown", "-h", "now"], check=True)


if __name__ == "__main__":
    main()
