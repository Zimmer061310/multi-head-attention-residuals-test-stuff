"""Pause every Stage B training screen after an atomic milestone checkpoint.

This controller never powers off the host and never resumes training.  It is
safe to restart: a run with a complete milestone checkpoint and no live screen
is treated as already paused.
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCREENING_CONFIG = ROOT / "configs/experiment2/stage-b-screening.json"


def load_spec(milestone):
    manifest = json.loads(SCREENING_CONFIG.read_text(encoding="utf-8"))
    protected = {int(step) for step in manifest["protected_checkpoints"]}
    if milestone not in protected:
        raise ValueError(
            f"milestone {milestone} is not preregistered; expected one of "
            f"{sorted(protected)}"
        )
    return [row["id"] for row in manifest["runs"]]


def active_screen_names():
    result = subprocess.run(
        ["screen", "-list"], text=True, capture_output=True, check=False
    )
    return {
        line.split(".", 1)[1].split()[0]
        for line in result.stdout.splitlines()
        if ".mhar-stageb-" in line
    }


def checkpoint_step(output_root, variant, milestone):
    checkpoint = Path(output_root) / variant / f"step-{milestone}"
    manifest = checkpoint / "training_manifest.json"
    required = (
        checkpoint / "training_state.pt",
        checkpoint / "config.json",
    )
    if not manifest.is_file() or not all(path.is_file() for path in required):
        return None
    if not any(checkpoint.glob("*.safetensors")):
        return None
    try:
        return int(json.loads(manifest.read_text(encoding="utf-8"))["global_step"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def evaluate_state(variants, milestone, output_root, active_screens):
    ready = set()
    running = set()
    failed = set()
    for variant in variants:
        screen_name = f"mhar-stageb-{variant}"
        step = checkpoint_step(output_root, variant, milestone)
        if step == milestone:
            ready.add(variant)
        elif screen_name in active_screens:
            running.add(variant)
        else:
            failed.add(variant)
    return ready, running, failed


def stop_screen(screen_name):
    subprocess.run(
        ["screen", "-S", screen_name, "-X", "quit"],
        check=True,
        capture_output=True,
        text=True,
    )


def write_status(path, *, milestone, variants, paused, running, failed, complete):
    payload = {
        "format_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "milestone": milestone,
        "variants": variants,
        "paused": sorted(paused),
        "running": sorted(running),
        "failed": sorted(failed),
        "complete": complete,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone", type=int, required=True)
    parser.add_argument(
        "--output-root",
        default="/root/autodl-tmp/experiment2/stage-b-screening",
    )
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--status-path", default=None)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        raise ValueError("poll-seconds must be positive")

    variants = load_spec(args.milestone)
    status_path = args.status_path or str(
        Path(args.output_root) / f"milestone-{args.milestone}-pause-status.json"
    )
    paused = set()

    while True:
        active = active_screen_names()
        ready, running, failed = evaluate_state(
            variants, args.milestone, args.output_root, active
        )
        for variant in sorted(ready):
            screen_name = f"mhar-stageb-{variant}"
            if screen_name in active:
                print(
                    f"{variant}: atomic step-{args.milestone} checkpoint verified; "
                    "stopping training screen",
                    flush=True,
                )
                stop_screen(screen_name)
            paused.add(variant)

        remaining_active = active_screen_names()
        paused_now = {
            variant
            for variant in ready
            if f"mhar-stageb-{variant}" not in remaining_active
        }
        paused.update(paused_now)
        running = {
            variant
            for variant in variants
            if f"mhar-stageb-{variant}" in remaining_active
            and variant not in paused
        }
        failed = set(variants) - paused - running
        complete = len(paused) == len(variants) and not running and not failed
        write_status(
            status_path,
            milestone=args.milestone,
            variants=variants,
            paused=paused,
            running=running,
            failed=failed,
            complete=complete,
        )
        print(
            f"paused={len(paused)}/{len(variants)} "
            f"running={sorted(running)} failed={sorted(failed)}",
            flush=True,
        )
        if failed:
            raise RuntimeError(
                "a run exited before producing a complete milestone checkpoint: "
                f"{sorted(failed)}"
            )
        if complete:
            print(
                f"all {len(variants)} runs are safely paused at step "
                f"{args.milestone}; host remains online for evaluation",
                flush=True,
            )
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
