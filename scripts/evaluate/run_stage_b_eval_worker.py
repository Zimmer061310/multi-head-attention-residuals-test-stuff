"""Evaluate one or more Stage B variants sequentially on one visible GPU."""

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone", type=int, required=True)
    parser.add_argument("--variants", required=True)
    args = parser.parse_args()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    if not variants:
        raise ValueError("worker requires at least one variant")
    for variant in variants:
        print(f"evaluating {variant} at step {args.milestone}", flush=True)
        subprocess.run([
            str(ROOT / "scripts/evaluate/run_experiment2_stage_b_screen.sh"),
            variant, str(args.milestone),
        ], cwd=ROOT, env=os.environ.copy(), check=True)


if __name__ == "__main__":
    main()
