#!/usr/bin/env bash
set -euo pipefail

MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-python3}"
MHAR_RESULTS_ROOT="${MHAR_RESULTS_ROOT:-/root/autodl-tmp/experiment3/results}"
MHAR_OUTPUT_DIR="${MHAR_OUTPUT_DIR:-$MHAR_RESULTS_ROOT/cross-seed}"
MHAR_WANDB_ENTITY="${MHAR_WANDB_ENTITY:-zimmer061310-ena}"

cd "$MHAR_REPO_DIR"
"$MHAR_PYTHON_BIN" -m src.experiments.experiment3_cross_seed \
  --seed-bundle "42=$MHAR_RESULTS_ROOT/seed-42" \
  --seed-bundle "43=$MHAR_RESULTS_ROOT/seed-43" \
  --seed-bundle "44=$MHAR_RESULTS_ROOT/seed-44" \
  --output-dir "$MHAR_OUTPUT_DIR" --wandb-entity "$MHAR_WANDB_ENTITY" \
  --wandb-group mhar-exp3-boundary-learnability-cross-seed --wandb-mode online
