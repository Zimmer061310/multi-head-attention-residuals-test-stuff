#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <b|c4|g4|c8|g8>" >&2
  exit 2
fi
MHAR_VARIANT="$1"
case "$MHAR_VARIANT" in b|c4|g4|c8|g8) ;; *) echo "not a new Experiment 6 run: $MHAR_VARIANT" >&2; exit 2 ;; esac

MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-python3}"
MHAR_ROOT="${MHAR_ROOT:-/root/autodl-tmp/experiment6/screening}"
MHAR_CHECKPOINT="${MHAR_CHECKPOINT:-$MHAR_ROOT/$MHAR_VARIANT/step-2000}"
MHAR_ARTIFACT="${MHAR_ARTIFACT:-/root/autodl-tmp/experiment2/stage-b-screening/fixed_eval.pt}"
MHAR_RESULTS_ROOT="${MHAR_RESULTS_ROOT:-$MHAR_ROOT/results}"
MHAR_WANDB_PROJECT="${MHAR_WANDB_PROJECT:-MHAR Stuff}"
MHAR_WANDB_GROUP="${MHAR_WANDB_GROUP:-mhar-exp6-coupled-qkv-screen-seed42-eval}"

test -f "$MHAR_CHECKPOINT/training_manifest.json"
test -f "$MHAR_ARTIFACT"
mkdir -p "$MHAR_RESULTS_ROOT/$MHAR_VARIANT"
cd "$MHAR_REPO_DIR"
exec "$MHAR_PYTHON_BIN" -m src.experiments.experiment6_screening evaluate \
  --variant "$MHAR_VARIANT" \
  --checkpoint "$MHAR_CHECKPOINT" \
  --artifact "$MHAR_ARTIFACT" \
  --output-dir "$MHAR_RESULTS_ROOT/$MHAR_VARIANT" \
  --device cuda \
  --dtype bf16 \
  --batch-size 1 \
  --wandb-mode online \
  --wandb-project "$MHAR_WANDB_PROJECT" \
  --wandb-group "$MHAR_WANDB_GROUP" \
  --wandb-run-name "mhar-exp6-$MHAR_VARIANT-step2000-eval"
