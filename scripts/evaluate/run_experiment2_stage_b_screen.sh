#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <variant> <2000|5000|10000|20000>" >&2
  exit 2
fi

MHAR_VARIANT="$1"
MHAR_MILESTONE="$2"
case "$MHAR_MILESTONE" in
  2000|5000|10000|20000) ;;
  *) echo "unsupported milestone: $MHAR_MILESTONE" >&2; exit 2 ;;
esac

MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-/root/autodl-tmp/venvs/mhar-stage-b/bin/python}"
MHAR_TRAIN_ROOT="${MHAR_TRAIN_ROOT:-/root/autodl-tmp/experiment2/stage-b-screening}"
MHAR_CHECKPOINT="${MHAR_CHECKPOINT:-$MHAR_TRAIN_ROOT/$MHAR_VARIANT/step-$MHAR_MILESTONE}"
MHAR_ARTIFACT="${MHAR_ARTIFACT:-$MHAR_TRAIN_ROOT/fixed_eval.pt}"
MHAR_RESULTS_ROOT="${MHAR_RESULTS_ROOT:-$MHAR_TRAIN_ROOT/milestones/step-$MHAR_MILESTONE}"
MHAR_WANDB_PROJECT="${MHAR_WANDB_PROJECT:-MHAR Stuff}"
MHAR_WANDB_GROUP="${MHAR_WANDB_GROUP:-mhar-exp2-stage-b-screening-seed42-step-$MHAR_MILESTONE}"
MHAR_WANDB_DIR="${MHAR_WANDB_DIR:-/root/autodl-tmp/wandb}"

export WANDB_DIR="$MHAR_WANDB_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

test -f "$MHAR_CHECKPOINT/training_manifest.json"
test -f "$MHAR_ARTIFACT"
mkdir -p "$MHAR_RESULTS_ROOT/$MHAR_VARIANT" "$MHAR_WANDB_DIR"
cd "$MHAR_REPO_DIR"

exec "$MHAR_PYTHON_BIN" -m src.experiments.experiment2_stage_b_screening evaluate \
  --variant "$MHAR_VARIANT" \
  --milestone "$MHAR_MILESTONE" \
  --checkpoint "$MHAR_CHECKPOINT" \
  --artifact "$MHAR_ARTIFACT" \
  --output-dir "$MHAR_RESULTS_ROOT/$MHAR_VARIANT" \
  --device cuda \
  --dtype bf16 \
  --batch-size 1 \
  --wandb-mode online \
  --wandb-project "$MHAR_WANDB_PROJECT" \
  --wandb-group "$MHAR_WANDB_GROUP" \
  --wandb-run-name "mhar-exp2-stage-b-$MHAR_VARIANT-step-$MHAR_MILESTONE-eval"
