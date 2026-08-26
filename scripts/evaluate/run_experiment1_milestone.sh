#!/usr/bin/env bash
set -euo pipefail

# Pause the exact 1B H=4 training run at a preregistered checkpoint and run the
# complete Experiment 1 workflow against that immutable checkpoint. Relaunching
# this script resumes evaluation in the same output directories.

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <2000|5000|10000|20000>" >&2
  exit 2
fi

MHAR_MILESTONE_STEP="$1"
case "$MHAR_MILESTONE_STEP" in
  2000|5000|10000|20000) ;;
  *)
    echo "unsupported milestone: $MHAR_MILESTONE_STEP" >&2
    exit 2
    ;;
esac

MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-python3}"
MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_TRAIN_OUTPUT_DIR="${MHAR_TRAIN_OUTPUT_DIR:-/root/autodl-tmp/experiment1/checkpoint-1b-h4-fineweb-edu}"
MHAR_CHECKPOINT_DIR="${MHAR_CHECKPOINT_DIR:-$MHAR_TRAIN_OUTPUT_DIR/step-$MHAR_MILESTONE_STEP}"
MHAR_FIXED_ARTIFACT="${MHAR_FIXED_ARTIFACT:-/root/autodl-tmp/experiment1/partition-compatibility-1b-h4/fixed_eval.pt}"
MHAR_OUTPUT_ROOT="${MHAR_OUTPUT_ROOT:-/root/autodl-tmp/experiment1/milestones/step-$MHAR_MILESTONE_STEP}"
MHAR_TRAIN_SCREEN="${MHAR_TRAIN_SCREEN:-mhar-1b-h4}"
MHAR_POLL_SECONDS="${MHAR_POLL_SECONDS:-60}"
MHAR_STOP_TRAINING="${MHAR_STOP_TRAINING:-1}"
MHAR_WANDB_PROJECT="${MHAR_WANDB_PROJECT:-MHAR Stuff}"
MHAR_WANDB_GROUP="${MHAR_WANDB_GROUP:-mhar-exp1-1b-h4-step-$MHAR_MILESTONE_STEP}"
MHAR_HF_HOME="${MHAR_HF_HOME:-/root/autodl-tmp/huggingface}"
MHAR_WANDB_DIR="${MHAR_WANDB_DIR:-/root/autodl-tmp/wandb}"

export HF_HOME="$MHAR_HF_HOME"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_DIR="$MHAR_WANDB_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$MHAR_OUTPUT_ROOT" "$MHAR_HF_HOME" "$MHAR_WANDB_DIR"
cd "$MHAR_REPO_DIR"

if [[ ! -f "$MHAR_FIXED_ARTIFACT" ]]; then
  echo "fixed evaluation artifact is missing: $MHAR_FIXED_ARTIFACT" >&2
  exit 1
fi

TRAIN_PATTERN="src.training.train_scratch.*--out_dir $MHAR_TRAIN_OUTPUT_DIR"
while [[ ! -f "$MHAR_CHECKPOINT_DIR/training_manifest.json" ]]; do
  if ! pgrep -f "$TRAIN_PATTERN" >/dev/null; then
    echo "training stopped before milestone checkpoint appeared: $MHAR_CHECKPOINT_DIR" >&2
    exit 1
  fi
  echo "waiting for atomic milestone checkpoint: $MHAR_CHECKPOINT_DIR"
  sleep "$MHAR_POLL_SECONDS"
done

"$MHAR_PYTHON_BIN" -c \
  'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); expected=int(sys.argv[2]); actual=json.loads(p.read_text())["global_step"]; assert actual == expected, (actual, expected)' \
  "$MHAR_CHECKPOINT_DIR/training_manifest.json" "$MHAR_MILESTONE_STEP"

if [[ "$MHAR_STOP_TRAINING" == "1" ]] && pgrep -f "$TRAIN_PATTERN" >/dev/null; then
  echo "checkpoint is complete; interrupting training screen $MHAR_TRAIN_SCREEN"
  screen -S "$MHAR_TRAIN_SCREEN" -X stuff $'\003'
  while pgrep -f "$TRAIN_PATTERN" >/dev/null; do
    sleep 5
  done
fi

COMMON_EVAL_ARGS=(
  --checkpoint "$MHAR_CHECKPOINT_DIR"
  --artifact "$MHAR_FIXED_ARTIFACT"
  --device cuda
  --dtype bf16
  --batch-size 1
  --wandb-mode online
  --wandb-project "$MHAR_WANDB_PROJECT"
  --wandb-group "$MHAR_WANDB_GROUP"
)

RUN_PREFIX="mhar-exp1-1b-h4-step-$MHAR_MILESTONE_STEP"

"$MHAR_PYTHON_BIN" -m src.experiments.experiment1_partition_compatibility evaluate \
  "${COMMON_EVAL_ARGS[@]}" \
  --output-dir "$MHAR_OUTPUT_ROOT/smoke" \
  --split discovery \
  --smoke-limit 3 \
  --wandb-run-name "$RUN_PREFIX-smoke"

"$MHAR_PYTHON_BIN" -m src.experiments.experiment1_partition_compatibility evaluate \
  "${COMMON_EVAL_ARGS[@]}" \
  --output-dir "$MHAR_OUTPUT_ROOT/run" \
  --split discovery \
  --wandb-run-name "$RUN_PREFIX-discovery-105"

"$MHAR_PYTHON_BIN" -m src.experiments.experiment1_partition_compatibility analyze \
  --discovery-results "$MHAR_OUTPUT_ROOT/run/discovery_results.jsonl" \
  --output-dir "$MHAR_OUTPUT_ROOT/run/analysis" \
  --wandb-mode online \
  --wandb-project "$MHAR_WANDB_PROJECT" \
  --wandb-group "$MHAR_WANDB_GROUP" \
  --wandb-run-name "$RUN_PREFIX-discovery-analysis"

"$MHAR_PYTHON_BIN" -m src.experiments.experiment1_partition_compatibility evaluate \
  "${COMMON_EVAL_ARGS[@]}" \
  --output-dir "$MHAR_OUTPUT_ROOT/run" \
  --split confirmation \
  --discovery-results "$MHAR_OUTPUT_ROOT/run/discovery_results.jsonl" \
  --wandb-run-name "$RUN_PREFIX-confirmation"

"$MHAR_PYTHON_BIN" -m src.experiments.experiment1_partition_compatibility analyze \
  --discovery-results "$MHAR_OUTPUT_ROOT/run/discovery_results.jsonl" \
  --confirmation-results "$MHAR_OUTPUT_ROOT/run/confirmation_results.jsonl" \
  --output-dir "$MHAR_OUTPUT_ROOT/run/final-analysis" \
  --wandb-mode online \
  --wandb-project "$MHAR_WANDB_PROJECT" \
  --wandb-group "$MHAR_WANDB_GROUP" \
  --wandb-run-name "$RUN_PREFIX-final-analysis"

echo "Milestone $MHAR_MILESTONE_STEP complete: $MHAR_OUTPUT_ROOT/run/final-analysis"
