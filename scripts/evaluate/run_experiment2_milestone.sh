#!/usr/bin/env bash
set -euo pipefail

# Pause the native-H16 run at a preregistered atomic checkpoint and execute the
# complete Experiment 2 Stage A workflow. This script never resumes training.

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
MHAR_TRAIN_OUTPUT_DIR="${MHAR_TRAIN_OUTPUT_DIR:-/root/autodl-tmp/experiment2/checkpoint-1b-h16-fineweb-edu}"
MHAR_CHECKPOINT_DIR="${MHAR_CHECKPOINT_DIR:-$MHAR_TRAIN_OUTPUT_DIR/step-$MHAR_MILESTONE_STEP}"
MHAR_FIXED_ARTIFACT="${MHAR_FIXED_ARTIFACT:-/root/autodl-tmp/experiment2/mixed-width-1b-h16/fixed_eval.pt}"
MHAR_OUTPUT_ROOT="${MHAR_OUTPUT_ROOT:-/root/autodl-tmp/experiment2/milestones/step-$MHAR_MILESTONE_STEP}"
MHAR_TRAIN_SCREEN="${MHAR_TRAIN_SCREEN:-mhar-exp2-h16}"
MHAR_POLL_SECONDS="${MHAR_POLL_SECONDS:-60}"
MHAR_STOP_TRAINING="${MHAR_STOP_TRAINING:-1}"
MHAR_WANDB_PROJECT="${MHAR_WANDB_PROJECT:-MHAR Stuff}"
MHAR_WANDB_GROUP="${MHAR_WANDB_GROUP:-mhar-exp2-1b-h16-step-$MHAR_MILESTONE_STEP}"
MHAR_HF_HOME="${MHAR_HF_HOME:-/root/autodl-tmp/huggingface}"
MHAR_WANDB_DIR="${MHAR_WANDB_DIR:-/root/autodl-tmp/wandb}"

export HF_HOME="$MHAR_HF_HOME"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_DIR="$MHAR_WANDB_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$MHAR_OUTPUT_ROOT" "$MHAR_HF_HOME" "$MHAR_WANDB_DIR"
cd "$MHAR_REPO_DIR"

if [[ ! -f "$MHAR_FIXED_ARTIFACT" ]]; then
  echo "fixed Experiment 2 artifact is missing: $MHAR_FIXED_ARTIFACT" >&2
  exit 1
fi

TRAIN_PATTERN="src.training.train_scratch.*--out_dir $MHAR_TRAIN_OUTPUT_DIR"
while [[ ! -f "$MHAR_CHECKPOINT_DIR/training_manifest.json" ]]; do
  if ! pgrep -f "$TRAIN_PATTERN" >/dev/null; then
    echo "training stopped before milestone checkpoint appeared: $MHAR_CHECKPOINT_DIR" >&2
    exit 1
  fi
  echo "waiting for atomic H16 milestone: $MHAR_CHECKPOINT_DIR"
  sleep "$MHAR_POLL_SECONDS"
done

"$MHAR_PYTHON_BIN" -c \
  'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); expected=int(sys.argv[2]); actual=json.loads(p.read_text())["global_step"]; assert actual == expected, (actual, expected)' \
  "$MHAR_CHECKPOINT_DIR/training_manifest.json" "$MHAR_MILESTONE_STEP"

if [[ "$MHAR_STOP_TRAINING" == "1" ]] && pgrep -f "$TRAIN_PATTERN" >/dev/null; then
  echo "milestone checkpoint is atomic; interrupting $MHAR_TRAIN_SCREEN"
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

RUN_PREFIX="mhar-exp2-1b-h16-step-$MHAR_MILESTONE_STEP"

"$MHAR_PYTHON_BIN" -m src.experiments.experiment2_mixed_width evaluate \
  "${COMMON_EVAL_ARGS[@]}" \
  --output-dir "$MHAR_OUTPUT_ROOT/smoke" \
  --split discovery \
  --smoke-limit 3 \
  --wandb-run-name "$RUN_PREFIX-smoke"

"$MHAR_PYTHON_BIN" -m src.experiments.experiment2_mixed_width evaluate \
  "${COMMON_EVAL_ARGS[@]}" \
  --output-dir "$MHAR_OUTPUT_ROOT/run" \
  --split discovery \
  --wandb-run-name "$RUN_PREFIX-discovery-495"

"$MHAR_PYTHON_BIN" -m src.experiments.experiment2_mixed_width analyze \
  --discovery-results "$MHAR_OUTPUT_ROOT/run/discovery_results.jsonl" \
  --output-dir "$MHAR_OUTPUT_ROOT/run/discovery-analysis" \
  --wandb-mode online \
  --wandb-project "$MHAR_WANDB_PROJECT" \
  --wandb-group "$MHAR_WANDB_GROUP" \
  --wandb-run-name "$RUN_PREFIX-discovery-analysis"

"$MHAR_PYTHON_BIN" -m src.experiments.experiment2_mixed_width evaluate \
  "${COMMON_EVAL_ARGS[@]}" \
  --output-dir "$MHAR_OUTPUT_ROOT/run" \
  --split confirmation \
  --selection-manifest "$MHAR_OUTPUT_ROOT/run/discovery-analysis/confirmation_selection.json" \
  --wandb-run-name "$RUN_PREFIX-confirmation"

"$MHAR_PYTHON_BIN" -m src.experiments.experiment2_mixed_width analyze \
  --discovery-results "$MHAR_OUTPUT_ROOT/run/discovery_results.jsonl" \
  --confirmation-results "$MHAR_OUTPUT_ROOT/run/confirmation_results.jsonl" \
  --output-dir "$MHAR_OUTPUT_ROOT/run/final-analysis" \
  --wandb-mode online \
  --wandb-project "$MHAR_WANDB_PROJECT" \
  --wandb-group "$MHAR_WANDB_GROUP" \
  --wandb-run-name "$RUN_PREFIX-final-analysis"

echo "Experiment 2 milestone $MHAR_MILESTONE_STEP complete. Training remains paused."
