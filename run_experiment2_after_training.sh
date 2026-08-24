#!/usr/bin/env bash
set -euo pipefail

# Queue the locked Experiment 2 discovery and confirmation workflow behind the
# matched native-H16 checkpoint. Every evaluation stage is resumable.

MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-python3}"
MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MHAR_TRAIN_OUTPUT_DIR="${MHAR_TRAIN_OUTPUT_DIR:-/root/autodl-tmp/experiment2/checkpoint-1b-h16-fineweb-edu}"
MHAR_CHECKPOINT_DIR="${MHAR_CHECKPOINT_DIR:-$MHAR_TRAIN_OUTPUT_DIR/final}"
MHAR_EVAL_DATA_FILES="${MHAR_EVAL_DATA_FILES:-/root/autodl-tmp/datasets/fineweb-edu-sample-10BT/eval/*.parquet}"
MHAR_OUTPUT_ROOT="${MHAR_OUTPUT_ROOT:-/root/autodl-tmp/experiment2/mixed-width-1b-h16}"
MHAR_FIXED_ARTIFACT="${MHAR_FIXED_ARTIFACT:-$MHAR_OUTPUT_ROOT/fixed_eval.pt}"
MHAR_POLL_SECONDS="${MHAR_POLL_SECONDS:-300}"
MHAR_WANDB_PROJECT="${MHAR_WANDB_PROJECT:-MHAR Stuff}"
MHAR_WANDB_GROUP="${MHAR_WANDB_GROUP:-mhar-exp2-1b-h16-fineweb-edu-seed42}"
MHAR_HF_HOME="${MHAR_HF_HOME:-/root/autodl-tmp/huggingface}"
MHAR_WANDB_DIR="${MHAR_WANDB_DIR:-/root/autodl-tmp/wandb}"

export HF_HOME="$MHAR_HF_HOME"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_DIR="$MHAR_WANDB_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$MHAR_OUTPUT_ROOT" "$MHAR_HF_HOME" "$MHAR_WANDB_DIR"
cd "$MHAR_REPO_DIR"

if [[ ! -f "$MHAR_FIXED_ARTIFACT" ]]; then
  "$MHAR_PYTHON_BIN" experiment2_mixed_width.py materialize \
    --data-files "$MHAR_EVAL_DATA_FILES" \
    --tokenizer Qwen/Qwen3-0.6B \
    --tokenizer-revision c1899de289a04d12100db370d81485cdf75e47ca \
    --seq-len 1024 \
    --discovery-sequences 512 \
    --confirmation-sequences 512 \
    --seed 20260824 \
    --output "$MHAR_FIXED_ARTIFACT"
fi

while [[ ! -f "$MHAR_CHECKPOINT_DIR/training_manifest.json" ]]; do
  if ! pgrep -f "train_scratch.py.*--out_dir $MHAR_TRAIN_OUTPUT_DIR" >/dev/null; then
    echo "training stopped before final H16 checkpoint appeared: $MHAR_CHECKPOINT_DIR" >&2
    exit 1
  fi
  echo "waiting for final H16 checkpoint: $MHAR_CHECKPOINT_DIR"
  sleep "$MHAR_POLL_SECONDS"
done

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

"$MHAR_PYTHON_BIN" experiment2_mixed_width.py evaluate \
  "${COMMON_EVAL_ARGS[@]}" \
  --output-dir "$MHAR_OUTPUT_ROOT/smoke" \
  --split discovery \
  --smoke-limit 3 \
  --wandb-run-name mhar-exp2-1b-h16-smoke

"$MHAR_PYTHON_BIN" experiment2_mixed_width.py evaluate \
  "${COMMON_EVAL_ARGS[@]}" \
  --output-dir "$MHAR_OUTPUT_ROOT/run" \
  --split discovery \
  --wandb-run-name mhar-exp2-1b-h16-discovery-495

"$MHAR_PYTHON_BIN" experiment2_mixed_width.py analyze \
  --discovery-results "$MHAR_OUTPUT_ROOT/run/discovery_results.jsonl" \
  --output-dir "$MHAR_OUTPUT_ROOT/run/discovery-analysis" \
  --wandb-mode online \
  --wandb-project "$MHAR_WANDB_PROJECT" \
  --wandb-group "$MHAR_WANDB_GROUP" \
  --wandb-run-name mhar-exp2-1b-h16-discovery-analysis

"$MHAR_PYTHON_BIN" experiment2_mixed_width.py evaluate \
  "${COMMON_EVAL_ARGS[@]}" \
  --output-dir "$MHAR_OUTPUT_ROOT/run" \
  --split confirmation \
  --selection-manifest "$MHAR_OUTPUT_ROOT/run/discovery-analysis/confirmation_selection.json" \
  --wandb-run-name mhar-exp2-1b-h16-confirmation

"$MHAR_PYTHON_BIN" experiment2_mixed_width.py analyze \
  --discovery-results "$MHAR_OUTPUT_ROOT/run/discovery_results.jsonl" \
  --confirmation-results "$MHAR_OUTPUT_ROOT/run/confirmation_results.jsonl" \
  --output-dir "$MHAR_OUTPUT_ROOT/run/final-analysis" \
  --wandb-mode online \
  --wandb-project "$MHAR_WANDB_PROJECT" \
  --wandb-group "$MHAR_WANDB_GROUP" \
  --wandb-run-name mhar-exp2-1b-h16-final-analysis

echo "Experiment 2 Stage A complete: $MHAR_OUTPUT_ROOT/run/final-analysis"
