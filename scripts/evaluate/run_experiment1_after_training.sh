#!/usr/bin/env bash
set -euo pipefail

# Queue the preregistered Experiment 1 workflow behind the exact 1B H=4
# pretraining job.  Every stage is idempotent or resumable, so relaunching this
# script after an interruption continues in the same output directories.

MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-python3}"
MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_TRAIN_OUTPUT_DIR="${MHAR_TRAIN_OUTPUT_DIR:-/root/autodl-tmp/experiment1/checkpoint-1b-h4-fineweb-edu}"
MHAR_CHECKPOINT_DIR="${MHAR_CHECKPOINT_DIR:-$MHAR_TRAIN_OUTPUT_DIR/final}"
MHAR_EVAL_DATA_FILES="${MHAR_EVAL_DATA_FILES:-/root/autodl-tmp/datasets/fineweb-edu-sample-10BT/eval/*.parquet}"
MHAR_OUTPUT_ROOT="${MHAR_OUTPUT_ROOT:-/root/autodl-tmp/experiment1/partition-compatibility-1b-h4}"
MHAR_FIXED_ARTIFACT="${MHAR_FIXED_ARTIFACT:-$MHAR_OUTPUT_ROOT/fixed_eval.pt}"
MHAR_POLL_SECONDS="${MHAR_POLL_SECONDS:-300}"
MHAR_WANDB_PROJECT="${MHAR_WANDB_PROJECT:-MHAR Stuff}"
MHAR_WANDB_GROUP="${MHAR_WANDB_GROUP:-mhar-exp1-1b-h4-fineweb-edu-seed42}"
MHAR_HF_HOME="${MHAR_HF_HOME:-/root/autodl-tmp/huggingface}"
MHAR_WANDB_DIR="${MHAR_WANDB_DIR:-/root/autodl-tmp/wandb}"

export HF_HOME="$MHAR_HF_HOME"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_DIR="$MHAR_WANDB_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$MHAR_OUTPUT_ROOT" "$MHAR_HF_HOME" "$MHAR_WANDB_DIR"
cd "$MHAR_REPO_DIR"

if [[ ! -f "$MHAR_FIXED_ARTIFACT" ]]; then
  "$MHAR_PYTHON_BIN" -m src.experiments.experiment1_partition_compatibility materialize \
    --data-files "$MHAR_EVAL_DATA_FILES" \
    --tokenizer Qwen/Qwen3-0.6B \
    --tokenizer-revision c1899de289a04d12100db370d81485cdf75e47ca \
    --seq-len 1024 \
    --discovery-sequences 512 \
    --confirmation-sequences 512 \
    --output "$MHAR_FIXED_ARTIFACT"
fi

while [[ ! -f "$MHAR_CHECKPOINT_DIR/training_manifest.json" ]]; do
  if ! pgrep -f "src.training.train_scratch.*--out_dir $MHAR_TRAIN_OUTPUT_DIR" >/dev/null; then
    echo "training process stopped before final checkpoint appeared: $MHAR_CHECKPOINT_DIR" >&2
    exit 1
  fi
  echo "waiting for final checkpoint: $MHAR_CHECKPOINT_DIR"
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

# A three-partition production smoke catches checkpoint/loading/parity failures
# before committing several GPU-hours to the exhaustive run.
"$MHAR_PYTHON_BIN" -m src.experiments.experiment1_partition_compatibility evaluate \
  "${COMMON_EVAL_ARGS[@]}" \
  --output-dir "$MHAR_OUTPUT_ROOT/smoke" \
  --split discovery \
  --smoke-limit 3 \
  --wandb-run-name mhar-exp1-1b-h4-smoke

"$MHAR_PYTHON_BIN" -m src.experiments.experiment1_partition_compatibility evaluate \
  "${COMMON_EVAL_ARGS[@]}" \
  --output-dir "$MHAR_OUTPUT_ROOT/run" \
  --split discovery \
  --wandb-run-name mhar-exp1-1b-h4-discovery-105

"$MHAR_PYTHON_BIN" -m src.experiments.experiment1_partition_compatibility analyze \
  --discovery-results "$MHAR_OUTPUT_ROOT/run/discovery_results.jsonl" \
  --output-dir "$MHAR_OUTPUT_ROOT/run/analysis" \
  --wandb-mode online \
  --wandb-project "$MHAR_WANDB_PROJECT" \
  --wandb-group "$MHAR_WANDB_GROUP" \
  --wandb-run-name mhar-exp1-1b-h4-discovery-analysis

"$MHAR_PYTHON_BIN" -m src.experiments.experiment1_partition_compatibility evaluate \
  "${COMMON_EVAL_ARGS[@]}" \
  --output-dir "$MHAR_OUTPUT_ROOT/run" \
  --split confirmation \
  --discovery-results "$MHAR_OUTPUT_ROOT/run/discovery_results.jsonl" \
  --wandb-run-name mhar-exp1-1b-h4-confirmation

"$MHAR_PYTHON_BIN" -m src.experiments.experiment1_partition_compatibility analyze \
  --discovery-results "$MHAR_OUTPUT_ROOT/run/discovery_results.jsonl" \
  --confirmation-results "$MHAR_OUTPUT_ROOT/run/confirmation_results.jsonl" \
  --output-dir "$MHAR_OUTPUT_ROOT/run/final-analysis" \
  --wandb-mode online \
  --wandb-project "$MHAR_WANDB_PROJECT" \
  --wandb-group "$MHAR_WANDB_GROUP" \
  --wandb-run-name mhar-exp1-1b-h4-final-analysis

echo "Experiment 1 complete: $MHAR_OUTPUT_ROOT/run/final-analysis"
