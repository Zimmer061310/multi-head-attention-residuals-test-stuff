#!/usr/bin/env bash
set -euo pipefail

MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-/root/autodl-tmp/venvs/mhar-experiment3/bin/python}"
MHAR_EVAL_DIR="${MHAR_EVAL_DIR:-/root/autodl-tmp/datasets/fineweb-edu-sample-10BT/experiment3-eval}"
MHAR_ARTIFACT="${MHAR_ARTIFACT:-/root/autodl-tmp/experiment3/fixed_eval.pt}"
MHAR_HF_HOME="${MHAR_HF_HOME:-/root/autodl-tmp/huggingface}"

export HF_HOME="$MHAR_HF_HOME"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false

cd "$MHAR_REPO_DIR"
"$MHAR_PYTHON_BIN" scripts/setup/download_stage_b_data.py \
  --dataset experiment3-evaluation --output-dir "$MHAR_EVAL_DIR"

mkdir -p "$(dirname "$MHAR_ARTIFACT")"
if [[ ! -f "$MHAR_ARTIFACT" ]]; then
  "$MHAR_PYTHON_BIN" -m src.experiments.experiment1_partition_compatibility materialize \
    --output "$MHAR_ARTIFACT" \
    --data-files "$MHAR_EVAL_DIR/*.parquet" \
    --tokenizer Qwen/Qwen3-0.6B \
    --tokenizer-revision c1899de289a04d12100db370d81485cdf75e47ca \
    --seed 20260828 \
    --shuffle-buffer 10000 \
    --seq-len 1024 \
    --discovery-sequences 512 \
    --confirmation-sequences 512
fi

"$MHAR_PYTHON_BIN" scripts/setup/preflight_experiment3_server.py \
  --artifact "$MHAR_ARTIFACT" --artifact-only
