#!/usr/bin/env bash
set -euo pipefail

MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-python3}"
MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_OUTPUT_DIR="${MHAR_OUTPUT_DIR:-/root/autodl-tmp/experiment2/checkpoint-1b-h16-fineweb-edu}"
MHAR_HF_HOME="${MHAR_HF_HOME:-/root/autodl-tmp/huggingface}"
MHAR_WANDB_DIR="${MHAR_WANDB_DIR:-/root/autodl-tmp/wandb}"
MHAR_HF_ENDPOINT="${MHAR_HF_ENDPOINT:-https://hf-mirror.com}"
MHAR_DATA_FILES="${MHAR_DATA_FILES:-/root/autodl-tmp/datasets/fineweb-edu-sample-10BT/train/*.parquet}"

mkdir -p "$MHAR_OUTPUT_DIR" "$MHAR_HF_HOME" "$MHAR_WANDB_DIR"

export HF_HOME="$MHAR_HF_HOME"
export HF_ENDPOINT="$MHAR_HF_ENDPOINT"
export WANDB_DIR="$MHAR_WANDB_DIR"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

MHAR_RESUME_ARGS=()
if [[ $# -gt 1 ]]; then
  echo "usage: $0 [checkpoint-directory]" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  MHAR_RESUME_ARGS=(--resume_from "$1")
fi

cd "$MHAR_REPO_DIR"

exec "$MHAR_PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node=1 \
  --module src.training.train_scratch \
  --mode full_mh \
  --attnres_heads 16 \
  --hidden_size 1280 \
  --num_layers 36 \
  --num_heads 16 \
  --num_kv_heads 8 \
  --intermediate_size 5120 \
  --seq_len 1024 \
  --steps 20000 \
  --batch_size 4 \
  --grad_accum 8 \
  --expected_global_batch 32 \
  --lr 5e-4 \
  --lr_min 5e-5 \
  --warmup 1000 \
  --max_norm 1.0 \
  --seed 42 \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset_name sample-10BT \
  --dataset_revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --data_files "$MHAR_DATA_FILES" \
  --tokenizer Qwen/Qwen3-0.6B \
  --tokenizer_revision c1899de289a04d12100db370d81485cdf75e47ca \
  --fused \
  --grad_ckpt \
  --save_every 500 \
  --keep_last 2 \
  --eval_every 500 \
  --eval_steps 50 \
  --log_every 10 \
  --out_dir "$MHAR_OUTPUT_DIR" \
  --wandb_project "MHAR Stuff" \
  --wandb_group mhar-exp2-1b-h16-fineweb-edu \
  --wandb_required \
  --run_name mhar-exp2-1b-h16-fineweb-edu-seed42 \
  "${MHAR_RESUME_ARGS[@]}"
