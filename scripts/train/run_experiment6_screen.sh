#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <b|m4|c4|g4|m8|c8|g8> [checkpoint-directory]" >&2
  exit 2
fi

MHAR_VARIANT="$1"
MHAR_RESUME_FROM="${2:-}"
MHAR_MODE=""
MHAR_ATTNRES_HEADS=8
MHAR_FUSED=0
MHAR_QKV_GROUPS=""

case "$MHAR_VARIANT" in
  b)  MHAR_MODE=baseline ;;
  m4) MHAR_MODE=full_mh; MHAR_ATTNRES_HEADS=4; MHAR_FUSED=1 ;;
  c4) MHAR_MODE=full_mh; MHAR_ATTNRES_HEADS=4; MHAR_QKV_GROUPS=4; MHAR_FUSED=1 ;;
  g4) MHAR_MODE=baseline; MHAR_QKV_GROUPS=4 ;;
  m8) MHAR_MODE=full_mh; MHAR_ATTNRES_HEADS=8; MHAR_FUSED=1 ;;
  c8) MHAR_MODE=full_mh; MHAR_ATTNRES_HEADS=8; MHAR_QKV_GROUPS=8; MHAR_FUSED=1 ;;
  g8) MHAR_MODE=baseline; MHAR_QKV_GROUPS=8 ;;
  *) echo "unknown Experiment 6 variant: $MHAR_VARIANT" >&2; exit 2 ;;
esac

MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-python3}"
MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_OUTPUT_ROOT="${MHAR_OUTPUT_ROOT:-/root/autodl-tmp/experiment6/screening}"
MHAR_OUTPUT_DIR="${MHAR_OUTPUT_DIR:-$MHAR_OUTPUT_ROOT/$MHAR_VARIANT}"
MHAR_HF_HOME="${MHAR_HF_HOME:-/root/autodl-tmp/huggingface}"
MHAR_WANDB_DIR="${MHAR_WANDB_DIR:-/root/autodl-tmp/wandb}"
MHAR_HF_ENDPOINT="${MHAR_HF_ENDPOINT:-https://hf-mirror.com}"
MHAR_DATA_FILES="${MHAR_DATA_FILES:-/root/autodl-tmp/datasets/fineweb-edu-sample-10BT/train/*.parquet}"
MHAR_WANDB_PROJECT="${MHAR_WANDB_PROJECT:-MHAR Stuff}"
MHAR_WANDB_GROUP="${MHAR_WANDB_GROUP:-mhar-exp6-coupled-qkv-screen-seed42}"
MHAR_MASTER_PORT="${MHAR_MASTER_PORT:-29600}"

mkdir -p "$MHAR_OUTPUT_DIR" "$MHAR_HF_HOME" "$MHAR_WANDB_DIR"
export HF_HOME="$MHAR_HF_HOME"
export HF_ENDPOINT="$MHAR_HF_ENDPOINT"
export WANDB_DIR="$MHAR_WANDB_DIR"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

MHAR_VARIANT_ARGS=()
if [[ "$MHAR_FUSED" == 1 ]]; then
  MHAR_VARIANT_ARGS+=(--fused)
fi
if [[ -n "$MHAR_QKV_GROUPS" ]]; then
  MHAR_VARIANT_ARGS+=(--qkv_groups "$MHAR_QKV_GROUPS")
fi
MHAR_RESUME_ARGS=()
if [[ -n "$MHAR_RESUME_FROM" ]]; then
  MHAR_RESUME_ARGS+=(--resume_from "$MHAR_RESUME_FROM")
fi

cd "$MHAR_REPO_DIR"
exec "$MHAR_PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port "$MHAR_MASTER_PORT" \
  --module src.training.train_scratch \
  --experiment6_variant "$MHAR_VARIANT" \
  --mode "$MHAR_MODE" \
  --attnres_heads "$MHAR_ATTNRES_HEADS" \
  --hidden_size 1280 \
  --num_layers 36 \
  --num_heads 16 \
  --num_kv_heads 8 \
  --intermediate_size 5120 \
  --seq_len 1024 \
  --steps 20000 \
  --stop_after_step 2000 \
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
  --grad_ckpt \
  --save_every 100 \
  --keep_last 2 \
  --keep_steps 500,1000,1500,2000 \
  --eval_every 500 \
  --eval_steps 50 \
  --log_every 10 \
  --out_dir "$MHAR_OUTPUT_DIR" \
  --wandb_project "$MHAR_WANDB_PROJECT" \
  --wandb_group "$MHAR_WANDB_GROUP" \
  --wandb_required \
  --run_name "mhar-exp6-$MHAR_VARIANT-seed42" \
  "${MHAR_VARIANT_ARGS[@]}" \
  "${MHAR_RESUME_ARGS[@]}"
