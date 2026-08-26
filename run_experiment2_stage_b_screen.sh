#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <h16|h8|mixed-k2|mixed-k3|mixed-k4-best|mixed-k5|mixed-k4-worst> [checkpoint-directory]" >&2
  exit 2
fi

MHAR_VARIANT="$1"
MHAR_RESUME_FROM="${2:-}"
MHAR_MIXED_PARTITION=""
MHAR_FUSED=0

case "$MHAR_VARIANT" in
  h16)
    MHAR_ATTNRES_HEADS=16
    MHAR_FUSED=1
    ;;
  h8)
    MHAR_ATTNRES_HEADS=8
    MHAR_FUSED=1
    ;;
  mixed-k2)
    MHAR_ATTNRES_HEADS=16
    MHAR_MIXED_PARTITION='0__1__2__3__4__5__6-7__8__9__10__11__12__13__14-15'
    ;;
  mixed-k3)
    MHAR_ATTNRES_HEADS=16
    MHAR_MIXED_PARTITION='0__1__2__3__4__5__6-7__8-9__10__11__12__13__14-15'
    ;;
  mixed-k4-best)
    MHAR_ATTNRES_HEADS=16
    MHAR_MIXED_PARTITION='0__1__2-3__4__5__6-7__8__9-10__11__12__13__14-15'
    ;;
  mixed-k5)
    MHAR_ATTNRES_HEADS=16
    MHAR_MIXED_PARTITION='0__1__2-3__4__5__6-7__8__9-10__11-12__13__14-15'
    ;;
  mixed-k4-worst)
    MHAR_ATTNRES_HEADS=16
    MHAR_MIXED_PARTITION='0-1__2__3__4-5__6__7__8-9__10-11__12__13__14__15'
    ;;
  *)
    echo "unknown Stage B variant: $MHAR_VARIANT" >&2
    exit 2
    ;;
esac

MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-python3}"
MHAR_OUTPUT_ROOT="${MHAR_OUTPUT_ROOT:-/root/autodl-tmp/experiment2/stage-b-screening}"
MHAR_OUTPUT_DIR="${MHAR_OUTPUT_DIR:-$MHAR_OUTPUT_ROOT/$MHAR_VARIANT}"
MHAR_HF_HOME="${MHAR_HF_HOME:-/root/autodl-tmp/huggingface}"
MHAR_WANDB_DIR="${MHAR_WANDB_DIR:-/root/autodl-tmp/wandb}"
MHAR_HF_ENDPOINT="${MHAR_HF_ENDPOINT:-https://hf-mirror.com}"
MHAR_DATA_FILES="${MHAR_DATA_FILES:-/root/autodl-tmp/datasets/fineweb-edu-sample-10BT/train/*.parquet}"
MHAR_WANDB_PROJECT="${MHAR_WANDB_PROJECT:-MHAR Stuff}"
MHAR_WANDB_GROUP="${MHAR_WANDB_GROUP:-mhar-exp2-stage-b-screening-seed42}"

mkdir -p "$MHAR_OUTPUT_DIR" "$MHAR_HF_HOME" "$MHAR_WANDB_DIR"

export HF_HOME="$MHAR_HF_HOME"
export HF_ENDPOINT="$MHAR_HF_ENDPOINT"
export WANDB_DIR="$MHAR_WANDB_DIR"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

MHAR_VARIANT_ARGS=()
if [[ "$MHAR_FUSED" == "1" ]]; then
  MHAR_VARIANT_ARGS+=(--fused)
else
  MHAR_VARIANT_ARGS+=(--mixed_partition "$MHAR_MIXED_PARTITION")
fi

MHAR_RESUME_ARGS=()
if [[ -n "$MHAR_RESUME_FROM" ]]; then
  MHAR_RESUME_ARGS+=(--resume_from "$MHAR_RESUME_FROM")
fi

exec "$MHAR_PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node=1 \
  train_scratch.py \
  --mode full_mh \
  --attnres_heads "$MHAR_ATTNRES_HEADS" \
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
  --grad_ckpt \
  --save_every 500 \
  --keep_last 2 \
  --keep_steps 2000,5000,10000,20000 \
  --eval_every 500 \
  --eval_steps 50 \
  --log_every 10 \
  --out_dir "$MHAR_OUTPUT_DIR" \
  --wandb_project "$MHAR_WANDB_PROJECT" \
  --wandb_group "$MHAR_WANDB_GROUP" \
  --wandb_required \
  --run_name "mhar-exp2-stage-b-$MHAR_VARIANT-seed42" \
  "${MHAR_VARIANT_ARGS[@]}" \
  "${MHAR_RESUME_ARGS[@]}"
