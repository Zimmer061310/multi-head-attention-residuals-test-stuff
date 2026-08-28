#!/usr/bin/env bash
set -euo pipefail

MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-python3}"
MHAR_CONTROLLER_REPO_DIR="${MHAR_CONTROLLER_REPO_DIR:-${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
MHAR_TRAINING_REPO_DIR="${MHAR_TRAINING_REPO_DIR:-$MHAR_CONTROLLER_REPO_DIR}"
MHAR_SEED="${MHAR_SEED:?set MHAR_SEED to 42, 43, or 44}"
MHAR_MASTER_PORT="${MHAR_MASTER_PORT:-29600}"
MHAR_TARGET_STEP="${MHAR_TARGET_STEP:-3000}"
MHAR_OUTPUT_ROOT="${MHAR_OUTPUT_ROOT:-/root/autodl-tmp/experiment3/checkpoints/h16}"
MHAR_DATA_FILES="${MHAR_DATA_FILES:-/root/autodl-tmp/datasets/fineweb-edu-sample-10BT/train/*.parquet}"
MHAR_HF_HOME="${MHAR_HF_HOME:-/root/autodl-tmp/huggingface}"
MHAR_WANDB_DIR="${MHAR_WANDB_DIR:-/root/autodl-tmp/wandb}"
MHAR_OUTPUT_DIR="$MHAR_OUTPUT_ROOT/seed-$MHAR_SEED"

case "$MHAR_SEED" in 42|43|44) ;; *) echo "seed is not preregistered" >&2; exit 2;; esac
mkdir -p "$MHAR_OUTPUT_DIR" "$MHAR_HF_HOME" "$MHAR_WANDB_DIR"
export HF_HOME="$MHAR_HF_HOME"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_DIR="$MHAR_WANDB_DIR"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

RESUME_ARGS=()
if [[ $# -gt 1 ]]; then echo "usage: $0 [checkpoint-directory]" >&2; exit 2; fi
if [[ $# -eq 1 ]]; then RESUME_ARGS=(--resume_from "$1"); fi

STOP_ARGS=()
if grep -q -- "--stop_after_step" "$MHAR_TRAINING_REPO_DIR/src/training/train_scratch.py"; then
  STOP_ARGS=(--stop_after_step "$MHAR_TARGET_STEP")
fi

cd "$MHAR_TRAINING_REPO_DIR"
exec "$MHAR_PYTHON_BIN" -m torch.distributed.run --nproc_per_node=1 \
  --master_port "$MHAR_MASTER_PORT" \
  --module src.training.train_scratch \
  --mode full_mh --attnres_heads 16 \
  --hidden_size 1280 --num_layers 36 --num_heads 16 --num_kv_heads 8 \
  --intermediate_size 5120 --seq_len 1024 \
  --steps 20000 "${STOP_ARGS[@]}" \
  --batch_size 4 --grad_accum 8 --expected_global_batch 32 \
  --lr 5e-4 --lr_min 5e-5 --warmup 1000 --max_norm 1.0 --seed "$MHAR_SEED" \
  --dataset HuggingFaceFW/fineweb-edu --dataset_name sample-10BT \
  --dataset_revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --data_files "$MHAR_DATA_FILES" \
  --tokenizer Qwen/Qwen3-0.6B \
  --tokenizer_revision c1899de289a04d12100db370d81485cdf75e47ca \
  --fused --grad_ckpt --save_every 500 --keep_last 2 \
  --keep_steps "1000,1500,2000,${MHAR_TARGET_STEP}" --eval_every 0 --log_every 10 \
  --out_dir "$MHAR_OUTPUT_DIR" \
  --wandb_project "MHAR Stuff" \
  --wandb_group mhar-exp3-boundary-learnability-h16-probes \
  --wandb_required --run_name "mhar-exp3-h16-seed${MHAR_SEED}" \
  "${RESUME_ARGS[@]}"
