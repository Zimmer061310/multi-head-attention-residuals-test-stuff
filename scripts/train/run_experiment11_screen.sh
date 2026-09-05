#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <run-id> <target-step> [resume-checkpoint]" >&2
  exit 2
fi

MHAR_RUN_ID="$1"
MHAR_TARGET_STEP="$2"
MHAR_RESUME_FROM="${3:-}"
case "$MHAR_RUN_ID" in
  s2q8-l000|s2q8-l010|s2q8-l025|s2q8-l050|gslq8-l000|gslq8-l010|gslq8-l025|gslq8-l050|m8-l100) ;;
  *) echo "unknown Experiment 11 run ID: $MHAR_RUN_ID" >&2; exit 2 ;;
esac
case "$MHAR_TARGET_STEP" in
  500|1000|1500|2000) ;;
  *) echo "target must be a frozen milestone: 500, 1000, 1500, or 2000" >&2; exit 2 ;;
esac

MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-python3}"
MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_OUTPUT_ROOT="${MHAR_EXP11_ROOT:-/root/autodl-tmp/experiment11}"
MHAR_OUTPUT_DIR="$MHAR_OUTPUT_ROOT/training/$MHAR_RUN_ID"
MHAR_ARTIFACT="${MHAR_EXP11_ARTIFACT:-/root/autodl-tmp/experiment2/stage-b-screening/fixed_eval.pt}"
MHAR_DATA_FILES="${MHAR_DATA_FILES:-/root/autodl-tmp/datasets/fineweb-edu-sample-10BT/train/*.parquet}"
MHAR_MASTER_PORT="${MHAR_MASTER_PORT:-29811}"
MHAR_HF_HOME="${MHAR_HF_HOME:-/root/hf-exp11}"
mkdir -p "$MHAR_OUTPUT_DIR" "$MHAR_OUTPUT_ROOT/probes/$MHAR_RUN_ID" "$MHAR_HF_HOME" /root/autodl-tmp/wandb
export HF_HOME="$MHAR_HF_HOME" HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_DIR=/root/autodl-tmp/wandb PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXTRA=()
if [[ -n "$MHAR_RESUME_FROM" ]]; then
  EXTRA+=(--resume_from "$MHAR_RESUME_FROM")
else
  EXTRA+=(
    --experiment11_probe_artifact "$MHAR_ARTIFACT"
    --experiment11_probe_output "$MHAR_OUTPUT_ROOT/probes/$MHAR_RUN_ID/step-0-discovery.json"
  )
fi

cd "$MHAR_REPO_DIR"
exec "$MHAR_PYTHON_BIN" -m torch.distributed.run --nproc_per_node=1 \
  --master_port "$MHAR_MASTER_PORT" --module src.training.train_scratch \
  --experiment11_run "$MHAR_RUN_ID" \
  --mode full_mh --attnres_heads 8 --fused \
  --hidden_size 1280 --num_layers 36 --num_heads 16 --num_kv_heads 8 \
  --intermediate_size 5120 --seq_len 1024 \
  --steps 20000 --stop_after_step "$MHAR_TARGET_STEP" --batch_size 4 --grad_accum 8 \
  --expected_global_batch 32 --lr 5e-4 --lr_min 5e-5 --warmup 1000 --max_norm 1.0 \
  --seed 42 --dataset HuggingFaceFW/fineweb-edu --dataset_name sample-10BT \
  --dataset_revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --data_files "$MHAR_DATA_FILES" --tokenizer Qwen/Qwen3-0.6B \
  --tokenizer_revision c1899de289a04d12100db370d81485cdf75e47ca \
  --grad_ckpt --save_every 100 --keep_last 1 --keep_steps "$MHAR_TARGET_STEP" \
  --reuse_step_checkpoint_as_final \
  --eval_every 500 --eval_steps 50 --log_every 10 --out_dir "$MHAR_OUTPUT_DIR" \
  --wandb_project "MHAR Stuff" \
  --wandb_group "mhar-exp11-soft-query-specialization-seed42" --wandb_required \
  --run_name "mhar-exp11-$MHAR_RUN_ID-seed42" "${EXTRA[@]}"
