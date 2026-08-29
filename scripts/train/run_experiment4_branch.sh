#!/usr/bin/env bash
set -euo pipefail

MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-/root/autodl-tmp/venvs/mhar-stage-b/bin/python}"
MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_BRANCH_MANIFEST="${MHAR_BRANCH_MANIFEST:?set MHAR_BRANCH_MANIFEST}"
MHAR_PARENT_CHECKPOINT="${MHAR_PARENT_CHECKPOINT:?set MHAR_PARENT_CHECKPOINT}"
MHAR_ROLE="${MHAR_ROLE:?set MHAR_ROLE}"
MHAR_MASTER_PORT="${MHAR_MASTER_PORT:-29900}"
MHAR_OUTPUT_ROOT="${MHAR_OUTPUT_ROOT:-/root/autodl-tmp/experiment4/branches}"
MHAR_HF_HOME="${MHAR_HF_HOME:-/root/autodl-tmp/huggingface}"
MHAR_WANDB_DIR="${MHAR_WANDB_DIR:-/root/autodl-tmp/wandb}"
MHAR_DATA_FILES="${MHAR_DATA_FILES:-/root/autodl-tmp/datasets/fineweb-edu-sample-10BT/train/*.parquet}"

readarray -t fields < <(
  "$MHAR_PYTHON_BIN" - "$MHAR_BRANCH_MANIFEST" "$MHAR_ROLE" <<'PY'
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
row = payload["branches"][sys.argv[2]]
print(payload["seed"])
print(payload["endpoint_step"])
print(row["partition_id"] or "")
PY
)
MHAR_SEED="${fields[0]}"
MHAR_ENDPOINT="${fields[1]}"
MHAR_PARTITION="${fields[2]}"
MHAR_OUTPUT_DIR="$MHAR_OUTPUT_ROOT/$MHAR_ROLE"

mkdir -p "$MHAR_OUTPUT_DIR" "$MHAR_HF_HOME" "$MHAR_WANDB_DIR"
export HF_HOME="$MHAR_HF_HOME"
export WANDB_DIR="$MHAR_WANDB_DIR"
# All pinned tokenizer assets are already present in the local cache. Prevent a
# transformers compatibility probe from making an unnecessary Hub API request;
# AutoDL may block huggingface.co even while W&B remains reachable.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

partition_args=()
if [[ -n "$MHAR_PARTITION" ]]; then
  partition_args=(--mixed_partition "$MHAR_PARTITION")
fi

cd "$MHAR_REPO_DIR"
exec "$MHAR_PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node=1 --master_port "$MHAR_MASTER_PORT" \
  --module src.training.train_scratch \
  --mode full_mh --attnres_heads 16 \
  --hidden_size 1280 --num_layers 36 --num_heads 16 --num_kv_heads 8 \
  --intermediate_size 5120 --seq_len 1024 --steps 20000 \
  --stop_after_step "$MHAR_ENDPOINT" \
  --batch_size 4 --grad_accum 8 --expected_global_batch 32 \
  --lr 5e-4 --lr_min 5e-5 --warmup 1000 --max_norm 1.0 --seed "$MHAR_SEED" \
  --dataset HuggingFaceFW/fineweb-edu --dataset_name sample-10BT \
  --dataset_revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --data_files "$MHAR_DATA_FILES" \
  --tokenizer Qwen/Qwen3-0.6B \
  --tokenizer_revision c1899de289a04d12100db370d81485cdf75e47ca \
  --grad_ckpt --save_every 500 --keep_last 1 --keep_steps "$MHAR_ENDPOINT" \
  --eval_every 0 --log_every 10 \
  --branch_from "$MHAR_PARENT_CHECKPOINT" \
  --branch_manifest "$MHAR_BRANCH_MANIFEST" --branch_role "$MHAR_ROLE" \
  --out_dir "$MHAR_OUTPUT_DIR" \
  --wandb_project "MHAR Stuff" \
  --wandb_group "mhar-exp4-short-horizon-seed${MHAR_SEED}-step1500" \
  --wandb_required --run_name "mhar-exp4-seed${MHAR_SEED}-${MHAR_ROLE}" \
  "${partition_args[@]}"
