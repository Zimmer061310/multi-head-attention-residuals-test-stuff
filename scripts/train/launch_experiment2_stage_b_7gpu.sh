#!/usr/bin/env bash
set -euo pipefail

MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-/root/autodl-tmp/venvs/mhar-stage-b/bin/python}"
MHAR_OUTPUT_ROOT="${MHAR_OUTPUT_ROOT:-/root/autodl-tmp/experiment2/stage-b-screening}"
MHAR_LOG_DIR="${MHAR_LOG_DIR:-$MHAR_OUTPUT_ROOT/logs}"
MHAR_GPU_IDS="${MHAR_GPU_IDS:-0,1,2,3,4,5,6}"
MHAR_WANDB_GROUP="${MHAR_WANDB_GROUP:-mhar-exp2-stage-b-screening-seed42}"

IFS=',' read -r -a GPU_IDS <<< "$MHAR_GPU_IDS"
VARIANTS=(h16 h8 mixed-k2 mixed-k3 mixed-k4-best mixed-k5 mixed-k4-worst)
if [[ "${#GPU_IDS[@]}" -ne "${#VARIANTS[@]}" ]]; then
  echo "MHAR_GPU_IDS must contain exactly seven GPU ids" >&2
  exit 2
fi

mkdir -p "$MHAR_LOG_DIR"
cd "$MHAR_REPO_DIR"

MHAR_MIN_GPUS=7 MHAR_OUTPUT_ROOT="$MHAR_OUTPUT_ROOT" \
  "$MHAR_PYTHON_BIN" scripts/setup/preflight_stage_b_server.py

for index in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$index]}"
  gpu="${GPU_IDS[$index]}"
  screen_name="mhar-stageb-$variant"
  output_dir="$MHAR_OUTPUT_ROOT/$variant"
  log_file="$MHAR_LOG_DIR/$variant.log"
  if screen -list | grep -Fq ".$screen_name"; then
    echo "screen already exists: $screen_name" >&2
    exit 1
  fi
  if [[ -f "$output_dir/training_run_manifest.json" ]]; then
    echo "run output already exists; resume explicitly instead: $output_dir" >&2
    exit 1
  fi
  screen -L -Logfile "$log_file" -dmS "$screen_name" \
    env CUDA_VISIBLE_DEVICES="$gpu" \
      MHAR_PYTHON_BIN="$MHAR_PYTHON_BIN" \
      MHAR_OUTPUT_ROOT="$MHAR_OUTPUT_ROOT" \
      MHAR_WANDB_GROUP="$MHAR_WANDB_GROUP" \
      "$MHAR_REPO_DIR/scripts/train/run_experiment2_stage_b_screen.sh" "$variant"
  echo "launched $variant on physical GPU $gpu in screen $screen_name"
done

echo "All seven Stage B screening runs launched; GPU 7 remains free when using default ids."
