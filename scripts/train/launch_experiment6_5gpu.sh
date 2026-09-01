#!/usr/bin/env bash
set -euo pipefail

MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-python3}"
MHAR_OUTPUT_ROOT="${MHAR_OUTPUT_ROOT:-/root/autodl-tmp/experiment6/screening}"
MHAR_LOG_DIR="${MHAR_LOG_DIR:-$MHAR_OUTPUT_ROOT/logs}"
MHAR_GPU_IDS="${MHAR_GPU_IDS:-0,1,2,3,4}"
MHAR_WANDB_GROUP="${MHAR_WANDB_GROUP:-mhar-exp6-coupled-qkv-screen-seed42}"

IFS=',' read -r -a GPU_IDS <<< "$MHAR_GPU_IDS"
VARIANTS=(b c4 g4 c8 g8)
if [[ "${#GPU_IDS[@]}" -ne "${#VARIANTS[@]}" ]]; then
  echo "MHAR_GPU_IDS must contain exactly five GPU ids" >&2
  exit 2
fi

mkdir -p "$MHAR_LOG_DIR"
cd "$MHAR_REPO_DIR"
for index in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$index]}"
  gpu="${GPU_IDS[$index]}"
  screen_name="mhar-exp6-$variant"
  output_dir="$MHAR_OUTPUT_ROOT/$variant"
  if screen -list | grep -Fq ".$screen_name"; then
    echo "screen already exists: $screen_name" >&2
    exit 1
  fi
  if [[ -f "$output_dir/training_run_manifest.json" ]]; then
    echo "run output already exists; resume explicitly: $output_dir" >&2
    exit 1
  fi
  screen -L -Logfile "$MHAR_LOG_DIR/$variant.log" -dmS "$screen_name" \
    env CUDA_VISIBLE_DEVICES="$gpu" \
      MHAR_PYTHON_BIN="$MHAR_PYTHON_BIN" \
      MHAR_OUTPUT_ROOT="$MHAR_OUTPUT_ROOT" \
      MHAR_WANDB_GROUP="$MHAR_WANDB_GROUP" \
      MHAR_MASTER_PORT="$((29600 + index))" \
      "$MHAR_REPO_DIR/scripts/train/run_experiment6_screen.sh" "$variant"
  echo "launched $variant on physical GPU $gpu in screen $screen_name"
done

echo "Launched five new Experiment 6 runs. Existing M4/M8 results are not retrained."
