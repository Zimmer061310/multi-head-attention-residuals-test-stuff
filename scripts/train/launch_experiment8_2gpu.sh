#!/usr/bin/env bash
set -euo pipefail

MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-python3}"
MHAR_OUTPUT_ROOT="${MHAR_OUTPUT_ROOT:-/root/autodl-tmp/experiment8/screening}"
MHAR_LOG_DIR="$MHAR_OUTPUT_ROOT/logs"
MHAR_GPU_IDS="${MHAR_GPU_IDS:-0,1}"
IFS=',' read -r -a GPUS <<< "$MHAR_GPU_IDS"
[[ "${#GPUS[@]}" -eq 2 ]] || {
  echo "MHAR_GPU_IDS must contain exactly two ids" >&2
  exit 2
}
mkdir -p "$MHAR_LOG_DIR"
for name in mhar-exp8-hq8 mhar-exp8-bhq8; do
  screen -list | grep -Fq ".$name" && {
    echo "screen already exists: $name" >&2
    exit 1
  }
done
for variant in hq8 bhq8; do
  [[ ! -f "$MHAR_OUTPUT_ROOT/$variant/training_run_manifest.json" ]] || {
    echo "existing output requires explicit review or resume: $variant" >&2
    exit 1
  }
done

screen -L -Logfile "$MHAR_LOG_DIR/hq8.log" -dmS mhar-exp8-hq8 \
  env CUDA_VISIBLE_DEVICES="${GPUS[0]}" MHAR_REPO_DIR="$MHAR_REPO_DIR" \
    MHAR_PYTHON_BIN="$MHAR_PYTHON_BIN" MHAR_OUTPUT_ROOT="$MHAR_OUTPUT_ROOT" \
    MHAR_MASTER_PORT=29800 "$MHAR_REPO_DIR/scripts/train/run_experiment8_screen.sh" hq8
screen -L -Logfile "$MHAR_LOG_DIR/bhq8.log" -dmS mhar-exp8-bhq8 \
  env CUDA_VISIBLE_DEVICES="${GPUS[1]}" MHAR_REPO_DIR="$MHAR_REPO_DIR" \
    MHAR_PYTHON_BIN="$MHAR_PYTHON_BIN" MHAR_OUTPUT_ROOT="$MHAR_OUTPUT_ROOT" \
    MHAR_MASTER_PORT=29801 "$MHAR_REPO_DIR/scripts/train/run_experiment8_screen.sh" bhq8

echo "Launched HQ8 and BHQ8 simultaneously on two GPUs."
