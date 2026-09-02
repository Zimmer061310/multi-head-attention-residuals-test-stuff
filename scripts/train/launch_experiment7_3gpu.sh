#!/usr/bin/env bash
set -euo pipefail

MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-python3}"
MHAR_OUTPUT_ROOT="${MHAR_OUTPUT_ROOT:-/root/autodl-tmp/experiment7/screening}"
MHAR_LOG_DIR="$MHAR_OUTPUT_ROOT/logs"
MHAR_GPU_IDS="${MHAR_GPU_IDS:-0,1,2}"
MHAR_LQ_STAGGER_SECONDS="${MHAR_LQ_STAGGER_SECONDS:-600}"
IFS=',' read -r -a GPUS <<< "$MHAR_GPU_IDS"
[[ "${#GPUS[@]}" -eq 3 ]] || { echo "MHAR_GPU_IDS must contain exactly three ids" >&2; exit 2; }
mkdir -p "$MHAR_LOG_DIR"
for name in mhar-exp7-lq4 mhar-exp7-lq8 mhar-exp7-blq-queue; do
  screen -list | grep -Fq ".$name" && { echo "screen already exists: $name" >&2; exit 1; }
done
for variant in lq4 lq8 blq4 blq8; do
  [[ ! -f "$MHAR_OUTPUT_ROOT/$variant/training_run_manifest.json" ]] || {
    echo "existing output requires explicit resume: $variant" >&2; exit 1; }
done
screen -L -Logfile "$MHAR_LOG_DIR/lq4.log" -dmS mhar-exp7-lq4 \
  env CUDA_VISIBLE_DEVICES="${GPUS[0]}" MHAR_REPO_DIR="$MHAR_REPO_DIR" \
    MHAR_PYTHON_BIN="$MHAR_PYTHON_BIN" MHAR_OUTPUT_ROOT="$MHAR_OUTPUT_ROOT" \
    MHAR_MASTER_PORT=29700 "$MHAR_REPO_DIR/scripts/train/run_experiment7_screen.sh" lq4
screen -L -Logfile "$MHAR_LOG_DIR/blq-queue.log" -dmS mhar-exp7-blq-queue \
  /bin/bash -lc "set -euo pipefail; export CUDA_VISIBLE_DEVICES='${GPUS[2]}' MHAR_REPO_DIR='$MHAR_REPO_DIR' MHAR_PYTHON_BIN='$MHAR_PYTHON_BIN' MHAR_OUTPUT_ROOT='$MHAR_OUTPUT_ROOT'; export MHAR_MASTER_PORT=29702; '$MHAR_REPO_DIR/scripts/train/run_experiment7_screen.sh' blq4; export MHAR_MASTER_PORT=29703; '$MHAR_REPO_DIR/scripts/train/run_experiment7_screen.sh' blq8"
if [[ "$MHAR_LQ_STAGGER_SECONDS" -gt 0 ]]; then
  sleep "$MHAR_LQ_STAGGER_SECONDS"
fi
screen -L -Logfile "$MHAR_LOG_DIR/lq8.log" -dmS mhar-exp7-lq8 \
  env CUDA_VISIBLE_DEVICES="${GPUS[1]}" MHAR_REPO_DIR="$MHAR_REPO_DIR" \
    MHAR_PYTHON_BIN="$MHAR_PYTHON_BIN" MHAR_OUTPUT_ROOT="$MHAR_OUTPUT_ROOT" \
    MHAR_MASTER_PORT=29701 "$MHAR_REPO_DIR/scripts/train/run_experiment7_screen.sh" lq8
echo "Launched LQ4 and LQ8 in parallel; BLQ4 then BLQ8 share the third GPU."
