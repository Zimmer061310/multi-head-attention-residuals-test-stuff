#!/usr/bin/env bash
set -euo pipefail

MHAR_CONTROLLER_REPO_DIR="${MHAR_CONTROLLER_REPO_DIR:-${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
MHAR_TRAINING_REPO_DIR="${MHAR_TRAINING_REPO_DIR:-$MHAR_CONTROLLER_REPO_DIR}"
MHAR_LOG_DIR="${MHAR_LOG_DIR:-/root/autodl-tmp/experiment3/logs/h16-probes}"
MHAR_OUTPUT_ROOT="${MHAR_OUTPUT_ROOT:-/root/autodl-tmp/experiment3/checkpoints/h16}"
MHAR_MASTER_PORT_BASE="${MHAR_MASTER_PORT_BASE:-29600}"
MHAR_TARGET_STEP="${MHAR_TARGET_STEP:-3000}"
command -v screen >/dev/null || { echo "screen is required" >&2; exit 1; }
mkdir -p "$MHAR_LOG_DIR"

seeds=(42 43 44)
for index in "${!seeds[@]}"; do
  seed="${seeds[$index]}"
  session="mhar-exp3-h16-seed${seed}"
  if screen -ls | grep -q "[.]${session}[[:space:]]"; then
    echo "refusing to duplicate $session" >&2; exit 1
  fi
  resume_var="MHAR_RESUME_SEED${seed}"
  resume_checkpoint="${!resume_var:-}"
  resume_args=()
  if [[ -n "$resume_checkpoint" ]]; then resume_args=("$resume_checkpoint"); fi
  master_port=$((MHAR_MASTER_PORT_BASE + index))
  screen -L -Logfile "$MHAR_LOG_DIR/seed-${seed}.log" -dmS "$session" \
    env CUDA_VISIBLE_DEVICES="$index" \
      MHAR_CONTROLLER_REPO_DIR="$MHAR_CONTROLLER_REPO_DIR" \
      MHAR_TRAINING_REPO_DIR="$MHAR_TRAINING_REPO_DIR" MHAR_SEED="$seed" \
      MHAR_MASTER_PORT="$master_port" MHAR_TARGET_STEP="$MHAR_TARGET_STEP" \
      MHAR_OUTPUT_ROOT="$MHAR_OUTPUT_ROOT" \
      "$MHAR_CONTROLLER_REPO_DIR/scripts/train/run_experiment3_h16_probe_seed.sh" \
      "${resume_args[@]}"
  echo "launched H16 seed $seed on GPU $index in $session"
  if ! grep -q -- "--stop_after_step" "$MHAR_TRAINING_REPO_DIR/src/training/train_scratch.py"; then
    stop_session="mhar-exp3-stop-seed${seed}"
    screen -L -Logfile "$MHAR_LOG_DIR/seed-${seed}-stop.log" -dmS "$stop_session" \
      "$MHAR_CONTROLLER_REPO_DIR/scripts/train/watch_experiment3_h16_probe_stop.sh" \
      "$session" "$MHAR_OUTPUT_ROOT/seed-$seed/step-$MHAR_TARGET_STEP" \
      "$MHAR_TARGET_STEP"
    echo "armed atomic step-$MHAR_TARGET_STEP stop watcher in $stop_session"
  fi
done
