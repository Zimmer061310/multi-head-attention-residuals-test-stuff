#!/usr/bin/env bash
set -euo pipefail

MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_LOG_DIR="${MHAR_LOG_DIR:-/root/autodl-tmp/experiment3/logs/h16-probes}"
command -v screen >/dev/null || { echo "screen is required" >&2; exit 1; }
mkdir -p "$MHAR_LOG_DIR"

seeds=(42 43 44)
for index in "${!seeds[@]}"; do
  seed="${seeds[$index]}"
  session="mhar-exp3-h16-seed${seed}"
  if screen -ls | grep -q "[.]${session}[[:space:]]"; then
    echo "refusing to duplicate $session" >&2; exit 1
  fi
  screen -L -Logfile "$MHAR_LOG_DIR/seed-${seed}.log" -dmS "$session" \
    env CUDA_VISIBLE_DEVICES="$index" MHAR_REPO_DIR="$MHAR_REPO_DIR" MHAR_SEED="$seed" \
      "$MHAR_REPO_DIR/scripts/train/run_experiment3_h16_probe_seed.sh"
  echo "launched H16 seed $seed on GPU $index in $session"
done
