#!/usr/bin/env bash
set -euo pipefail

MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_LOG_DIR="${MHAR_LOG_DIR:-/root/autodl-tmp/experiment3/actionability/logs}"
MHAR_BRANCH_MANIFEST="${MHAR_BRANCH_MANIFEST:?set MHAR_BRANCH_MANIFEST}"
MHAR_PARENT_CHECKPOINT="${MHAR_PARENT_CHECKPOINT:?set MHAR_PARENT_CHECKPOINT}"
MHAR_MASTER_PORT_BASE="${MHAR_MASTER_PORT_BASE:-29700}"

command -v screen >/dev/null || { echo "screen is required" >&2; exit 1; }
mkdir -p "$MHAR_LOG_DIR"

roles=(predicted-good predicted-bad random unchanged)
for index in "${!roles[@]}"; do
  role="${roles[$index]}"
  session="mhar-exp3c-${role}"
  if screen -ls | grep -q "[.]${session}[[:space:]]"; then
    echo "refusing to duplicate active screen: $session" >&2
    exit 1
  fi
  screen -L -Logfile "$MHAR_LOG_DIR/${role}.log" -dmS "$session" \
    env CUDA_VISIBLE_DEVICES="$index" \
      MHAR_MASTER_PORT="$((MHAR_MASTER_PORT_BASE + index))" \
      MHAR_REPO_DIR="$MHAR_REPO_DIR" \
      MHAR_BRANCH_MANIFEST="$MHAR_BRANCH_MANIFEST" \
      MHAR_PARENT_CHECKPOINT="$MHAR_PARENT_CHECKPOINT" \
      MHAR_ROLE="$role" \
      "$MHAR_REPO_DIR/scripts/train/run_experiment3_actionability_branch.sh"
  echo "launched $role on GPU $index in screen $session"
done
