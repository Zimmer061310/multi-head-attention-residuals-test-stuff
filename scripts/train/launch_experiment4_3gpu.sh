#!/usr/bin/env bash
set -euo pipefail

MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_BRANCH_MANIFEST="${MHAR_BRANCH_MANIFEST:?set MHAR_BRANCH_MANIFEST}"
MHAR_PARENT_ROOT="${MHAR_PARENT_ROOT:?set MHAR_PARENT_ROOT}"
MHAR_LOG_DIR="${MHAR_LOG_DIR:-/root/autodl-tmp/experiment4/logs}"
MHAR_MASTER_PORT_BASE="${MHAR_MASTER_PORT_BASE:-29900}"

roles=(predicted-good predicted-bad unchanged)
for index in "${!roles[@]}"; do
  role="${roles[$index]}"
  session="mhar-exp4-${role}"
  parent="$MHAR_PARENT_ROOT/$role/step-1500"
  [[ -d "$parent" ]] || { echo "missing parent copy: $parent" >&2; exit 1; }
  if screen -ls | grep -q "[.]${session}[[:space:]]"; then
    echo "refusing duplicate screen: $session" >&2
    exit 1
  fi
done

mkdir -p "$MHAR_LOG_DIR"
for index in "${!roles[@]}"; do
  role="${roles[$index]}"
  session="mhar-exp4-${role}"
  screen -L -Logfile "$MHAR_LOG_DIR/$role.log" -dmS "$session" \
    env CUDA_VISIBLE_DEVICES="$index" \
      MHAR_MASTER_PORT="$((MHAR_MASTER_PORT_BASE + index))" \
      MHAR_REPO_DIR="$MHAR_REPO_DIR" \
      MHAR_BRANCH_MANIFEST="$MHAR_BRANCH_MANIFEST" \
      MHAR_PARENT_CHECKPOINT="$MHAR_PARENT_ROOT/$role/step-1500" \
      MHAR_ROLE="$role" \
      "$MHAR_REPO_DIR/scripts/train/run_experiment4_branch.sh"
  echo "launched $role on GPU $index in $session"
done
