#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 PHASE WORKER_INDEX" >&2
  exit 2
fi

MHAR_PHASE="$1"
MHAR_WORKER_INDEX="$2"
: "${MHAR_REPO_DIR:?must be set}"
: "${MHAR_PYTHON_BIN:?must be set}"
: "${MHAR_EXP10_ROOT:?must be set}"
: "${MHAR_EXP10_CHECKPOINT:?must be set}"
: "${MHAR_EXP10_ARTIFACT:?must be set}"

exec "$MHAR_PYTHON_BIN" -m src.experiments.experiment10_per_group_contribution evaluate \
  --phase "$MHAR_PHASE" \
  --worker-index "$MHAR_WORKER_INDEX" \
  --worker-count 2 \
  --manifest "$MHAR_EXP10_ROOT/condition_manifest.json" \
  --checkpoint "$MHAR_EXP10_CHECKPOINT" \
  --artifact "$MHAR_EXP10_ARTIFACT" \
  --output-root "$MHAR_EXP10_ROOT" \
  --device cuda \
  --dtype bf16 \
  --batch-size 1
