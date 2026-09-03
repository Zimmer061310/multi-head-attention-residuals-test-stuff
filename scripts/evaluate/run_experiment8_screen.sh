#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "usage: $0 <hq8|bhq8>" >&2; exit 2; }
MHAR_VARIANT="$1"
case "$MHAR_VARIANT" in hq8|bhq8) ;; *) exit 2 ;; esac
MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-python3}"
MHAR_ROOT="${MHAR_ROOT:-/root/autodl-tmp/experiment8/screening}"
MHAR_CHECKPOINT="${MHAR_CHECKPOINT:-$MHAR_ROOT/$MHAR_VARIANT/step-2000}"
MHAR_ARTIFACT="${MHAR_ARTIFACT:-/root/autodl-tmp/experiment2/stage-b-screening/fixed_eval.pt}"
MHAR_RESULTS_ROOT="${MHAR_RESULTS_ROOT:-$MHAR_ROOT/results}"
test -f "$MHAR_CHECKPOINT/training_manifest.json"
test -f "$MHAR_ARTIFACT"
mkdir -p "$MHAR_RESULTS_ROOT/$MHAR_VARIANT"
cd "$MHAR_REPO_DIR"
exec "$MHAR_PYTHON_BIN" -m src.experiments.experiment8_screening evaluate \
  --variant "$MHAR_VARIANT" --checkpoint "$MHAR_CHECKPOINT" --artifact "$MHAR_ARTIFACT" \
  --output-dir "$MHAR_RESULTS_ROOT/$MHAR_VARIANT" --device cuda --dtype bf16 --batch-size 1 \
  --wandb-mode online --wandb-project "MHAR Stuff" \
  --wandb-group mhar-exp8-hybrid-q8-global-kv-screen-seed42-eval \
  --wandb-run-name "mhar-exp8-$MHAR_VARIANT-step2000-eval"
