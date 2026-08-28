#!/usr/bin/env bash
set -euo pipefail

MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-python3}"
MHAR_SEED="${MHAR_SEED:-42}"
MHAR_STEP="${MHAR_STEP:-2000}"
MHAR_CHECKPOINT="${MHAR_CHECKPOINT:?set MHAR_CHECKPOINT to the native H8 checkpoint}"
MHAR_ARTIFACT="${MHAR_ARTIFACT:-/root/autodl-tmp/experiment3/fixed_eval.pt}"
MHAR_OUTPUT_DIR="${MHAR_OUTPUT_DIR:-/root/autodl-tmp/experiment3/results/landscape-seed-${MHAR_SEED}-step-${MHAR_STEP}}"
MHAR_WANDB_ENTITY="${MHAR_WANDB_ENTITY:-zimmer061310-ena}"

cd "$MHAR_REPO_DIR"
"$MHAR_PYTHON_BIN" scripts/setup/preflight_experiment3_server.py --artifact "$MHAR_ARTIFACT" --artifact-only
for split in discovery confirmation; do
  "$MHAR_PYTHON_BIN" -m src.experiments.experiment3_landscape evaluate \
    --checkpoint "$MHAR_CHECKPOINT" --artifact "$MHAR_ARTIFACT" \
    --seed "$MHAR_SEED" --step "$MHAR_STEP" --split "$split" --output-dir "$MHAR_OUTPUT_DIR" \
    --wandb-entity "$MHAR_WANDB_ENTITY" \
    --wandb-group mhar-exp3-boundary-learnability-landscape --wandb-mode online
done
"$MHAR_PYTHON_BIN" -m src.experiments.experiment3_landscape analyze \
  --discovery-results "$MHAR_OUTPUT_DIR/discovery_results.jsonl" \
  --confirmation-results "$MHAR_OUTPUT_DIR/confirmation_results.jsonl" \
  --output-dir "$MHAR_OUTPUT_DIR" --wandb-entity "$MHAR_WANDB_ENTITY" \
  --wandb-group mhar-exp3-boundary-learnability-landscape --wandb-mode online
