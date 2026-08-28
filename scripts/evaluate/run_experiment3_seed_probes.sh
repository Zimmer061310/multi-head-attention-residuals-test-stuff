#!/usr/bin/env bash
set -euo pipefail

MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-python3}"
MHAR_SEED="${MHAR_SEED:?set MHAR_SEED}"
MHAR_CHECKPOINT_ROOT="${MHAR_CHECKPOINT_ROOT:-/root/autodl-tmp/experiment3/checkpoints/h16/seed-${MHAR_SEED}}"
MHAR_ARTIFACT="${MHAR_ARTIFACT:-/root/autodl-tmp/experiment3/fixed_eval.pt}"
MHAR_OUTPUT_ROOT="${MHAR_OUTPUT_ROOT:-/root/autodl-tmp/experiment3/results/seed-${MHAR_SEED}}"
MHAR_WANDB_ENTITY="${MHAR_WANDB_ENTITY:-zimmer061310-ena}"
MHAR_GROUP="mhar-exp3-boundary-learnability-seed${MHAR_SEED}"

case "$MHAR_SEED" in 42|43|44) ;; *) echo "seed is not preregistered" >&2; exit 2;; esac
cd "$MHAR_REPO_DIR"
"$MHAR_PYTHON_BIN" scripts/setup/preflight_experiment3_server.py \
  --artifact "$MHAR_ARTIFACT" --artifact-only

for step in 1000 1500 2000 3000; do
  checkpoint="$MHAR_CHECKPOINT_ROOT/step-$step"
  output="$MHAR_OUTPUT_ROOT/probes/step-$step"
  for split in discovery confirmation; do
    "$MHAR_PYTHON_BIN" -m src.experiments.experiment3_signal evaluate \
      --checkpoint "$checkpoint" --artifact "$MHAR_ARTIFACT" \
      --seed "$MHAR_SEED" --step "$step" --split "$split" --output-dir "$output" \
      --wandb-entity "$MHAR_WANDB_ENTITY" --wandb-group "$MHAR_GROUP" --wandb-mode online
  done
done

primary="$MHAR_OUTPUT_ROOT/probes/step-1500"
selection="$MHAR_OUTPUT_ROOT/signal/selection.json"
mkdir -p "$MHAR_OUTPUT_ROOT/signal" "$MHAR_OUTPUT_ROOT/temporal"
"$MHAR_PYTHON_BIN" -m src.experiments.experiment3_signal select \
  --discovery-results "$primary/discovery_results.jsonl" --output "$selection"
"$MHAR_PYTHON_BIN" -m src.experiments.experiment3_signal analyze \
  --discovery-results "$primary/discovery_results.jsonl" \
  --confirmation-results "$primary/confirmation_results.jsonl" \
  --selection "$selection" --output-dir "$MHAR_OUTPUT_ROOT/signal" \
  --wandb-entity "$MHAR_WANDB_ENTITY" --wandb-group "$MHAR_GROUP" --wandb-mode online

temporal_args=()
for step in 1000 1500 2000 3000; do
  temporal_args+=(--discovery "$step=$MHAR_OUTPUT_ROOT/probes/step-$step/discovery_results.jsonl")
  temporal_args+=(--confirmation "$step=$MHAR_OUTPUT_ROOT/probes/step-$step/confirmation_results.jsonl")
done
"$MHAR_PYTHON_BIN" -m src.experiments.experiment3_temporal \
  "${temporal_args[@]}" --output-dir "$MHAR_OUTPUT_ROOT/temporal" \
  --wandb-entity "$MHAR_WANDB_ENTITY" --wandb-group "$MHAR_GROUP" --wandb-mode online

echo "Seed $MHAR_SEED probes complete. Inspect both frozen gates before branching."
