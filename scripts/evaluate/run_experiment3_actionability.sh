#!/usr/bin/env bash
set -euo pipefail

MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-python3}"
MHAR_SEED="${MHAR_SEED:?set MHAR_SEED}"
MHAR_ARTIFACT="${MHAR_ARTIFACT:-/root/autodl-tmp/experiment3/fixed_eval.pt}"
MHAR_SEED_ROOT="${MHAR_SEED_ROOT:-/root/autodl-tmp/experiment3/results/seed-${MHAR_SEED}}"
MHAR_PARENT_CHECKPOINT="${MHAR_PARENT_CHECKPOINT:-/root/autodl-tmp/experiment3/checkpoints/h16/seed-${MHAR_SEED}/step-1500}"
MHAR_BRANCH_CHECKPOINT_ROOT="${MHAR_BRANCH_CHECKPOINT_ROOT:-/root/autodl-tmp/experiment3/actionability/seed-${MHAR_SEED}}"
MHAR_WANDB_ENTITY="${MHAR_WANDB_ENTITY:-zimmer061310-ena}"
manifest="$MHAR_SEED_ROOT/actionability/branch_selection.json"
results="$MHAR_SEED_ROOT/actionability/results"
group="mhar-exp3-boundary-learnability-seed${MHAR_SEED}"
replication_args=()
if [[ "$MHAR_SEED" != "42" ]]; then
  MHAR_SEED42_ACTIONABILITY_SUMMARY="${MHAR_SEED42_ACTIONABILITY_SUMMARY:-/root/autodl-tmp/experiment3/results/seed-42/actionability/actionability_results.json}"
  replication_args=(--cross-seed-replication \
    --seed42-actionability-summary "$MHAR_SEED42_ACTIONABILITY_SUMMARY")
fi

cd "$MHAR_REPO_DIR"
"$MHAR_PYTHON_BIN" scripts/setup/preflight_experiment3_server.py --artifact "$MHAR_ARTIFACT" --artifact-only
mkdir -p "$MHAR_SEED_ROOT/actionability"
"$MHAR_PYTHON_BIN" -m src.experiments.experiment3_actionability select \
  --seed "$MHAR_SEED" \
  --discovery-results "$MHAR_SEED_ROOT/probes/step-1500/discovery_results.jsonl" \
  --signal-summary "$MHAR_SEED_ROOT/signal/signal_summary.json" \
  --temporal-summary "$MHAR_SEED_ROOT/temporal/temporal_summary.json" \
  --parent-checkpoint "$MHAR_PARENT_CHECKPOINT" --output "$manifest" \
  "${replication_args[@]}"

echo "Branch manifest frozen at $manifest"
echo "Train the four branches with launch_experiment3_actionability_4gpu.sh, then rerun this script with MHAR_EVALUATE_BRANCHES=1."
if [[ "${MHAR_EVALUATE_BRANCHES:-0}" != "1" ]]; then exit 0; fi

for role in predicted-good predicted-bad random unchanged; do
  "$MHAR_PYTHON_BIN" -m src.experiments.experiment3_actionability evaluate \
    --role "$role" --branch-manifest "$manifest" \
    --checkpoint "$MHAR_BRANCH_CHECKPOINT_ROOT/$role/step-2000" \
    --artifact "$MHAR_ARTIFACT" --output-dir "$results/$role" \
    --wandb-entity "$MHAR_WANDB_ENTITY" --wandb-group "$group" --wandb-mode online
done
"$MHAR_PYTHON_BIN" -m src.experiments.experiment3_actionability analyze \
  --results-root "$results" --training-root "$MHAR_BRANCH_CHECKPOINT_ROOT" \
  --output-dir "$MHAR_SEED_ROOT/actionability" \
  --wandb-entity "$MHAR_WANDB_ENTITY" --wandb-group "$group" --wandb-mode online
