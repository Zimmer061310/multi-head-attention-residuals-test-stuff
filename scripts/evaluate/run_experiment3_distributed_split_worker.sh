#!/usr/bin/env bash
set -euo pipefail

MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-python3}"
MHAR_SEED="${MHAR_SEED:?set MHAR_SEED to 42, 43, or 44}"
MHAR_STEP="${MHAR_STEP:?set MHAR_STEP to 1000, 1500, 2000, or 3000}"
MHAR_SPLIT="${MHAR_SPLIT:?set MHAR_SPLIT to discovery or confirmation}"
MHAR_CHECKPOINT="${MHAR_CHECKPOINT:-/root/autodl-tmp/experiment3/checkpoints/h16/seed-${MHAR_SEED}/step-${MHAR_STEP}}"
MHAR_ARTIFACT="${MHAR_ARTIFACT:-/root/autodl-tmp/experiment3/fixed_eval.pt}"
MHAR_OUTPUT_DIR="${MHAR_OUTPUT_DIR:-/root/autodl-tmp/experiment3/results/seed-${MHAR_SEED}/probes/step-${MHAR_STEP}}"
MHAR_WANDB_ENTITY="${MHAR_WANDB_ENTITY:-zimmer061310-ena}"
MHAR_WANDB_GROUP="${MHAR_WANDB_GROUP:-mhar-exp3-boundary-learnability-seed${MHAR_SEED}}"
MHAR_POLL_SECONDS="${MHAR_POLL_SECONDS:-30}"
MHAR_REMOTE_PORT="${MHAR_REMOTE_PORT:-22}"
MHAR_REMOTE_KEY="${MHAR_REMOTE_KEY:-/root/.ssh/mhar_transfer_ed25519}"
MHAR_REMOTE_PYTHON="${MHAR_REMOTE_PYTHON:-/root/miniconda3/bin/python3}"
MHAR_SYNC_PORT="${MHAR_SYNC_PORT:-$MHAR_REMOTE_PORT}"
MHAR_SYNC_KEY="${MHAR_SYNC_KEY:-$MHAR_REMOTE_KEY}"

case "$MHAR_SEED" in 42|43|44) ;; *) echo "seed is not preregistered" >&2; exit 2;; esac
case "$MHAR_STEP" in 1000|1500|2000|3000) ;; *) echo "step is not preregistered" >&2; exit 2;; esac
case "$MHAR_SPLIT" in discovery|confirmation) ;; *) echo "split is not preregistered" >&2; exit 2;; esac

checkpoint_complete() {
  "$MHAR_PYTHON_BIN" - "$1" "$MHAR_SEED" "$MHAR_STEP" <<'PY'
import json
import pathlib
import sys

checkpoint = pathlib.Path(sys.argv[1])
seed = int(sys.argv[2])
step = int(sys.argv[3])
required = (
    checkpoint / "training_manifest.json",
    checkpoint / "training_state.pt",
    checkpoint / "model.safetensors",
)
if not all(path.is_file() and path.stat().st_size > 0 for path in required):
    raise SystemExit(1)
manifest = json.loads(required[0].read_text(encoding="utf-8"))
if int(manifest["global_step"]) != step:
    raise SystemExit(1)
if int(manifest["run_identity"]["seed"]) != seed:
    raise SystemExit(1)
PY
}

remote_checkpoint_complete() {
  ssh -i "$MHAR_REMOTE_KEY" -p "$MHAR_REMOTE_PORT" \
    -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$MHAR_REMOTE_HOST" \
    "$MHAR_REMOTE_PYTHON" - "$MHAR_REMOTE_CHECKPOINT" "$MHAR_SEED" "$MHAR_STEP" <<'PY'
import json
import pathlib
import sys

checkpoint = pathlib.Path(sys.argv[1])
seed = int(sys.argv[2])
step = int(sys.argv[3])
required = (
    checkpoint / "training_manifest.json",
    checkpoint / "training_state.pt",
    checkpoint / "model.safetensors",
)
if not all(path.is_file() and path.stat().st_size > 0 for path in required):
    raise SystemExit(1)
manifest = json.loads(required[0].read_text(encoding="utf-8"))
if int(manifest["global_step"]) != step:
    raise SystemExit(1)
if int(manifest["run_identity"]["seed"]) != seed:
    raise SystemExit(1)
PY
}

if ! checkpoint_complete "$MHAR_CHECKPOINT"; then
  if [[ -n "${MHAR_REMOTE_HOST:-}" || -n "${MHAR_REMOTE_CHECKPOINT:-}" ]]; then
    : "${MHAR_REMOTE_HOST:?set both MHAR_REMOTE_HOST and MHAR_REMOTE_CHECKPOINT}"
    : "${MHAR_REMOTE_CHECKPOINT:?set both MHAR_REMOTE_HOST and MHAR_REMOTE_CHECKPOINT}"
    echo "waiting for complete remote seed-$MHAR_SEED step-$MHAR_STEP checkpoint"
    until remote_checkpoint_complete; do sleep "$MHAR_POLL_SECONDS"; done

    checkpoint_parent="$(dirname "$MHAR_CHECKPOINT")"
    checkpoint_partial="${MHAR_CHECKPOINT}.partial-${MHAR_SEED}-${MHAR_STEP}"
    mkdir -p "$checkpoint_parent"
    if [[ -e "$checkpoint_partial" ]]; then
      echo "partial checkpoint target already exists: $checkpoint_partial" >&2
      exit 1
    fi
    mkdir "$checkpoint_partial"
    rsync -a --partial --info=progress2 \
      -e "ssh -i $MHAR_REMOTE_KEY -p $MHAR_REMOTE_PORT -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
      "$MHAR_REMOTE_HOST:$MHAR_REMOTE_CHECKPOINT/" "$checkpoint_partial/"
    checkpoint_complete "$checkpoint_partial"
    mv "$checkpoint_partial" "$MHAR_CHECKPOINT"
  else
    echo "waiting for complete local seed-$MHAR_SEED step-$MHAR_STEP checkpoint"
    until checkpoint_complete "$MHAR_CHECKPOINT"; do sleep "$MHAR_POLL_SECONDS"; done
  fi
fi

if [[ -n "${MHAR_WAIT_SCREEN:-}" ]]; then
  while screen -ls | grep -q "[.]${MHAR_WAIT_SCREEN}[[:space:]]"; do
    sleep "$MHAR_POLL_SECONDS"
  done
fi

mkdir -p "$MHAR_OUTPUT_DIR" "${WANDB_DIR:-/root/autodl-tmp/wandb}"
cd "$MHAR_REPO_DIR"
"$MHAR_PYTHON_BIN" scripts/setup/preflight_experiment3_server.py \
  --artifact "$MHAR_ARTIFACT" --artifact-only
"$MHAR_PYTHON_BIN" -m src.experiments.experiment3_signal evaluate \
  --checkpoint "$MHAR_CHECKPOINT" --artifact "$MHAR_ARTIFACT" \
  --seed "$MHAR_SEED" --step "$MHAR_STEP" --split "$MHAR_SPLIT" \
  --output-dir "$MHAR_OUTPUT_DIR" \
  --wandb-entity "$MHAR_WANDB_ENTITY" --wandb-group "$MHAR_WANDB_GROUP" \
  --wandb-mode online

if [[ -n "${MHAR_SYNC_HOST:-}" || -n "${MHAR_SYNC_OUTPUT_DIR:-}" ]]; then
  : "${MHAR_SYNC_HOST:?set both MHAR_SYNC_HOST and MHAR_SYNC_OUTPUT_DIR}"
  : "${MHAR_SYNC_OUTPUT_DIR:?set both MHAR_SYNC_HOST and MHAR_SYNC_OUTPUT_DIR}"
  ssh -i "$MHAR_SYNC_KEY" -p "$MHAR_SYNC_PORT" \
    -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$MHAR_SYNC_HOST" \
    mkdir -p "$MHAR_SYNC_OUTPUT_DIR"
  rsync -a \
    -e "ssh -i $MHAR_SYNC_KEY -p $MHAR_SYNC_PORT -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
    "$MHAR_OUTPUT_DIR/${MHAR_SPLIT}_results.jsonl" \
    "$MHAR_OUTPUT_DIR/${MHAR_SPLIT}_run_manifest.json" \
    "$MHAR_SYNC_HOST:$MHAR_SYNC_OUTPUT_DIR/"
fi

echo "Experiment 3 seed-$MHAR_SEED step-$MHAR_STEP $MHAR_SPLIT complete"
