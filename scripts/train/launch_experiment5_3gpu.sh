#!/usr/bin/env bash
set -euo pipefail
MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-/root/autodl-tmp/venvs/mhar-stage-b/bin/python}"
MHAR_EXP5_ROOT="${MHAR_EXP5_ROOT:-/root/autodl-tmp/experiment5}"
MHAR_EXP5_PARENT="${MHAR_EXP5_PARENT:-/root/autodl-tmp/experiment3/checkpoints/h16/seed-43/step-1500}"
MHAR_EXP5_ARTIFACT="${MHAR_EXP5_ARTIFACT:-/root/autodl-tmp/experiment3/fixed_eval.pt}"
export HF_HOME="${MHAR_HF_HOME:-/root/autodl-tmp/huggingface}"
export WANDB_DIR="${MHAR_WANDB_DIR:-/root/autodl-tmp/wandb}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$MHAR_REPO_DIR"
[[ ! -e "$MHAR_EXP5_ROOT" ]] || { echo 'refusing existing Exp5 output' >&2; exit 1; }
if screen -ls | grep -q '[.]mhar-exp5-controller[[:space:]]'; then
  echo 'Exp5 controller already exists' >&2; exit 1
fi
screen -L -Logfile "${MHAR_EXP5_ROOT}-controller.log" -dmS mhar-exp5-controller \
  "$MHAR_PYTHON_BIN" -m src.experiments.experiment5_controller run \
  --root "$MHAR_EXP5_ROOT" --parent "$MHAR_EXP5_PARENT" --artifact "$MHAR_EXP5_ARTIFACT"
echo 'Exp5 controller dispatched; inspect its log to confirm preflight and startup.'
