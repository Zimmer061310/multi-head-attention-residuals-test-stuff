#!/usr/bin/env bash
set -euo pipefail

MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_VENV_DIR="${MHAR_VENV_DIR:-/root/autodl-tmp/venvs/mhar-experiment3}"
MHAR_BOOTSTRAP_PYTHON="${MHAR_BOOTSTRAP_PYTHON:-python3.12}"
MHAR_EVAL_DIR="${MHAR_EVAL_DIR:-/root/autodl-tmp/datasets/fineweb-edu-sample-10BT/experiment3-eval}"
MHAR_ARTIFACT="${MHAR_ARTIFACT:-/root/autodl-tmp/experiment3/fixed_eval.pt}"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "Experiment 3 GPU setup requires Linux x86_64" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi is missing" >&2
  exit 1
fi

if command -v apt-get >/dev/null; then
  APT=(apt-get)
  if [[ "$(id -u)" != "0" ]]; then APT=(sudo apt-get); fi
  "${APT[@]}" update
  DEBIAN_FRONTEND=noninteractive "${APT[@]}" install -y git gh screen curl ca-certificates
fi

command -v "$MHAR_BOOTSTRAP_PYTHON" >/dev/null || {
  echo "$MHAR_BOOTSTRAP_PYTHON is required" >&2
  exit 1
}
"$MHAR_BOOTSTRAP_PYTHON" -m venv "$MHAR_VENV_DIR"
MHAR_PYTHON_BIN="$MHAR_VENV_DIR/bin/python"
"$MHAR_PYTHON_BIN" -m pip install --upgrade pip wheel setuptools
"$MHAR_PYTHON_BIN" -m pip install -r "$MHAR_REPO_DIR/requirements/experiment3.txt"

if ! gh auth status --hostname github.com >/dev/null 2>&1; then
  [[ -t 0 ]] || { echo "GitHub login required; rerun interactively" >&2; exit 1; }
  gh auth login --hostname github.com --web
fi
gh auth setup-git
if ! "$MHAR_PYTHON_BIN" -c 'import wandb; assert wandb.Api(timeout=30).viewer' >/dev/null 2>&1; then
  [[ -t 0 ]] || { echo "W&B login required; rerun interactively" >&2; exit 1; }
  "$MHAR_VENV_DIR/bin/wandb" login --relogin
fi

MHAR_REPO_DIR="$MHAR_REPO_DIR" MHAR_PYTHON_BIN="$MHAR_PYTHON_BIN" \
MHAR_EVAL_DIR="$MHAR_EVAL_DIR" MHAR_ARTIFACT="$MHAR_ARTIFACT" \
  "$MHAR_REPO_DIR/scripts/setup/prepare_experiment3_eval_artifact.sh"

cd "$MHAR_REPO_DIR"
"$MHAR_PYTHON_BIN" -m unittest -v \
  tests.test_experiment3_router \
  tests.test_experiment3_signal \
  tests.test_experiment3_actionability \
  tests.test_experiment3_cross_seed \
  tests.test_experiment3_landscape \
  tests.test_experiment3_figures \
  tests.test_experiment3_server_setup \
  tests.test_train_resume

MHAR_MIN_GPUS="${MHAR_MIN_GPUS:-1}" "$MHAR_PYTHON_BIN" \
  scripts/setup/preflight_experiment3_server.py \
  --artifact "$MHAR_ARTIFACT" --eval-dir "$MHAR_EVAL_DIR"

echo "Experiment 3 server is prepared. No GPU experiment was launched."
echo "Python: $MHAR_PYTHON_BIN"
echo "Artifact: $MHAR_ARTIFACT"
