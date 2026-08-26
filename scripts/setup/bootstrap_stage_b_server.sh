#!/usr/bin/env bash
set -euo pipefail

MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_VENV_DIR="${MHAR_VENV_DIR:-/root/autodl-tmp/venvs/mhar-stage-b}"
MHAR_BOOTSTRAP_PYTHON="${MHAR_BOOTSTRAP_PYTHON:-python3.12}"
MHAR_DATA_DIR="${MHAR_DATA_DIR:-/root/autodl-tmp/datasets/fineweb-edu-sample-10BT/train}"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "Stage B requires Linux x86_64; found $(uname -s) $(uname -m)" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi is missing; choose a CUDA/PyTorch GPU image" >&2
  exit 1
fi

if command -v apt-get >/dev/null; then
  APT=(apt-get)
  if [[ "$(id -u)" != "0" ]]; then
    APT=(sudo apt-get)
  fi
  "${APT[@]}" update
  DEBIAN_FRONTEND=noninteractive "${APT[@]}" install -y git gh screen curl ca-certificates
fi

if ! command -v "$MHAR_BOOTSTRAP_PYTHON" >/dev/null; then
  echo "$MHAR_BOOTSTRAP_PYTHON is missing; set MHAR_BOOTSTRAP_PYTHON to a Python 3.12 executable" >&2
  exit 1
fi

"$MHAR_BOOTSTRAP_PYTHON" -m venv "$MHAR_VENV_DIR"
MHAR_PYTHON_BIN="$MHAR_VENV_DIR/bin/python"
"$MHAR_PYTHON_BIN" -m pip install --upgrade pip wheel setuptools
"$MHAR_PYTHON_BIN" -m pip install -r "$MHAR_REPO_DIR/requirements/stage-b.txt"

if ! gh auth status --hostname github.com >/dev/null 2>&1; then
  if [[ ! -t 0 ]]; then
    echo "GitHub authentication is missing; rerun interactively and complete gh auth login" >&2
    exit 1
  fi
  gh auth login --hostname github.com --git-protocol https --web
fi
gh auth setup-git

if ! "$MHAR_PYTHON_BIN" -c 'import wandb; assert wandb.Api(timeout=30).viewer' >/dev/null 2>&1; then
  if [[ ! -t 0 ]]; then
    echo "W&B authentication is missing; rerun interactively and complete wandb login" >&2
    exit 1
  fi
  "$MHAR_VENV_DIR/bin/wandb" login --relogin
fi

MHAR_DATA_DIR="$MHAR_DATA_DIR" "$MHAR_PYTHON_BIN" \
  "$MHAR_REPO_DIR/scripts/setup/download_stage_b_data.py" --output-dir "$MHAR_DATA_DIR"

MHAR_DATA_DIR="$MHAR_DATA_DIR" "$MHAR_PYTHON_BIN" \
  "$MHAR_REPO_DIR/scripts/setup/preflight_stage_b_server.py"

cd "$MHAR_REPO_DIR"
MHAR_RUN_MODEL_INTEGRATION=1 "$MHAR_PYTHON_BIN" -m unittest -v \
  tests.test_experiment2_stage_b_screening tests.test_train_resume

echo "Stage B server setup complete"
echo "Python: $MHAR_PYTHON_BIN"
