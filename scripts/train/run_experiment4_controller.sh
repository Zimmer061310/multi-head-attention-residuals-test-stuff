#!/usr/bin/env bash
set -euo pipefail

MHAR_REPO_DIR="${MHAR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MHAR_PYTHON_BIN="${MHAR_PYTHON_BIN:-/root/autodl-tmp/venvs/mhar-stage-b/bin/python}"
MHAR_ROOT="${MHAR_ROOT:-/root/autodl-tmp/experiment4}"
MHAR_SHUTDOWN_GRACE_SECONDS="${MHAR_SHUTDOWN_GRACE_SECONDS:-4200}"
roles=(predicted-good predicted-bad unchanged)

for role in "${roles[@]}"; do
  session="mhar-exp4-$role"
  while screen -ls | grep -q "[.]${session}[[:space:]]"; do
    sleep 60
  done
  final_manifest="$MHAR_ROOT/branches/$role/final/training_manifest.json"
  [[ -f "$final_manifest" ]] || {
    echo "branch $role exited without final manifest; refusing analysis and shutdown" >&2
    exit 1
  }
  "$MHAR_PYTHON_BIN" - "$final_manifest" "$MHAR_ROOT/branches/$role/training_metrics.jsonl" <<'PY'
import json
import pathlib
import sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest["global_step"] != 2000:
    raise SystemExit(f"invalid final step: {manifest['global_step']}")
rows = [json.loads(line) for line in pathlib.Path(sys.argv[2]).read_text(encoding="utf-8").splitlines() if line.strip()]
steps = [int(row["step"]) for row in rows if 1500 < int(row["step"]) <= 2000]
if steps != list(range(1510, 2001, 10)):
    raise SystemExit(f"invalid paired metric steps: {steps}")
PY
done

cd "$MHAR_REPO_DIR"
"$MHAR_PYTHON_BIN" -m src.experiments.experiment4_short_horizon analyze \
  --branch-root "$MHAR_ROOT/branches" \
  --output-dir "$MHAR_ROOT/results" \
  --wandb | tee "$MHAR_ROOT/analysis.log"

[[ -f "$MHAR_ROOT/results/short_horizon_summary.json" ]] || {
  echo "analysis summary missing; refusing shutdown" >&2
  exit 1
}
touch "$MHAR_ROOT/EXPERIMENT4_SUCCESS"
echo "Experiment 4 complete; success-only shutdown grace is ${MHAR_SHUTDOWN_GRACE_SECONDS}s"
sleep "$MHAR_SHUTDOWN_GRACE_SECONDS"
[[ -f "$MHAR_ROOT/EXPERIMENT4_SUCCESS" ]] || exit 1
shutdown -h now
