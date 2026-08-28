#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 SCREEN_NAME CHECKPOINT_DIR EXPECTED_STEP" >&2
  exit 2
fi

target_screen="$1"
checkpoint_dir="$2"
expected_step="$3"
poll_seconds="${MHAR_POLL_SECONDS:-10}"

while screen -ls | grep -q "[.]${target_screen}[[:space:]]"; do
  if [[ -s "$checkpoint_dir/training_manifest.json" \
        && -s "$checkpoint_dir/training_state.pt" \
        && -s "$checkpoint_dir/model.safetensors" ]]; then
    observed_step="$(python3 - "$checkpoint_dir/training_manifest.json" <<'PY'
import json
import pathlib
import sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["global_step"])
PY
)"
    if [[ "$observed_step" == "$expected_step" ]]; then
      screen -S "$target_screen" -X quit
      printf 'atomically stopped %s after complete step %s checkpoint\n' \
        "$target_screen" "$expected_step"
      exit 0
    fi
  fi
  sleep "$poll_seconds"
done

if [[ -s "$checkpoint_dir/training_manifest.json" ]]; then
  observed_step="$(python3 - "$checkpoint_dir/training_manifest.json" <<'PY'
import json
import pathlib
import sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["global_step"])
PY
)"
  [[ "$observed_step" == "$expected_step" ]] && exit 0
fi
echo "$target_screen exited before a complete step-$expected_step checkpoint" >&2
exit 1
