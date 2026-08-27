#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <destination-host> <destination-port> <ssh-key>" >&2
  exit 2
fi

DESTINATION_HOST="$1"
DESTINATION_PORT="$2"
SSH_KEY="$3"
SOURCE_OUTPUT_ROOT="${MHAR_OUTPUT_ROOT:-/root/autodl-tmp/experiment2/stage-b-screening}"
DESTINATION_OUTPUT_ROOT="${MHAR_DEST_OUTPUT_ROOT:-/root/autodl-tmp/experiment2/stage-b-screening}"
RSYNC_SSH="ssh -i $SSH_KEY -o BatchMode=yes -o StrictHostKeyChecking=yes -p $DESTINATION_PORT"
DESTINATION="root@$DESTINATION_HOST"

ssh -i "$SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -p "$DESTINATION_PORT" "$DESTINATION" \
  "mkdir -p /root/venvs /root/autodl-tmp/venvs /root/autodl-tmp/datasets \
    /root/autodl-tmp/huggingface /root/autodl-tmp/wandb \
    '$DESTINATION_OUTPUT_ROOT/logs'; \
   ln -sfn /root/venvs/mhar-stage-b /root/autodl-tmp/venvs/mhar-stage-b"

rsync -aH --partial --info=progress2 -e "$RSYNC_SSH" \
  /root/mhar-experiment/ "$DESTINATION:/root/mhar-experiment/"
rsync -aH --partial --info=progress2 -e "$RSYNC_SSH" \
  /root/mhar-training-81ff305/ "$DESTINATION:/root/mhar-training-81ff305/"
rsync -aH --partial --info=progress2 -e "$RSYNC_SSH" \
  /root/autodl-tmp/venvs/mhar-stage-b/ \
  "$DESTINATION:/root/venvs/mhar-stage-b/"
rsync -aH --partial --info=progress2 -e "$RSYNC_SSH" \
  /root/autodl-tmp/datasets/fineweb-edu-sample-10BT/ \
  "$DESTINATION:/root/autodl-tmp/datasets/fineweb-edu-sample-10BT/"
rsync -aH --partial --info=progress2 -e "$RSYNC_SSH" \
  /root/autodl-tmp/huggingface/ \
  "$DESTINATION:/root/autodl-tmp/huggingface/"

if [[ -f /root/.netrc ]]; then
  rsync -a --partial -e "$RSYNC_SSH" /root/.netrc "$DESTINATION:/root/.netrc"
fi

for variant in h16 h8 h4; do
  ssh -i "$SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=yes \
    -p "$DESTINATION_PORT" "$DESTINATION" \
    "mkdir -p '$DESTINATION_OUTPUT_ROOT/$variant'"
  rsync -aH --partial --info=progress2 -e "$RSYNC_SSH" \
    "$SOURCE_OUTPUT_ROOT/$variant/step-1500" \
    "$SOURCE_OUTPUT_ROOT/$variant/training_run_manifest.json" \
    "$DESTINATION:$DESTINATION_OUTPUT_ROOT/$variant/"
done

rsync -aH --partial --info=progress2 -e "$RSYNC_SSH" \
  "$SOURCE_OUTPUT_ROOT/fixed_eval.pt" \
  "$SOURCE_OUTPUT_ROOT/fixed_eval.pt.manifest.json" \
  "$DESTINATION:$DESTINATION_OUTPUT_ROOT/"

echo "TRANSFER_COMPLETE"
