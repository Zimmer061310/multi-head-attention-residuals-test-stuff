# Experiment 5 three-GPU operation

Use server `root@connect.bjb2.seetacloud.com:45694` when powered on. Credentials
must not be committed. Do not start the obsolete Experiment 3/4 controllers.

## Deploy and preflight

1. Push the tested Exp5 commit first. Transfer that exact commit to a clean
   detached worktree (suggest `/root/mhar-experiment5`). If GitHub access on the
   server fails, transfer a git bundle from the local verified branch and
   create the worktree from its commit. Do not modify old training worktrees.
2. Use `/root/autodl-tmp/venvs/mhar-stage-b/bin/python`. Check torch, Transformers,
   datasets, safetensors, NumPy, matplotlib, W&B imports; W&B authenticated to the
   existing entity. Do not print authentication tokens or persist new secrets.
3. Confirm three free RTX5090 GPUs; no other experiment processes; at least
   180 GiB free on `/root/autodl-tmp`; exact parent and fixed artifact present.
   Read the hash-pinned Exp3 parent manifest; data shards must match it.
4. Preserve the existing offline tokenizer cache. The launcher sets offline
   tokenizer flags to avoid the previous blocked Hugging Face API probe.

```bash
cd /root/mhar-experiment5
bash scripts/train/launch_experiment5_3gpu.sh
```

This dispatches `mhar-exp5-controller`. Inspect
`/root/autodl-tmp/experiment5-controller.log`; dispatch is not proof of startup.
Worker logs are `experiment5/logs/step0-{role}.log` then
`train-eval-{role}.log`, roles predicted-good/predicted-bad/unchanged on GPUs
0/1/2. The controller waits for **all** step0 parity checks before training.

Create a monitoring heartbeat only once the live launch is verified. Check
every 10–15 minutes during this short experiment; notify meaningful milestones
or failures. No silent restarts. Read `FAILED.json` on error. Failed/partial
experiments must not trigger shutdown or accepted analysis.

## Completion and backup

After `READY_FOR_BACKUP.json` exists:

1. Read all 24 measurements and `analysis/FINAL_REPORT.md`. Check step0 agreement,
   all seven protected trained snapshots per role, complete final state at1600,
   actual W&B upload-complete marker and all four W&B run links. Never claim
   a held-out curve is training loss or an insignificant gap is equivalent.
2. Copy `branch_manifest.json`, `step0_gate.json`, `measurements/`, `analysis/`,
   and completion markers to **local** `results/experiment5/`. Verify file hashes
   against the server. Do not copy or delete checkpoint trees for this backup.
3. Run focused tests, commit result-only changes and push. Directly verify
   GitHub branch head at the result commit. Leave the execution worktree pinned.
4. Acknowledge using the full verified result commit and locally verified
   summary SHA-256 (substitute actual values):

```bash
/root/autodl-tmp/venvs/mhar-stage-b/bin/python -m src.experiments.experiment5_controller acknowledge \
  --root /root/autodl-tmp/experiment5 \
  --manifest /root/autodl-tmp/experiment5/branch_manifest.json \
  --pushed-commit VERIFIED_40_CHARACTER_COMMIT \
  --summary-sha256 VERIFIED_LOCAL_SUMMARY_SHA256
```

The controller verifies the acknowledgment matches these results, waits ten
minutes, rechecks completion and GPU idleness, then shuts down. Until backup is
acknowledged it stays on; monitor promptly to avoid unnecessary billing. Verify
provider power status if available; an unreachable SSH endpoint alone is only
consistent with shutdown, not independent proof of billing state. Close the
heartbeat after completion/shutdown reporting.
