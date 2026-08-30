# Experiment 5 shutdown closure — 2026-08-30

All scientific work and the verified result backup completed before shutdown.
The result commit is `3807e66edb35212fc5ee196cce20b60715c6e662` on
`codex/experiment-1-plan`, directly verified on GitHub. The server accepted its
backup acknowledgment at14:11:18 CST and scheduled shutdown for14:21:47 CST.

## Operational failure and recovery

At14:26 CST the server was still reachable. The controller had completed its
grace period and success checks but direct execution of `shutdown` failed with
`OSError: [Errno 8] Exec format error: 'shutdown'`. The exact server error marker
is preserved as `shutdown_failure.json`; this is a shutdown failure, not a
failed training or evaluation run.

Read-only inspection found `/usr/bin/shutdown` was a shell script without a
shebang. Its shutdown action terminates the container supervisor. It also
contained an unrelated Trash-deletion command, which was deliberately not run.

Before recovery, the existing pinned verification code revalidated completed
branches, analysis/upload hashes, measurement hashes, the backup acknowledgment
and the exact published result commit. All GPUs were idle, no experiment jobs
were active, and `/root/mhar-experiment5-run` remained clean at
`8e763c25559ef76b45549049f7443925d66460e7`.

The provider wrapper's shutdown action was then executed narrowly: SIGTERM to
PID819, after verifying its exact command line was
`/bin/supervisord -c /init/supervisor/supervisor.ini`. The server immediately
closed the SSH session. No code, checkpoint, scientific input, accepted result,
or Trash content was changed or deleted.

A subsequent bounded SSH check failed with connection closed (exit255); remote
command execution was no longer available. This is consistent with shutdown,
but is not independent confirmation of the provider's power or billing status.
