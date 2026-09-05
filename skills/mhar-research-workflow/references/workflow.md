# MHAR experiment workflow

Use only the section relevant to the current task. Explicit user instructions
and a frozen experiment plan override these defaults.

## 1. Plan and preregister

Write the plan before architecture code or GPU use. Put it in
`docs/plans/experiment-N.md` and define:

- the single causal/scientific question;
- accepted evidence from earlier experiments and what remains unknown;
- exact architecture graph, tensor shapes, head/group ordering, intervention
  site, endpoints, and what remains unchanged;
- frozen run matrix with stable IDs and duplicate/missing-run rejection;
- initialization, data order, optimizer, schedule, seeds, checkpoints, and
  stopping point;
- primary baseline and signed contrasts;
- discovery-only selection followed by an immutable selection manifest before
  confirmation;
- uncertainty method, seeds, practical margins, gates, result classifications,
  and interpretation limits;
- required machine-readable outputs, figures, W&B records, and shutdown rules.

Do not add multi-seed, longer-training, or adjacent architecture work unless
the user authorizes it. State clearly what the plan does not authorize.

## 2. Isolate and implement

Start from verified remote `main` and create an experiment-specific branch.
Keep the architecture opt-in and isolated from legacy behavior.

Preferred repository layout:

```text
docs/plans/experiment-N.md
configs/experimentN/protocol.json or screening.json
src/experiments/experimentN_*.py
scripts/setup/*experimentN*
scripts/train/*experimentN*
scripts/evaluate/*experimentN*
tests/test_experimentN_*.py
figures/gen_fig_experimentN_*.py
results/experimentN/FINAL_REPORT.md
```

Modify shared training/model code only at a narrow, opt-in integration point
when an experiment-only wrapper cannot faithfully implement the graph. Prove
legacy modes are unchanged.

Before GPU launch, test meaningful invariants:

- intended and forbidden graph influence;
- mapping, ordering, shapes, dtype/device behavior, and endpoint equivalence;
- matched initialization and parameter/MAC accounting;
- gradients and optimizer parameter coverage;
- checkpoint save/resume and exact identity validation;
- incomplete, duplicate, dirty, and hash-mismatch rejection;
- focused experiment tests plus relevant legacy regressions.

Commit and push the verified implementation before rental or remote execution.

## 3. Preflight and launch

GPU use requires explicit authorization. Before starting paid work, verify:

- server identity, GPU count/type, disk capacity, and expected runtime;
- clean pinned execution worktree and exact source commit;
- immutable data/tokenizer revisions and evaluation artifact SHA-256;
- dependency environment, GitHub access, W&B authentication, and safe storage;
- run-matrix/protocol hash and unique output roots;
- no conflicting GPU processes or duplicate active runs;
- atomic resumable checkpoint cadence appropriate to shutdown risk;
- persistent controller/worker names and fail-closed completion markers.

Never place passwords or tokens in repository files, logs, process arguments,
or reports. Do not delete checkpoints merely to create space without explicit
scope and a verified backup.

## 4. Monitor and recover

Report meaningful milestones, failures, changed finish estimates, gate
decisions, or required user action. Stay quiet when healthy work is merely
progressing unless the user asks for periodic reports.

Preserve the scientific run:

- do not change recipe, source, data order, thresholds, or accepted results;
- do not silently restart a failed run;
- diagnose startup-only failures and prove no optimizer step or accepted
  measurement occurred before a corrected relaunch;
- resume only from a verified atomic checkpoint with matching optimizer,
  scheduler, RNG, seed, step, and data position;
- record watcher failures, overshoots, provider shutdowns, and manual actions;
- use idle GPUs only for ready, preregistered, non-duplicate work.

When hardware changes, re-run preflight and verify transferred files by hash.

## 5. Analyze without leakage

Keep discovery and confirmation distinct. If discovery selects a candidate,
write and hash an immutable selection manifest before reading confirmation.
Never revise the candidate, threshold, metric, or classification after seeing
confirmation.

Use token-weighted held-out NLL as the default primary LM metric when the plan
does not specify otherwise. Preserve per-sequence losses and use paired
per-sequence bootstrap intervals for matched fixed examples. State that these
intervals do not measure training-seed uncertainty.

Place the true baseline at zero in intervention figures. Retain intervention
versus intervention contrasts as diagnostics unless the plan makes them
primary. Distinguish:

- descriptive association from causal intervention;
- frozen ablation from retrained architecture performance;
- parameters/MAC/FLOP proxies from measured throughput, latency, and memory;
- single-seed screening from convergence or replication claims.

Generate figures from committed JSON/CSV artifacts with deterministic scripts,
not manual W&B editing. Save both 300-DPI PNG and vector PDF versions. W&B is
for tracking, tables, metrics, and links; repository artifacts remain the
reproducible record.

## 6. Publish and close

Require complete conditions and explicit success markers before closure.
Create `results/experimentN/FINAL_REPORT.md` containing:

- question, frozen protocol, result classification, and primary numbers;
- confidence intervals, gate/selection decisions, figures, and W&B links;
- limitations, negative results, and operational deviations;
- exact commits and artifact/checkpoint identities when relevant.

Then:

1. copy compact artifacts into the repository and verify hashes;
2. run focused and relevant regression tests;
3. commit and push a result-only update;
4. verify the remote branch directly;
5. update the root README with the accepted result and figures;
6. merge or fast-forward into `main` only after verification;
7. verify remote `main` at the expected commit;
8. delete the experiment branch only when it is identical to merged `main`;
9. preserve unrelated local files and user changes;
10. release paid compute only after backup/push acknowledgment, a grace period,
    and GPU-idleness checks.

SSH unreachability after shutdown is consistent with poweroff but is not, by
itself, independent provider billing confirmation. Refuse shutdown if work is
incomplete, hashes mismatch, publication failed, scientific code is dirty, or
GPU processes remain active.
