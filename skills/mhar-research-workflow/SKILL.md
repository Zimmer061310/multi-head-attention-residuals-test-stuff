---
name: mhar-research-workflow
description: Plan, implement, launch, monitor, analyze, publish, and close rigorous MHAR or related Transformer architecture experiments. Use for experiment plans, GPU runs, fixed evaluations, result reports, W&B tracking, checkpoint recovery, and branch closure in the MHAR research repository.
---

# MHAR Research Workflow

Preserve the scientific question and authorization boundary while moving work
through explicit stages. A plan does not authorize implementation; an
implementation does not authorize a paid GPU launch; a screening result does
not authorize longer training or multi-seed replication.

Before taking substantive action, identify the active stage:

1. plan and preregistration;
2. isolated implementation and local validation;
3. remote preflight and authorized launch;
4. monitoring or recovery;
5. frozen analysis and interpretation;
6. publication, merge, branch cleanup, and shutdown.

Read [references/workflow.md](references/workflow.md) for the stage being
performed. Follow explicit experiment plans, protocol files, user-provided
authorization, hashes, and frozen gates over general defaults in this skill.

## Always preserve

- the stated primary comparison, controls, baseline convention, and scope;
- train/discovery/confirmation separation and frozen selection order;
- source, data, artifact, checkpoint, seed, optimizer, scheduler, RNG, and W&B
  identities;
- negative, inconclusive, and operational-deviation results;
- unrelated user changes in the worktree.

Never silently retry a failed scientific run, tune after confirmation, replace
an accepted checkpoint, overwrite result directories, expose credentials, or
claim end-to-end speed from parameter/MAC proxies alone.

Use reproducible scripts for analysis and figures. Save machine-readable
per-sequence results, 300-DPI PNGs, vector PDFs, and a concise
`FINAL_REPORT.md`. Before releasing paid compute, verify complete results,
backups, hashes, tests, pushed commits, remote visibility, and idle GPUs.
