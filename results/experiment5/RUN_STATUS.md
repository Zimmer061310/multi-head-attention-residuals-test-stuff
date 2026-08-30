# Experiment 5 — verified launch, 2026-08-30

Status at 12:37 CST: step0 reproduction passed; all three training processes
restored the seed43 step1500 state and are replay-skipping the original 48,000
packed chunks before their next optimizer update. This is a launch record,
not the completed washout result.

## Execution and tracking

- Server: three RTX5090 GPUs, SSH port45694; controller `mhar-exp5-controller`.
- Pinned worktree: `/root/mhar-experiment5-run`, commit
  `8e763c25559ef76b45549049f7443925d66460e7`.
- Output root: `/root/autodl-tmp/experiment5`.
- Frozen branch-manifest SHA-256:
  `70200b6cc0a3e10b7ad9bd5724518296024b080f6a22872f7db3bf6c9d248e24`.
- Parent, fixed artifact, both training shards, seed/step/data counters,
  optimizer state, scheduler and saved CPU/CUDA RNG states were verified.
- 38 focused tests passed both locally and in the actual server environment.
- Monitor heartbeat: `monitor-mhar-experiment-5`, every15 minutes.

| Branch | GPU | Intervention | W&B training |
|---|---:|---|---|
| A | 0 | remove-03 | [v4t00sfq](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/v4t00sfq) |
| B | 1 | remove-13 | [t03sqdoy](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/t03sqdoy) |
| C | 2 | native H16 | [xrp2boi5](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/xrp2boi5) |

## Reproduced fixed-validation baseline

Every aggregate NLL and every per-sequence loss matched Experiment 3 exactly
on both original512-sequence splits: all six maximum-sequence and aggregate
errors were **0.0**. See `step0_gate.json` and `measurements/*-000.json`.

| Split | A NLL | B NLL | C NLL | A−B |
|---|---:|---:|---:|---:|
| Confirmation (primary) | 4.566358858073095 | 4.577415381941511 | 4.5422118584081925 | −0.011056523868416 |
| Discovery (secondary) | 4.615742427512697 | 4.627136916592208 | 4.592444301467254 | −0.011394489079511 |

A/B are both worse than C at step0. The known A−B advantage has been reproduced;
whether it survives training is still unmeasured in this launch record.

## Startup deviation and safeguards

The initial39fc117 attempt failed before loss measurements or optimizer updates
because a hash helper did not accept CLI string paths. The controller failed
closed. No accepted result or checkpoint existed for that attempt. The tested
8e763c2 fix changed only path handling and added a regression test; all scientific
inputs and tolerances remain unchanged. Failed logs/manifest are preserved at
`/root/autodl-tmp/experiment5-startup-failed-39fc117` and the sibling controller
log. The old execution worktree remains untouched.

All three branches stop at1600 while retaining the original20,000-step LR
schedule. Evaluations use snapshots at offsets0/1/2/5/10/20/50/100, never raw
training losses. Shutdown requires successful analysis, a completed W&B artifact
upload, verified local backup and GitHub push acknowledgment, then ten minutes
of grace and a final GPU-idleness check. No new architecture run is authorized.
