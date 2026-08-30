# Experiment 5 — How quickly does the frozen boundary advantage wash out?

Status: implementation prepared; GPU execution requires the three-GPU server.
This is a newly authorized experiment, not a retroactive change to Experiment
3's failed stability gate or Experiment 4's training-loss results.

## Question and fixed design

Does the seed-43 step-1500 fixed-validation best-versus-worst merge gap persist
after 1–100 optimizer updates? Experiment 4 compared losses on changing training
batches; it did not measure this held-out decay curve.

| Branch | Fixed intervention | Routing groups |
|---|---|---:|
| A, predicted-good | remove-03, merge atoms 3 and 4 | 15 |
| B, predicted-bad | remove-13, merge atoms 13 and 14 | 15 |
| C, unchanged | native H16 | 16 |

Use the existing discovery-selected pair. Never select a different pair after
examining Experiment 5. A/B have the same width composition and router count;
C is a different-capacity contextual control. All three load the same immutable
parent into independent processes; no physical duplication of its 6.5 GB of
files is necessary. They start with identical weights, optimizer moments,
scheduler, CPU/CUDA RNG states, and packed-data position (48,000 chunks).

Save/evaluate at added updates **0, 1, 2, 5, 10, 20, 50, 100**, corresponding to
absolute steps **1500, 1501, 1502, 1505, 1510, 1520, 1550, 1600**. No training
beyond step1600 or automatic new architecture experiment is authorized here.

## Scientific invariants

Machine-readable constants are in `configs/experiment5/protocol.json`.

- Parent: Experiment 3 seed43 native H16 step1500; SHA-256
  `ece1370fb201f3ba661424bdb7126a4b1d59221bccba6e46332372e63c9faaeb`.
- Evaluation artifact: original Experiment 3 `fixed_eval.pt`; SHA-256
  `1a239dfc65c3b4f9184ccbdc28e9165b4bc452312a2d7360d6b7519fafb1a5af`.
- Primary: all **512 confirmation sequences**; secondary cross-check: all
  **512 discovery sequences**, each 1024 tokens, unchanged order/content.
  These sets are reused, not newly untouched data.
- Exactly the original shifted next-token, token-weighted NLL calculation:
  523,776 scored tokens per split, bf16 model, FP32 cross entropy, batch1,
  model.eval(), no cache, no gradients. Save per-sequence losses for pairing.
- Train on FineWeb-Edu using the parent recipe: 1280 width, 36 layers,
  16 attention heads, 8 KV heads, FFN5120, context1024, AdamW, global batch32,
  per-GPU batch4 × accumulation8, LR5e-4, minimum5e-5, warmup1000.
  Keep `--steps 20000`; use `--stop_after_step 1600` separately.
- All branches use eager routing, including C, as in Experiment 4; the original
  parent training was fused. Record this implementation difference.
- Freeze the clean execution commit and branch manifest before the first eval.
  Do not update either execution worktree during the run. The earlier immutable
  `/root/mhar-training-81ff305` worktree stays untouched.

## Before any training: reproduce step0

Evaluate A/B/C in parallel from the parent on both fixed splits. Compare every
NLL and every per-sequence loss to archived Experiment 3 results. Fail closed if
absolute aggregate error exceeds 1e-5 or maximum sequence error exceeds 1e-4.
These are numerical-reproduction tolerances, not hypothesis-test thresholds.
Do not relax them to force a pass. Any mismatch needs diagnosis before training.

Expected confirmation values from archived measurements:

| A NLL | B NLL | C NLL | A−B | A−C |
|---:|---:|---:|---:|---:|
| 4.566358858 | 4.577415382 | 4.542211858 | −0.011056524 | +0.024147000 |

The archived paired-sequence A−B mean is −0.011056518; the tiny difference from
subtracting aggregate NLLs is floating-point reduction. "Good" means the best
merge, not a gain over native H16.

## Execution

1. Check parent/artifact/reference hashes, GPU availability, disk capacity and
   clean code. Run the global step0 reproduction gate first.
2. Run A/B/C on separate GPUs. Each trains **one uninterrupted 100-update
   trajectory**, saving protected full-state checkpoints at each requested step.
   Log each training step as secondary context, never as validation NLL.
3. After its training exits successfully, each worker evaluates its own saved
   snapshots in increasing step order on its GPU. This evaluates the exact
   requested weights without disturbing training RNG/data state. C can begin
   its evaluations while A/B are still training.
4. Require all 24 branch/time records (48 split measurements), complete
   step1600/final checkpoint manifests, and matching branch/code/artifact hashes.
   Failed processes are not automatically restarted. No overwrite or pruning
   of any older experiment or requested snapshot.
5. Analyze, upload actual JSON/CSV/figures/manifest to W&B, copy compact results
   locally, test, commit, push and verify GitHub. Only then acknowledge backup
   on the server, wait ten minutes, and shut down if no GPU work remains.

Allow at least **180 GiB free** for 21 full-state snapshots plus three final
copies and operational headroom. Local backup is compact results, not all model
checkpoints. Keep model checkpoints on persistent server storage.

## Analysis and interpretation

Report every sampled time, not only favorable windows:

- A/B/C token-weighted held-out NLL, A−B and A−C.
- Paired bootstrap over the same 512 sequences, 10,000 resamples, seed20260830;
  pointwise 95% intervals. These describe held-out-example uncertainty, not
  training-seed variance. Repeated time points are correlated.
- Signed gap divided by the step0 gap. A sign reversal must remain visible.
- First **sampled** time whose A−B 95% interval is entirely inside
  **[−0.001, +0.001]** and stays there at every later sampled time. This is an
  explicitly chosen practical washout diagnostic (approximately a 90% reduction
  from the initial gap), not an exact crossing time or universal cutoff.
- Report separately whether A−B's upper interval bound is below zero at **both
  +50 and +100**. A tiny persistent negative difference can also be practically
  negligible; do not conflate sign with useful magnitude.

Practical washout by +20 supports stopping this hard-boundary direction in this
setting. A material persistent gap motivates discussion of adaptive grouping,
not automatic follow-up training. Otherwise report late washout, reversal, or
inconclusive evidence. A CI merely crossing zero does **not** prove equivalence.
The design cannot establish a universal mechanism, a population-level null,
cross-seed robustness, or improvement over unchanged C from A−B alone.

## Outputs

- W&B project `MHAR Stuff`, group
  `mhar-exp5-fixed-validation-washout-seed43-step1500`.
- Three training runs and one fixed-validation analysis run.
- `measurements/*.json`: all original per-sequence results and provenance.
- `branch_manifest.json`, `step0_gate.json`, completion and upload markers.
- `analysis/fixed_eval_losses.csv`, `washout_summary.json`, `FINAL_REPORT.md`.
- `fig_washout.png` and vector PDF: A−B/A−C versus updates with pointwise
  intervals. The symlog x-axis is labeled and shows all eight sampled offsets.

Operational instructions: `docs/runbooks/experiment5-three-gpu.md`.
