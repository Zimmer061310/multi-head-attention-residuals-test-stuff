# Experiment 7 — Local-Q / Global-KV Coupling

Completed 2026-09-03 at the step-2,000 screening milestone. All four new
seed-42 runs and both fixed evaluation splits completed. B, M4, and M8 reuse
accepted results on the identical artifact; C4 and C8 are reused only as
diagnostic Experiment 6 comparisons. This is a single-seed early screen, not a
multi-seed architectural claim.

## Main finding

Restoring global K and V recovers most of the loss from Experiment 6, but hard
local Q still does not improve MHAR. Relative to dense MHAR, confirmation NLL
rises by **+0.048601** for LQ4 and **+0.054053** for LQ8. Both paired 95%
confidence intervals lie wholly above zero, and discovery agrees.

The ordinary-residual controls show a generic local-query penalty: BLQ4−B is
+0.032501 and BLQ8−B is +0.059149. MHAR makes the four-group local-Q penalty
larger, not smaller. At eight groups, MHAR reduces the generic penalty by about
0.0051 NLL on confirmation, but this interaction is small, discovery's
interaction interval crosses zero, and LQ8 still clearly loses to M8.

The result rejects this exact hard Local-Q / Global-KV design as an improvement.
It also isolates why Experiment 6 was much worse: forcing K and V to remain
local caused most of the earlier damage.

## Question and architectures

Experiment 7 tests whether an MHAR chunk can specialize only the query heads
assigned to it while keys and values retain access to the full routed residual:

`Q = local MHAR chunk; K,V = full routed residual; W_O = dense`.

| ID | Residual routing | Q input | K/V input | Role |
|---|---|---|---|---|
| M4 | MHAR, 4 × 320 | dense/global | dense/global | accepted MHAR-4 control |
| LQ4 | MHAR, 4 × 320 | 4 local groups | dense/global | proposed four-group model |
| M8 | MHAR, 8 × 160 | dense/global | dense/global | accepted MHAR-8 control |
| LQ8 | MHAR, 8 × 160 | 8 local groups | dense/global | proposed eight-group model |
| B | ordinary residual | dense/global | dense/global | accepted baseline |
| BLQ4 | ordinary residual | 4 local groups | dense/global | generic restriction control |
| BLQ8 | ordinary residual | 8 local groups | dense/global | generic restriction control |

For every contrast below, ΔNLL is `first model − second model`; lower NLL is
better, so a negative value favors the first model. The interaction is

`(LQ−M)−(BLQ−B)`.

A negative interaction means MHAR bears less of the local-Q penalty than an
ordinary residual model. LQ4−C4 and LQ8−C8 are diagnostic only because they
also restore dense K/V parameters and are not isolated causal contrasts.

## Frozen protocol

All four new models were trained from scratch under the same setup as
Experiments 2 and 6:

- hidden width 1,280, 36 layers, and FFN width 5,120;
- 16 query heads, 8 KV heads, and head dimension 80;
- sequence length 1,024 and global batch size 32;
- AdamW in bf16, peak/minimum LR 5e-4/5e-5, and 1,000-step warmup;
- the original 20,000-step schedule, stopped atomically at step 2,000;
- seed 42, matched initialization where tensor shapes permit, and the same
  packed FineWeb-Edu `sample-10BT` data order;
- pinned dataset revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`
  and tokenizer revision `c1899de289a04d12100db370d81485cdf75e47ca`;
- the document-disjoint fixed artifact with SHA-256
  `29e545dd9399e9eaea6f5abf38f20ef76ba232466a01b6ddf13c3a6a287a3691`.

Each artifact split contains 512 sequences and 523,776 scored next-token
positions. NLL is token weighted. Confidence intervals use 10,000 paired
per-sequence bootstrap resamples with seed 20260903. Confirmation is the
primary split; discovery is reported as a consistency check.

## Held-out results

### Absolute NLL and perplexity

| Rank | Model | Confirmation NLL ↓ | Confirmation PPL ↓ | Discovery NLL ↓ | Discovery PPL ↓ |
|---:|---|---:|---:|---:|---:|
| 1 | M8 | **4.137027** | **62.616** | **4.247944** | **69.961** |
| 2 | M4 | 4.137516 | 62.647 | 4.253231 | 70.332 |
| 3 | LQ4 | 4.186116 | 65.767 | 4.305539 | 74.109 |
| 4 | LQ8 | 4.191079 | 66.094 | 4.306005 | 74.144 |
| 5 | C4 | 4.319431 | 75.146 | 4.450584 | 85.677 |
| 6 | B | 4.325586 | 75.610 | 4.445220 | 85.219 |
| 7 | BLQ4 | 4.358087 | 78.108 | 4.482452 | 88.451 |
| 8 | BLQ8 | 4.384735 | 80.217 | 4.504991 | 90.468 |
| 9 | C8 | 4.423170 | 83.360 | 4.549776 | 94.611 |

The rank column follows primary confirmation NLL. B is slightly better than C4
on discovery, so the discovery-only ordering swaps those two models.

### Primary and control contrasts

| Contrast | Confirmation ΔNLL | Paired 95% CI | Discovery ΔNLL | Paired 95% CI |
|---|---:|:---|---:|:---|
| LQ4−M4 | +0.048601 | [+0.046087, +0.051125] | +0.052308 | [+0.049356, +0.055201] |
| LQ8−M8 | +0.054053 | [+0.051504, +0.056699] | +0.058061 | [+0.054673, +0.061572] |
| BLQ4−B | +0.032501 | [+0.029952, +0.035108] | +0.037233 | [+0.034403, +0.040128] |
| BLQ8−B | +0.059149 | [+0.055859, +0.062616] | +0.059771 | [+0.056771, +0.062617] |
| H4 interaction | +0.016099 | [+0.012839, +0.019317] | +0.015075 | [+0.011459, +0.018662] |
| H8 interaction | −0.005097 | [−0.008955, −0.001362] | −0.001710 | [−0.005435, +0.002111] |

The four-group interaction is positive on both splits: MHAR does not protect
against local-Q restriction at H4. The H8 interaction is mildly favorable on
confirmation but is not consistent enough to overturn the absolute LQ8−M8
penalty or justify a success claim.

### Diagnostic recovery from Experiment 6

| Contrast | Confirmation ΔNLL | Paired 95% CI | Discovery ΔNLL | Paired 95% CI |
|---|---:|:---|---:|:---|
| LQ4−C4 | −0.133315 | [−0.139886, −0.127105] | −0.145045 | [−0.151776, −0.138558] |
| LQ8−C8 | −0.232091 | [−0.242224, −0.222343] | −0.243771 | [−0.253328, −0.234520] |

Global K/V recovers 73.3% of the Experiment 6 C4−M4 confirmation penalty and
81.1% of the C8−M8 penalty. The remaining local-Q penalty is still clearly
positive, so “better than C4/C8” is not “better than dense MHAR.”

## Parameters, projection work, and measured evaluation speed

The Local-Q models remove disallowed Q blocks but keep K, V, and W_O dense;
they are not parameter matched to their dense controls. The MAC counts below
cover only QKV linear projections per token, not attention score computation or
whole-model FLOPs. W_O remains dense at 58,982,400 parameters/MACs per token.

| Model | Trainable parameters | QKV parameters / MACs per token | QKV reduction vs dense | Confirmation eval tok/s |
|---|---:|---:|---:|---:|
| LQ4 | 1,035,263,360 | 73,728,000 | 37.5% | 7,722 |
| BLQ4 | 1,035,076,480 | 73,728,000 | 37.5% | 31,146 |
| LQ8 | 1,027,890,560 | 66,355,200 | 43.75% | 7,635 |
| BLQ8 | 1,027,703,680 | 66,355,200 | 43.75% | 31,284 |

Relative to dense B, LQ4 and LQ8 contain 4.08% and 4.76% fewer total trainable
parameters. Measured evaluation throughput is descriptive, not a clean
connectivity speedup: MHAR and baseline controls execute different routing
paths, and the eager MHAR path dominates the observed difference.

## Decision and limits

The frozen controller recorded `eligible_for_review` because no primary
confirmation penalty exceeded the preregistered catastrophic threshold of
+0.5 NLL and every checkpoint/evaluation was finite and complete. That status
does not mean the models matched or beat their controls, and it does not
authorize automatic continuation.

The scientific screen supports a narrower negative conclusion:

- keeping K/V global fixes most of the damage caused by full Q/K/V isolation;
- hard local Q still loses clearly to dense-QKV MHAR at both granularities;
- the control and interaction results provide no robust evidence that MHAR
  makes hard local queries beneficial;
- this exact design should not advance to longer or multi-seed training without
  a new rationale.

This is one seed at step 2,000. Paired sequence-bootstrap intervals do not
represent training-seed uncertainty. The models are not parameter or compute
matched, and no result establishes convergence or end-to-end hardware speedup.
The experiment does not rule out softer coupling, shared/low-rank query access,
or other mechanisms that preserve some cross-chunk information.

## Verification, provenance, and operational deviations

- Training and evaluation manifests report source commit
  `6ab9254cb14c72cee57b255d7f71e81e00c938bc`.
- The server recorded compact results in commit
  `82e8c4d7acb2eb9504b55c7d63662e8f64719bfa`; the verified local integration is
  commit `4fa808f4896759420f19b8e0a16633a42cc49da6`.
- Every new checkpoint has complete training/checkpoint manifests and was
  evaluated on both immutable fixed splits.
- All 15 focused Experiment 7 architecture, analysis, and controller tests
  passed when this report was prepared.
- Machine-readable full-precision results are in
  `step-2000/analysis/summary.json` and `step-2000/analysis/contrasts.csv`;
  per-sequence values and resource measurements remain in each model's
  `evaluation/result.json`.

Two operational-only deviations did not change scientific inputs or results:

1. The first delayed LQ8 launch failed in its shell wrapper before creating a
   checkpoint manifest, W&B run, or optimizer update. The identical frozen run
   was then launched successfully.
2. The first analysis upload (`fbrob5nj`) computed the accepted analysis but
   failed while writing W&B summary metadata because the installed client did
   not accept keyword arguments for that update. The corrected uploader reused
   the same saved results without rerunning evaluation. The successful analysis
   run is `vqqcsr6m`.

The remote HTTPS push timed out after the result commit was created. The exact
commit bundle was transferred, pushed, and verified locally; no scientific
file changed. The server shut down only after backup verification, GPU-idleness
checks, and the success-only grace period.

## W&B

Analysis:

- [Experiment 7 paired analysis](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/vqqcsr6m)
- [Archived failed metadata-upload attempt](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/fbrob5nj)

Training runs:

- [LQ4](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/8sjrg1ya)
- [LQ8](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/t5b6sc03)
- [BLQ4](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/1ck42204)
- [BLQ8](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/05z2myqo)

Fixed-set evaluation runs:

- [LQ4](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/14pswxfp)
- [LQ8](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/y1n7s52r)
- [BLQ4](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/f9b0v1q7)
- [BLQ8](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/hxo0izpi)
- [B accepted control](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/p1os0g15)
- [M4 accepted control](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/bp4b3c13)
- [M8 accepted control](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/rkvwhctm)
- [C4 diagnostic](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/v8pw946q)
- [C8 diagnostic](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/qbq97duu)
