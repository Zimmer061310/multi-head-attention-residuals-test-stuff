# Experiment 6 — Coupled MHAR Chunks to Attention Groups

Completed 2026-09-02 at the step-2,000 screening milestone. All five new
seed-42 runs and both fixed evaluation splits completed. M4 and M8 reuse the
accepted Experiment 2 Stage B results on the identical artifact. This is a
single-seed early screen, not a multi-seed architectural claim.

## Main finding

Hard chunk-to-attention coupling does not improve MHAR. Making Q, K, and V
block-restricted raises confirmation NLL by **+0.181916** for four chunks and
**+0.286143** for eight chunks relative to dense-QKV MHAR. Both paired 95%
confidence intervals lie wholly above zero.

Restricted QKV also harms the ordinary-residual controls: G4−B is +0.220316
and G8−B is +0.302994. The negative interaction terms show that MHAR partially
reduces this generic restriction penalty, but they do not make coupling
beneficial. C4 and C8 remain substantially worse than M4 and M8.

The practical conclusion is to stop this exact hard Q/K/V coupling design.
The result motivated Experiment 7's weaker Local-Q / Global-KV test.

## Question and architectures

Experiment 6 tests whether each depth-routed MHAR chunk should remain isolated
through Q/K/V projection instead of being concatenated and immediately remixed
by dense projections. Attention, Q/K normalization, RoPE, GQA repetition, and
the dense output projection remain unchanged.

| ID | Residual routing | Q/K/V input structure | Role |
|---|---|---|---|
| B | ordinary residual | dense/global | dense baseline |
| M4 | MHAR, 4 × 320 | dense/global | accepted MHAR-4 control |
| C4 | MHAR, 4 × 320 | 4 restricted groups | proposed coupled model |
| G4 | ordinary residual | 4 restricted groups | restriction control |
| M8 | MHAR, 8 × 160 | dense/global | accepted MHAR-8 control |
| C8 | MHAR, 8 × 160 | 8 restricted groups | proposed coupled model |
| G8 | ordinary residual | 8 restricted groups | restriction control |

For every contrast below, ΔNLL is `first model − second model`; lower NLL is
better, so a negative value favors the first model. For an interaction,

`(C−M)−(G−B)`,

a negative value means MHAR bears less of the restriction penalty than the
ordinary residual control.

## Frozen protocol

All new models were trained from scratch with the same 1B-class setup:

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
per-sequence bootstrap resamples with seed 20260902. Confirmation is the
primary split; discovery is reported as a consistency check.

## Held-out results

### Absolute NLL and perplexity

| Rank | Model | Confirmation NLL ↓ | Confirmation PPL ↓ | Discovery NLL ↓ | Discovery PPL ↓ |
|---:|---|---:|---:|---:|---:|
| 1 | M8 | **4.137027** | **62.616** | **4.247944** | **69.961** |
| 2 | M4 | 4.137516 | 62.647 | 4.253231 | 70.332 |
| 3 | C4 | 4.319431 | 75.146 | 4.450584 | 85.677 |
| 4 | B | 4.325586 | 75.610 | 4.445220 | 85.219 |
| 5 | C8 | 4.423170 | 83.360 | 4.549776 | 94.611 |
| 6 | G4 | 4.545902 | 94.245 | 4.675566 | 107.293 |
| 7 | G8 | 4.628580 | 102.369 | 4.759393 | 116.675 |

The rank column follows primary confirmation NLL. B is slightly better than C4
on discovery, so the discovery-only ordering swaps those two models.

### Frozen paired contrasts

| Contrast | Confirmation ΔNLL | Paired 95% CI | Discovery ΔNLL | Paired 95% CI |
|---|---:|:---|---:|:---|
| C4−M4 | +0.181916 | [+0.174898, +0.189248] | +0.197352 | [+0.189415, +0.205760] |
| C8−M8 | +0.286143 | [+0.275798, +0.296909] | +0.301832 | [+0.290909, +0.313219] |
| G4−B | +0.220316 | [+0.210061, +0.231241] | +0.230346 | [+0.220369, +0.240781] |
| G8−B | +0.302994 | [+0.291127, +0.315902] | +0.314173 | [+0.302398, +0.326110] |
| H4 interaction | −0.038401 | [−0.044389, −0.032657] | −0.032994 | [−0.038534, −0.027616] |
| H8 interaction | −0.016851 | [−0.022325, −0.011461] | −0.012341 | [−0.017484, −0.007289] |

The result is consistent across both fixed splits. The eight-group restriction
is more damaging than the four-group restriction. The negative interactions
mean that MHAR partially mitigates the generic damage from restricted QKV; they
do not mean C4 or C8 beats its dense MHAR control.

## Parameters, projection work, and measured evaluation speed

The restricted models deliberately remove disallowed QKV blocks and are not
parameter matched to the dense models. The MAC counts below cover only the QKV
linear projections per token, not attention score computation or whole-model
FLOPs. W_O remains dense at 58,982,400 parameters/MACs per token for every row.

| Model | Trainable parameters | QKV parameters / MACs per token | QKV reduction vs B | Confirmation eval tok/s |
|---|---:|---:|---:|---:|
| B | 1,079,313,280 | 117,964,800 | — | 30,479 |
| C4 | 991,026,560 | 29,491,200 | 75.0% | 7,748 |
| G4 | 990,839,680 | 29,491,200 | 75.0% | 31,354 |
| C8 | 976,280,960 | 14,745,600 | 87.5% | 7,703 |
| G8 | 976,094,080 | 14,745,600 | 87.5% | 31,668 |

C4 and C8 contain 8.18% and 9.55% fewer total trainable parameters than B.
Measured evaluation throughput is reported descriptively, not as a clean
connectivity speedup: MHAR and ordinary-residual rows execute different routing
paths, and the eager MHAR path dominates the observed difference.

## Decision and limits

The frozen controller recorded `eligible_for_review` because no primary
confirmation penalty exceeded the preregistered catastrophic threshold of
+0.5 NLL and every checkpoint/evaluation was finite and complete. That label
means only “not catastrophic by the frozen cutoff.” It is not evidence that
the architecture is competitive or eligible for automatic continuation.

The scientific screen is nevertheless clearly negative for the tested design:

- C4 and C8 lose to their dense MHAR controls by large, precisely estimated
  margins on both splits;
- G4 and G8 show that hard QKV restriction is intrinsically costly;
- MHAR recovers only a small part of that cost;
- no continuation or multi-seed claim is warranted for hard local Q/K/V.

This is one seed at an early checkpoint. The intervals quantify variation
across paired validation sequences, not uncertainty across training seeds.
The models are not parameter or compute matched, and step 2,000 is not a
convergence result. These limitations prevent a general claim that every form
of chunk-attention coupling is harmful; they do reject the exact hard
block-diagonal Q/K/V design tested here.

## Verification and provenance

- Training manifests report source commit
  `6d769a17342f6f29d011e22a93e9afd0f2011d34`.
- Evaluation manifests report source commit
  `cc6332d2843529a32609e550b98aa451d143caec`.
- Compact results were recorded in commit
  `81dbd76e8a0bb7f207557a95a53331a93c37918d`.
- New-run checkpoints have complete training/checkpoint manifests and were
  evaluated on both immutable fixed splits.
- All 19 focused Experiment 6 architecture, analysis, and controller tests
  passed when this report was prepared.
- Machine-readable full-precision results are in
  `step-2000/analysis/summary.json` and `step-2000/analysis/contrasts.csv`;
  per-sequence values and resource measurements remain in each model's
  `evaluation/result.json`.

## W&B

Analysis:

- [Experiment 6 paired analysis](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/m5i39szz)

Training runs:

- [B](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/eve6q9lj)
- [C4](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/tsici7lo)
- [C8](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/n6vw65hi)
- [G4](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/t1nq2m4y)
- [G8](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/otir20sx)

Fixed-set evaluation runs:

- [B](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/p1os0g15)
- [C4](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/v8pw946q)
- [C8](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/qbq97duu)
- [G4](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/y3c5zpac)
- [G8](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/gf4spq1m)
- [M4 accepted control](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/bp4b3c13)
- [M8 accepted control](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/rkvwhctm)
