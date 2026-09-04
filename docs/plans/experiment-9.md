# Experiment 9 — HQ8 Local-vs-Global Head Contribution

## Question

Inside the accepted Experiment 8 HQ8 model, do the eight local-query heads
perform useful computation, or is performance carried mainly by the eight
global-query heads?

This is a frozen causal-ablation experiment. It performs no training and never
changes the accepted checkpoint or evaluation artifact.

## Starting point

- model: Experiment 8 HQ8, seed 42, atomic step 2,000
- checkpoint content SHA-256:
  `74cff0ab19409dac9f6104e8986e4890c0837bc730d8ac233ae18011dbc58333`
- fixed document-disjoint artifact SHA-256:
  `29e545dd9399e9eaea6f5abf38f20ef76ba232466a01b6ddf13c3a6a287a3691`
- reference reproduction: the unchanged condition must reproduce the accepted
  Experiment 8 per-sequence and aggregate NLLs before ablations are accepted

## Experiment 9A — contribution

At every layer, zero selected attention-head outputs immediately before the
dense output projection. Query, key, value, MHAR, FFN, normalization, and all
weights remain untouched.

1. `hq8-unchanged`: keep all 16 heads.
2. `zero-local`: zero even heads 0,2,...,14 (all eight local-Q heads).
3. `zero-global`: zero odd heads 1,3,...,15 (all eight global-Q heads).
4. `balanced-XXXX`: exhaust all 70 ways to remove one head per GQA group while
   removing four local and four global heads.

The exhaustive balanced family is the matched half-head-removal reference. It
avoids sampling discretion and holds constant the number of removed heads, the
local/global count, and the one-removal-per-GQA-group structure.

Primary quantities are `zero-local − unchanged` and
`zero-global − unchanged`, with paired per-sequence 10,000-sample bootstrap
intervals on both fixed splits. Structured ablations are also located within
the empirical distribution of the 70 balanced masks. The direct paired
contrast `(zero-local − unchanged) − (zero-global − unchanged)` tests which
trained head population is more important; positive values favor greater local
head importance.

## Frozen 9B gate

Experiment 9B runs only if all of the following hold:

- discovery `zero-local − unchanged > 0`;
- confirmation `zero-local − unchanged >= 0.001` NLL;
- confirmation paired-bootstrap 95% CI lower bound is above zero.

Otherwise the workflow stops after 9A and reports that local-head contribution
was not established under the preregistered rule.

## Experiment 9B — alignment

If authorized, evaluate 32 unique frozen random derangements of the eight local
chunk assignments. The same derangement is applied in every layer for one
condition. Global Q, K, V, dense W_O, learned weights, and MHAR routing remain
unchanged. A local query group keeps its learned projection but reads the
permuted source chunk.

Report every derangement, its paired sequence interval, the distribution of
effects, fraction positive, and a two-stage bootstrap over the frozen
derangement sample and held-out sequences. Alignment evidence requires:

- positive mean effect on discovery;
- confirmation mean effect at least 0.001 NLL;
- confirmation two-stage 95% CI lower bound above zero;
- at least 75% of derangements have positive confirmation delta NLL.

This gate is diagnostic only and does not authorize additional training.

## Interpretation limits

The result is specific to one seed at step 2,000. Head zeroing is an
off-distribution causal intervention, not a retraining comparison. Sequence
bootstrap intervals do not quantify training-seed uncertainty. Passing 9B
would show dependence on learned chunk alignment inside this trained HQ8; it
would not by itself establish generality or final-training superiority.
