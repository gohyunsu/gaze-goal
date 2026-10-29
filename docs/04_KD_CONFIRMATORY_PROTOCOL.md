# GoalExit KD-only confirmatory addendum

**Status:** pre-outer protocol; no GoalExit outer outcome has been evaluated

**Date fixed:** 2026-10-27 KST

**Target venue:** IEEE ICCE 2027

**Scope:** one final unseen-reader × unseen-article evaluation of the
development-selected privileged-distillation claim

## 1. Why this addendum exists

The original Phase 0–2 protocol tested a longer chain: multimodal teacher,
privileged distillation, anytime training, and adaptive stopping. The frozen
development gates supported the teacher and distillation stages but rejected the
anytime stage:

- Gate T passed: mean `T - B1 = +0.033459` balanced accuracy;
- Gate D passed: mean `D - B1 = +0.033128` balanced accuracy;
- Gate A0 failed: mean `A_FULL - D = -0.007676`, below the allowed `-0.005`;
- adaptive stopping was therefore not run.

This addendum does not reinterpret Gate A0 or rescue adaptive stopping. It closes
that direction and defines a narrower, development-selected confirmatory question:

> Does a text-and-gaze teacher transfer useful information to an otherwise
> identical compact gaze-only model on readers and articles excluded from model
> fitting and development selection?

The change of emphasis was made after development results and before any outer
outcome. The eventual paper must say so plainly. The contribution is privileged
training for compact consumer inference, not a preregistered end-to-end adaptive
system and not the first reading-goal classifier.

## 2. Frozen models and contrasts

Only five existing model families are authorized:

1. `B0`: 22-feature global gaze-statistics MLP;
2. `B1`: causal gaze-only GRU trained with hard labels;
3. `T`: frozen-RoBERTa text embeddings plus word-aligned gaze teacher;
4. `T-text`: identical teacher with gaze values zeroed;
5. `D`: the same architecture, inputs, and parameter count as B1, trained with
   the already selected teacher-logit distillation loss.

`A`, adaptive traces, alternative students, new features, new pretrained text
models, and new hyperparameter searches are prohibited.

### 2.1 Primary contrast

The sole primary contrast is `D - B1` in pooled outer balanced accuracy. This
isolates the effect of privileged teacher supervision because B1 and D have the
same gaze inputs, architecture, and deployment capacity.

Uncertainty is a paired 5,000-draw reader-cluster percentile bootstrap,
stratified by the reader-constant goal label. The same sampled readers are used
for both models in every draw.

### 2.2 Required practical and mechanistic checks

The following are secondary but required for a viable paper:

- `D - B0`: whether distillation also beats the strongest compact development
  baseline rather than only the matched GRU control;
- `T - D`: whether the compact student remains within 1.0 balanced-accuracy
  point of the privileged teacher;
- `T-text - B1`: whether text alone reproduces the teacher gain;
- parameter count, serialized state size, and single-thread CPU latency for D;
- trial-level AUROC, macro-F1, Brier, NLL, ECE, macro-by-article balanced
  accuracy, equal-context mean balanced accuracy, and goal-specific recall.

A 2,000-draw paired reader-and-article bootstrap is a sensitivity analysis for
the crossed generalization claim. It does not replace the primary reader-cluster
interval.

## 3. Outer refit and prediction rule

The existing split manifest defines 25 contexts `(reader_fold, article_fold)`.
For each context:

- fit population: rows with neither the held reader fold nor held article fold;
- outer target: the intersection of the held reader fold and held article fold;
- development-only hyperparameters: the context's already selected learning rate
  and, for D, distillation alpha;
- fixed refit epoch count: the integer median of the three already saved
  selected inner-model epochs for that context and family;
- random seed: 13 only.

No outer label is used for preprocessing, checkpoint selection, calibration,
thresholding, epoch choice, or model selection. Refit uses all population-fit
rows for exactly the frozen epoch count and has no outer-informed early stopping.

For D, the context-specific refitted teacher generates in-sample logits only for
the same population-fit rows. These are training-only privileged targets. The
teacher never sees an outer target row during fitting or cache generation.

## 4. Staged one-shot execution

Execution is deliberately separated:

1. validate the signed/frozen lock, clean source commit, development hashes, and
   exact selected settings;
2. train all 25 refits and write target-free outer logits for B0/B1/T/T-text;
3. validate 25 teacher caches and train/write target-free logits for D;
4. require the exact complete set of 125 prediction/checkpoint records;
5. only then materialize outer labels, audit membership and provenance, compute
   all metrics once, and write the final report artifacts.

Partial scientific results must not be inspected. Infrastructure progress may
report only completed artifact counts and process health.

## 5. Predeclared paper viability decision

The KD-only paper direction may proceed only if all are true on the one-shot
outer evaluation:

1. primary `D - B1` point estimate is positive;
2. the paired reader-cluster 95% interval for `D - B1` has a lower bound above
   zero;
3. `D - B0` pooled balanced-accuracy difference is nonnegative;
4. `D` is no more than 0.010 balanced accuracy below `T`;
5. `T-text - B1` is no larger than half of `T - B1` when the latter is positive;
6. no leakage, provenance, completeness, or non-finite-output audit fails.

If any condition fails, stop the positive KD paper. Do not change the endpoint,
margin, bootstrap, baseline, model, seed, or title to rescue the result. A
negative technical report may be retained, but it does not satisfy the present
ICCE paper objective.

## 6. Claim boundary if the direction passes

Allowed claims are limited to the observed OneStop task and hardware evidence:

- privileged text-and-gaze supervision improves a matched compact gaze-only GRU;
- the compact student preserves the teacher result within the predeclared margin;
- deployment uses gaze only and omits RoBERTa, tokenization, and text encoding;
- measured latency is a CPU deployment proxy unless an actual consumer device is
  separately measured.

Do not claim general reading-intent inference, causal cognitive transfer,
embedded-device latency without measurement, or adaptive gaze acquisition.
