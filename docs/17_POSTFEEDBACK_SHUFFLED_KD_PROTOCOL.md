# Post hoc class-conditional target-shuffling protocol

## Purpose and status

This post hoc control was specified after all reported results were known. It
tests whether the benefit of distillation depends on matching each training
trial to its own teacher confidence.

## Controlled comparison

For each of the three existing seeds, the control retains the text--gaze
teacher logits, student architecture, training trials, optimizer settings,
distillation weight, temperature, and fixed epoch count. Within each outer
training set, teacher logits are reassigned among trials with the same binary
label. A deterministic cyclic permutation gives every trial another trial's
logit while preserving the exact class-conditional logit distributions.

The control is compared with the matched undistilled GRU and the original
text--gaze KD student. No hyperparameter is selected from outer results.

## Interpretation

- Lower performance than original KD supports the relevance of trial-specific
  teacher confidence.
- Similar performance indicates that class-conditional confidence
  distributions or other optimization effects can explain the gain.
- The control does not identify word alignment, teacher capacity, or
  contextual text as the source of any trial-specific signal.
