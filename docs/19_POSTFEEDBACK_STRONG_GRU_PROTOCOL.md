# Strong GRU control protocol

**Status:** frozen post hoc protocol. Outer results were known when this analysis was designed.

## Question

Does distillation improve a gaze-only GRU after the hard-label baseline receives a broader, matched search?

## Selection

The existing outer reader-by-article partitions and three paired inner holdouts are retained. BCE alone selects among four causal GRU capacities and four learning rates. Selection uses mean full-trial balanced accuracy across the three inner holdouts, followed by lower NLL, fewer parameters, and lower learning rate.

KD inherits the BCE-selected architecture in each outer context. Only its learning rate and distillation weight are selected. This ordering favors the hard-label control and prevents KD from receiving a broader architecture search.

All candidates use AdamW, the same fixation features, preprocessing, batch size, weight decay, head dropout, maximum epoch count, patience, and selection folds. No outer label is accessed during fitting or selection.

## Outer comparison

The selected BCE and KD configurations are refit for seeds 13, 29, and 47. The epoch count is the integer median of the three selected inner epochs. A class-conditional-mean KD control replaces each training trial's teacher logit with the mean logit for its training class. This retains class-level softening while removing trial-specific teacher correspondence.

Primary contrasts are tuned KD minus tuned BCE and tuned KD minus class-mean KD. Reader-cluster and crossed reader-article bootstrap intervals are reported over the mean of matched seed-specific balanced-accuracy differences.

## Interpretation

This is a post hoc robustness analysis, not a new confirmatory test. A positive tuned-KD contrast supports an optimization benefit beyond the original fixed GRU. A positive trial-specific contrast supports information in per-trial teacher targets. Neither contrast alone establishes that contextual text is necessary.
