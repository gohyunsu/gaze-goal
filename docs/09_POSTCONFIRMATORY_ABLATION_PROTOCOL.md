# Post-confirmatory ablation protocol

## Status and purpose

This analysis was designed after the original outer results were known. It is
therefore a post-confirmatory robustness and mechanism analysis, not a second
confirmatory test. The original results and artifacts remain unchanged.

All models entering the controlled comparisons are refitted in one software
environment on the same single NVIDIA L40S GPU allocation. Hardware is
therefore not changed
with model condition. The original CPU fits are retained only as an external
replication reference and are not combined with the controlled ablation
contrasts. CPU remains the target device for the separately reported
deployment-latency benchmark because that benchmark asks about CPU deployment.

The analysis addresses three questions:

1. Does ordinary label smoothing explain the gain over the undistilled GRU?
2. Does distillation from a word-aligned gaze Transformer produce the same
   gain as distillation from the text--gaze Transformer?
3. Does the text--gaze distillation gain persist under additional optimization
   seeds?

## Fixed variants

All variants use the original 25 outer reader-fold by article-fold partitions.
For each context, hyperparameters and epoch counts are inherited from the
already completed inner-development selections. Outer results are not used to
choose learning rates, epoch counts, distillation weights, or smoothing.

The uniform seed-13 rerun contains `stats`, `gru_s13`,
`text_gaze_teacher`, `text_only`, `text_gaze_kd_s13`, `gaze_teacher`,
`gaze_kd`, and `label_smoothing`. Thus every seed-13 number used in an
ablation contrast comes from the same GPU/software environment. The original
CPU confirmatory results are summarized separately as a replication check.

- `label_smoothing`: the GRU uses the original GRU learning rate and median
  selected epoch count. Binary targets are changed from 0/1 to 0.05/0.95 using
  epsilon 0.10.
- `gaze_teacher`: the original Teacher architecture and its selected learning
  rate and epoch count are retained, but every RoBERTa representation is
  replaced by zero. Word-aligned gaze and word positions remain available.
- `gaze_kd`: the original Distilled architecture, temperature, alpha, learning
  rate, and epoch count are retained. Soft targets come from `gaze_teacher`.
- `gru_s13`, `gru_s29`, `gru_s47`: the original GRU configuration is refitted
  with seeds 13, 29, and 47.
- `text_gaze_teacher_s29`, `text_gaze_teacher_s47` and the corresponding
  `text_gaze_kd_s29`, `text_gaze_kd_s47`: the complete text--gaze
  teacher-to-student pipeline is refitted with seeds 29 and 47. Each student
  uses logits from the teacher with the same seed. Together with the seed-13
  pair, this varies teacher initialization, student initialization, and both
  optimization paths. No model uses logits from the original CPU fit.

The word-aligned gaze teacher retains the same trainable parameter count as the
text--gaze teacher. This deliberately holds architecture and optimization
settings fixed so that the ablation changes only access to contextual text.

## Outcomes

The primary descriptive ablation contrasts are:

1. `label_smoothing - gru_s13`;
2. `gaze_kd - gru_s13`;
3. `text_gaze_kd_s13 - gaze_kd`;
4. `text_gaze_kd_s13 - gru_s13` for reference.

For each contrast, the analysis reports pooled balanced-accuracy difference,
the number of positive outer contexts, a reader-cluster percentile interval,
and a crossed reader/article sensitivity interval. It also reports BA, AUROC,
macro F1, NLL, Brier score, ECE, and article-macro BA for every variant.

Seed sensitivity reports the matched text--gaze KD minus GRU difference for
seeds 13, 29, and 47. No seed is excluded based on its result.

## Interpretation rules

- If label smoothing approaches the text--gaze KD gain, the manuscript must not
  attribute the improvement specifically to teacher information.
- If word-aligned gaze KD approaches text--gaze KD, the manuscript must frame the result
  as teacher-driven regularization or transfer from a larger gaze model.
- If text--gaze KD consistently exceeds word-aligned gaze KD, the result supports an
  incremental benefit from contextual text under this controlled ablation.
- Seed results quantify optimization sensitivity and do not replace the
  reader/article uncertainty intervals.
