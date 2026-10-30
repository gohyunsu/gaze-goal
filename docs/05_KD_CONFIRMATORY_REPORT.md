# GoalExit KD Confirmatory Report

**Verdict: PASS**

The positive KD manuscript is authorized.

## Pooled outer performance

| Model | Balanced accuracy | AUROC | Macro F1 | NLL |
|---|---:|---:|---:|---:|
| b0 | 0.5942 | 0.6359 | 0.5936 | 0.6632 |
| b1 | 0.5866 | 0.6178 | 0.5862 | 0.6710 |
| teacher | 0.6126 | 0.6613 | 0.6120 | 0.6561 |
| t_text | 0.4986 | 0.5017 | 0.4986 | 0.6952 |
| d | 0.6205 | 0.6708 | 0.6204 | 0.6427 |

## Frozen contrasts

- D − B1: +0.0339
- D − B0: +0.0263
- Teacher − D: -0.0079
- T-text − B1: -0.0880

## Primary uncertainty

Reader-cluster 95% CI for D − B1: [+0.0175, +0.0504].
Crossed reader/article sensitivity CI: [+0.0130, +0.0553].

## Viability checks

- PASS: d_minus_b1_point_positive
- PASS: d_minus_b1_reader_ci_lower_positive
- PASS: d_minus_b0_nonnegative
- PASS: d_within_0p010_of_teacher
- PASS: text_gain_at_most_half_teacher_gain
- PASS: complete_set_audit_pass
