# Post-confirmatory ablation report

**Status: PASS**

These analyses were designed after the original outer results were known.
They are mechanism and robustness evidence, not a second confirmatory test.

## Pooled performance

| Variant | BA | AUROC | Macro F1 | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| stats | 0.5950 | 0.6366 | 0.5945 | 0.6627 | 0.2358 | 0.0398 |
| gru_s13 | 0.5865 | 0.6186 | 0.5861 | 0.6717 | 0.2393 | 0.0209 |
| label_smoothing | 0.5853 | 0.6185 | 0.5851 | 0.6692 | 0.2385 | 0.0112 |
| gaze_teacher | 0.6117 | 0.6637 | 0.6097 | 0.6483 | 0.2284 | 0.0304 |
| gaze_kd | 0.6236 | 0.6732 | 0.6235 | 0.6409 | 0.2257 | 0.0099 |
| text_gaze_teacher | 0.6106 | 0.6599 | 0.6099 | 0.6579 | 0.2315 | 0.0505 |
| text_only | 0.4984 | 0.5022 | 0.4984 | 0.6951 | 0.2509 | 0.0298 |
| text_gaze_kd_s13 | 0.6204 | 0.6691 | 0.6203 | 0.6447 | 0.2270 | 0.0135 |
| gru_s29 | 0.5814 | 0.6142 | 0.5814 | 0.6755 | 0.2409 | 0.0287 |
| text_gaze_kd_s29 | 0.6303 | 0.6821 | 0.6301 | 0.6361 | 0.2238 | 0.0173 |
| gru_s47 | 0.5892 | 0.6207 | 0.5891 | 0.6708 | 0.2388 | 0.0237 |
| text_gaze_kd_s47 | 0.6190 | 0.6720 | 0.6190 | 0.6419 | 0.2262 | 0.0229 |

## Ablation contrasts

| Contrast | Pooled BA delta | Reader 95% CI | Crossed 95% CI | Positive contexts |
|---|---:|---:|---:|---:|
| text_gaze_kd_minus_gru | +0.0339 | [0.0176, 0.0505] | [0.0146, 0.0544] | 18/25 |
| label_smoothing_minus_gru | -0.0012 | [-0.0053, 0.0028] | [-0.0070, 0.0046] | 10/25 |
| gaze_kd_minus_gru | +0.0371 | [0.0187, 0.0552] | [0.0155, 0.0589] | 20/25 |
| text_gaze_kd_minus_gaze_kd | -0.0032 | [-0.0108, 0.0044] | [-0.0139, 0.0074] | 10/25 |
| text_gaze_teacher_minus_gaze_teacher | -0.0011 | [-0.0098, 0.0073] | [-0.0136, 0.0114] | 12/25 |

## Seed sensitivity

| Seed | Teacher BA | GRU BA | Text--gaze KD BA | Delta | Reader 95% CI |
|---:|---:|---:|---:|---:|---:|
| 13 | 0.6106 | 0.5865 | 0.6204 | +0.0339 | [0.0176, 0.0502] |
| 29 | 0.6154 | 0.5814 | 0.6303 | +0.0489 | [0.0298, 0.0681] |
| 47 | 0.6118 | 0.5892 | 0.6190 | +0.0299 | [0.0135, 0.0466] |

The KD difference is positive for 3/3 seeds; mean delta +0.0376, range [+0.0299, +0.0489].

## Original CPU to uniform GPU replication

| Original family | Original CPU BA | Uniform GPU rerun BA | Difference |
|---|---:|---:|---:|
| b0 | 0.5942 | 0.5950 | +0.0008 |
| b1 | 0.5866 | 0.5865 | -0.0001 |
| teacher | 0.6126 | 0.6106 | -0.0020 |
| t_text | 0.4986 | 0.4984 | -0.0003 |
| d | 0.6205 | 0.6204 | -0.0001 |
