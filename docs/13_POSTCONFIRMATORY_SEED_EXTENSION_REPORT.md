# Post-confirmatory seed-balanced extension report

**Status: PASS**

All four student conditions and both teacher modalities use the same three seeds.

## Seed-specific balanced accuracy and matched differences

| Seed | GRU | Label smoothing | Gaze-only KD | Text--gaze KD | LS $\Delta$ | Gaze KD $\Delta$ | Text--gaze KD $\Delta$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | 0.5865 | 0.5853 | 0.6236 | 0.6204 | -0.0012 | +0.0371 | +0.0339 |
| 29 | 0.5814 | 0.5807 | 0.6230 | 0.6303 | -0.0007 | +0.0416 | +0.0489 |
| 47 | 0.5892 | 0.5895 | 0.6198 | 0.6190 | +0.0004 | +0.0306 | +0.0299 |

## Three-seed mean contrasts

| Contrast | Mean $\Delta$BA | Seed range | Reader 95% CI | Crossed 95% CI | Positive seeds |
|---|---:|---:|---:|---:|---:|
| label_smoothing_minus_gru | -0.0005 | [-0.0012, +0.0004] | [-0.0022, +0.0011] | [-0.0034, +0.0024] | 1/3 |
| gaze_kd_minus_gru | +0.0364 | [+0.0306, +0.0416] | [+0.0199, +0.0534] | [+0.0168, +0.0569] | 3/3 |
| text_gaze_kd_minus_gru | +0.0376 | [+0.0299, +0.0489] | [+0.0223, +0.0529] | [+0.0191, +0.0571] | 3/3 |
| text_gaze_kd_minus_gaze_kd | +0.0011 | [-0.0032, +0.0073] | [-0.0030, +0.0052] | [-0.0046, +0.0069] | 1/3 |
| text_gaze_teacher_minus_gaze_teacher | +0.0020 | [-0.0011, +0.0039] | [-0.0042, +0.0083] | [-0.0067, +0.0105] | 2/3 |
