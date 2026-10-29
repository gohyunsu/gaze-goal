# Post hoc simple-baseline results

**Status:** complete; post hoc.

Both baselines use the frozen reader/article partitions and nested inner selection.

| Model | BA | AUROC | NLL |
|---|---:|---:|---:|
| Dwell time/word LR | 0.5887 | 0.6292 | 0.6635 |
| Gaze-summary LR | 0.6066 | 0.6493 | 0.6572 |
| Gaze-summary MLP | 0.5942 | 0.6359 | 0.6632 |
| Gaze GRU | 0.5866 | 0.6178 | 0.6710 |
| Distilled GRU | 0.6205 | 0.6708 | 0.6427 |

| Contrast | Delta BA | Reader 95% CI | Positive contexts |
|---|---:|---:|---:|
| Distilled GRU - Dwell time/word LR | +0.0317 | [+0.0084, +0.0550] | 18/25 |
| Distilled GRU - Gaze-summary LR | +0.0139 | [+0.0000, +0.0275] | 17/25 |
| Gaze-summary LR - Gaze GRU | +0.0200 | [-0.0005, +0.0399] | 19/25 |
