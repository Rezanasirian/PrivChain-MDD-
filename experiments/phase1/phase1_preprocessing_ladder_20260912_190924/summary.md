# Feature extraction — inner-CV ladder (ADR-0031)

Ranked on inner-fold mean ROC-AUC over the official train split; dev and
test are neither scored nor selected on. `wins` counts folds where the arm
beat the baseline, so an effect carried by one lucky fold is visible as
such. `pooled` bootstraps the difference between the two arms' pooled
out-of-fold predictions, which is the paired question worth asking.

Baseline arm: `committed` — the pipeline every committed result used.

| arm | ROC-AUC | ±sd | F1 | per-fold | wins | pooled | 95% CI | p |
|---|---|:--:|---|---|---|---|---|---|
| speech+mean+valid+video+mean+success | 0.734 | 0.118 | 0.465 | +0.006 | 8/15 | +0.000 | [-0.084, +0.082] | 0.989 |
| video+mean | 0.732 | 0.091 | 0.467 | +0.004 | 7/15 | +0.007 | [-0.041, +0.051] | 0.743 |
| committed | 0.728 | 0.094 | 0.457 | +0.000 | 0/15 | - | - | - |
| video+mean+success | 0.698 | 0.098 | 0.395 | -0.030 | 4/15 | -0.048 | [-0.111, +0.013] | 0.121 |
| video+mean+valid | 0.689 | 0.099 | 0.438 | -0.039 | 6/15 | -0.038 | [-0.123, +0.041] | 0.377 |
