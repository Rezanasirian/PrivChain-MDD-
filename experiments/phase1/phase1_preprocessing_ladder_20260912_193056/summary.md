# Feature extraction — inner-CV ladder (ADR-0031)

Ranked on inner-fold mean ROC-AUC over the official train split; dev and
test are neither scored nor selected on. `wins` counts folds where the arm
beat the baseline, so an effect carried by one lucky fold is visible as
such. `pooled` bootstraps the difference between the two arms' pooled
out-of-fold predictions, which is the paired question worth asking.

Baseline arm: `committed` — the pipeline every committed result used.

| arm | ROC-AUC | ±sd | F1 | per-fold | wins | pooled | 95% CI | p |
|---|---|:--:|---|---|---|---|---|---|
| speech+mean+valid+corpus | 0.760 | 0.132 | 0.499 | +0.033 | 8/15 | +0.043 | [-0.049, +0.126] | 0.337 |
| committed | 0.728 | 0.094 | 0.457 | +0.000 | 0/15 | - | - | - |
| speech+mean+valid | 0.721 | 0.092 | 0.435 | -0.007 | 6/15 | -0.016 | [-0.087, +0.049] | 0.668 |
| speech | 0.719 | 0.106 | 0.500 | -0.008 | 6/15 | +0.020 | [-0.034, +0.078] | 0.463 |
| speech+mean | 0.715 | 0.104 | 0.424 | -0.013 | 4/15 | +0.004 | [-0.058, +0.066] | 0.928 |
| speech+corpus | 0.700 | 0.100 | 0.455 | -0.027 | 4/15 | -0.026 | [-0.104, +0.038] | 0.473 |
