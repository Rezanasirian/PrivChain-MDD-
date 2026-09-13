# Feature extraction — inner-CV ladder (ADR-0031)

Ranked on inner-fold mean ROC-AUC over the official train split; dev and
test are neither scored nor selected on. `wins` counts folds where the arm
beat the baseline, so an effect carried by one lucky fold is visible as
such. `pooled` bootstraps the difference between the two arms' pooled
out-of-fold predictions, which is the paired question worth asking.

Baseline arm: `committed` — the pipeline every committed result used.

| arm | ROC-AUC | ±sd | F1 | per-fold | wins | pooled | 95% CI | p |
|---|---|:--:|---|---|---|---|---|---|
| committed | 0.670 | 0.069 | 0.391 | +0.000 | 0/15 | - | - | - |
| speech | 0.630 | 0.064 | 0.413 | -0.040 | 4/15 | +0.005 | [-0.088, +0.107] | 0.870 |
| speech+mean+valid | 0.620 | 0.077 | 0.392 | -0.050 | 6/15 | -0.020 | [-0.129, +0.088] | 0.747 |
| speech+corpus | 0.612 | 0.063 | 0.394 | -0.058 | 3/15 | -0.065 | [-0.178, +0.052] | 0.270 |
| speech+mean+valid+corpus | 0.601 | 0.087 | 0.388 | -0.069 | 2/15 | -0.044 | [-0.145, +0.065] | 0.404 |
