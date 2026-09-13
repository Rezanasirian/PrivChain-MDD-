# Feature extraction — inner-CV ladder (ADR-0031)

Ranked on inner-fold mean ROC-AUC over the official train split; dev and
test are neither scored nor selected on. `wins` counts folds where the arm
beat the baseline, so an effect carried by one lucky fold is visible as
such. `pooled` bootstraps the difference between the two arms' pooled
out-of-fold predictions, which is the paired question worth asking.

Baseline arm: `segment_baseline` — the pipeline every committed result used.

| arm | ROC-AUC | ±sd | F1 | per-fold | wins | pooled | 95% CI | p |
|---|---|:--:|---|---|---|---|---|---|
| segment_ctd_a2 | 0.614 | 0.033 | 0.396 | +0.033 | 2/2 | -0.012 | [-0.095, +0.063] | 0.747 |
| segment_a2 | 0.600 | 0.010 | 0.392 | +0.018 | 2/2 | +0.013 | [-0.058, +0.079] | 0.768 |
| segment_baseline | 0.582 | 0.001 | 0.376 | +0.000 | 0/2 | - | - | - |
| segment_ctd | 0.569 | 0.029 | 0.389 | -0.012 | 1/2 | -0.003 | [-0.067, +0.066] | 0.954 |
