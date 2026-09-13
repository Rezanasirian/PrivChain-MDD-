# Feature extraction — inner-CV ladder (ADR-0031)

Ranked on inner-fold mean ROC-AUC over the official train split; dev and
test are neither scored nor selected on. `wins` counts folds where the arm
beat the baseline, so an effect carried by one lucky fold is visible as
such. `pooled` bootstraps the difference between the two arms' pooled
out-of-fold predictions, which is the paired question worth asking.

Baseline arm: `committed` — the pipeline every committed result used.

| arm | ROC-AUC | ±sd | F1 | per-fold | wins | pooled | 95% CI | p |
|---|---|:--:|---|---|---|---|---|---|
| committed | 0.728 | 0.094 | 0.457 | +0.000 | 0/15 | - | - | - |
| speech+egemaps | 0.531 | 0.124 | 0.289 | -0.197 | 1/15 | -0.213 | [-0.328, -0.106] | 0.000 |
