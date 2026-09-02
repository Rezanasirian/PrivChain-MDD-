# Segment-aligned architecture — inner-CV comparison

Ranked on inner-fold mean ROC-AUC over the official train split; dev and
test are neither scored nor selected on. `Δ` is a paired bootstrap on
out-of-fold scores
averaged across seeds, resampled **per participant** — the unit that is
actually independent. `wins` counts folds beating the baseline arm.

Baseline arm: `document+concat`

| arm | ROC-AUC | ±sd | PR-AUC | F1 | OOF Δ | 95% CI | sig | wins | seconds |
|---|---|:--:|---|---|---|---|:--:|---|---|
| segments+attn | 0.774 | 0.163 | 0.645 | 0.532 | +0.107 | [-0.034, +0.255] | no | 77/100 | 3.3 |
| document+concat | 0.665 | 0.078 | 0.547 | 0.427 | — | — | — | 0/100 | 2.2 |
| document+gated | 0.662 | 0.079 | 0.542 | 0.429 | -0.003 | [-0.042, +0.037] | no | 45/100 | 2.7 |
| aligned+dropout | 0.652 | 0.128 | 0.507 | 0.434 | +0.058 | [-0.050, +0.173] | no | 46/100 | 1.4 |
| segments+av+gated | 0.638 | 0.122 | 0.516 | 0.399 | -0.111 | [-0.222, -0.006] | yes | 46/100 | 2.6 |
| aligned+session_norm | 0.635 | 0.100 | 0.519 | 0.413 | -0.019 | [-0.132, +0.093] | no | 40/100 | 1.3 |
| aligned+quality_gated | 0.630 | 0.084 | 0.487 | 0.402 | -0.031 | [-0.130, +0.062] | no | 34/100 | 1.2 |
| aligned+huber | 0.619 | 0.080 | 0.469 | 0.404 | -0.038 | [-0.141, +0.062] | no | 30/100 | 1.2 |
