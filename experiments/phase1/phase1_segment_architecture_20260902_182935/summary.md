# Segment-aligned architecture — inner-CV comparison

Ranked on inner-fold mean ROC-AUC over the official train split; dev and
test are neither scored nor selected on. `Δ` is a paired bootstrap on
out-of-fold scores
averaged across seeds, resampled **per participant** — the unit that is
actually independent. `wins` counts folds beating the baseline arm.

Baseline arm: `document+concat`

| arm | ROC-AUC | ±sd | PR-AUC | F1 | OOF Δ | 95% CI | sig | wins | seconds |
|---|---|:--:|---|---|---|---|:--:|---|---|
| segments+attn | 0.747 | 0.169 | 0.592 | 0.510 | +0.035 | [-0.113, +0.186] | no | 11/15 | 2.7 |
| document+gated | 0.676 | 0.051 | 0.516 | 0.463 | -0.009 | [-0.097, +0.072] | no | 10/15 | 2.3 |
| aligned+dropout | 0.665 | 0.092 | 0.519 | 0.384 | -0.002 | [-0.128, +0.121] | no | 8/15 | 1.3 |
| aligned+quality_gated | 0.649 | 0.089 | 0.481 | 0.389 | +0.023 | [-0.088, +0.132] | no | 6/15 | 1.2 |
| document+concat | 0.637 | 0.070 | 0.494 | 0.438 | — | — | — | 0/15 | 2.4 |
| segments+av+gated | 0.637 | 0.101 | 0.489 | 0.386 | -0.076 | [-0.199, +0.039] | no | 6/15 | 2.4 |
| aligned+session_norm | 0.632 | 0.084 | 0.503 | 0.383 | -0.053 | [-0.173, +0.066] | no | 9/15 | 1.4 |
| aligned+huber | 0.623 | 0.061 | 0.452 | 0.365 | -0.048 | [-0.157, +0.064] | no | 4/15 | 1.4 |
