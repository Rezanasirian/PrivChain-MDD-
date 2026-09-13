# Federated optimization — train-only inner CV (Phase 4)

Inner folds over the pooled official train split. Dev and test are
neither constructed nor scored. `pooled` bootstraps the paired
difference between an arm's out-of-fold predictions and the
`centralized` arm's.

| arm | ROC-AUC | ±sd | macro-F1 | pooled Δ | 95% CI | p |
|---|---|:--:|---|---|---|---|
| `centralized` | 0.719 | 0.092 | 0.597 | - | - | - |
| `fedavg_per_shard_sgd_lr0.1` | 0.629 | 0.124 | 0.557 | -0.120 | [-0.249, +0.014] | 0.083 |
| `fedavg_per_shard_adam_lr0.0003` | 0.613 | 0.125 | 0.510 | -0.095 | [-0.220, +0.032] | 0.157 |
| `fedavg_per_shard_sgd_lr0.0003` | 0.523 | 0.122 | 0.463 | -0.239 | [-0.422, -0.052] | 0.013 |
| `fedavg_per_shard_adam_lr0.1` | 0.450 | 0.155 | 0.313 | -0.258 | [-0.421, -0.093] | 0.001 |
