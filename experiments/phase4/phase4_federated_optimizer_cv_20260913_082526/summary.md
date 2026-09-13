# Federated optimization — train-only inner CV (Phase 4)

Inner folds over the pooled official train split. Dev and test are
neither constructed nor scored. `pooled` bootstraps the paired
difference between an arm's out-of-fold predictions and the
`centralized` arm's.

| arm | ROC-AUC | ±sd | macro-F1 | pooled Δ | 95% CI | p |
|---|---|:--:|---|---|---|---|
| `centralized` | 0.719 | 0.092 | 0.597 | - | - | - |
| `fedavg_per_shard_sgd_lr0.1` | 0.629 | 0.124 | 0.557 | -0.120 | [-0.249, +0.014] | 0.083 |
| `fedavg_aggregate_counts_sgd_lr0.1` | 0.622 | 0.108 | 0.561 | -0.091 | [-0.223, +0.048] | 0.214 |
| `fedavg_per_shard_adam_lr0.0003` | 0.613 | 0.125 | 0.510 | -0.095 | [-0.220, +0.032] | 0.157 |
| `fedavg_aggregate_counts_adam_lr0.0003` | 0.603 | 0.121 | 0.509 | -0.179 | [-0.323, -0.037] | 0.016 |
| `fedavg_aggregate_counts_sgd_lr0.0003` | 0.524 | 0.123 | 0.472 | -0.239 | [-0.415, -0.050] | 0.008 |
| `fedavg_per_shard_sgd_lr0.0003` | 0.523 | 0.122 | 0.463 | -0.239 | [-0.422, -0.052] | 0.013 |
| `fedavg_per_shard_adam_lr0.1` | 0.450 | 0.155 | 0.313 | -0.258 | [-0.421, -0.093] | 0.001 |
| `fedavg_aggregate_counts_adam_lr0.1` | 0.423 | 0.146 | 0.287 | -0.276 | [-0.437, -0.117] | 0.000 |
