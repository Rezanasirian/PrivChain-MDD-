# Federated sweep — inner-CV ranking

Ranked on inner-fold mean ROC-AUC over the official train split. Dev is
untouched here. Selecting the maximum over many configurations inflates
the winner's inner score, so treat the top row as a *candidate*: its
unbiased estimate is the single outer dev evaluation, run separately.

``steps/client`` is what a run actually spent; ``rounds`` shows run vs
planned, so a configuration truncated by early stopping is visible.

| arm | configuration | ROC-AUC | ±sd | F1 | steps/client | rounds | seconds |
|---|---|---|:--:|---|---|---|---|
| proposed | class_weight_mode=per_shard num_clients=3 | 0.718 | 0.076 | 0.464 | 115 | 69/120 | 30.4 |
| fedavg | class_weight_mode=per_shard num_clients=3 | 0.697 | 0.078 | 0.455 | 72 | 72/120 | 6.2 |
| proposed | class_weight_mode=pooled_oracle num_clients=3 | 0.665 | 0.084 | 0.416 | 104 | 62/120 | 27.3 |
| proposed | class_weight_mode=off num_clients=5 | 0.658 | 0.133 | 0.430 | 125 | 78/120 | 38.8 |
| fedavg | class_weight_mode=per_shard num_clients=10 | 0.649 | 0.117 | 0.394 | 75 | 75/120 | 14.9 |
| proposed | class_weight_mode=per_shard num_clients=10 | 0.644 | 0.118 | 0.413 | 111 | 69/120 | 43.7 |
| fedavg | class_weight_mode=pooled_oracle num_clients=10 | 0.634 | 0.153 | 0.416 | 71 | 71/120 | 14.0 |
| fedavg | class_weight_mode=off num_clients=5 | 0.627 | 0.092 | 0.371 | 79 | 79/120 | 9.4 |
| fedavg | class_weight_mode=pooled_oracle num_clients=3 | 0.627 | 0.093 | 0.399 | 64 | 64/120 | 4.8 |
| fedavg | class_weight_mode=off num_clients=3 | 0.623 | 0.096 | 0.438 | 76 | 76/120 | 6.7 |
| proposed | class_weight_mode=per_shard num_clients=5 | 0.617 | 0.138 | 0.388 | 89 | 56/120 | 27.8 |
| proposed | class_weight_mode=pooled_oracle num_clients=10 | 0.615 | 0.153 | 0.416 | 116 | 72/120 | 45.6 |
| proposed | class_weight_mode=off num_clients=3 | 0.612 | 0.097 | 0.380 | 117 | 70/120 | 31.5 |
| proposed | class_weight_mode=pooled_oracle num_clients=5 | 0.609 | 0.135 | 0.422 | 96 | 60/120 | 30.0 |
| fedavg | class_weight_mode=per_shard num_clients=5 | 0.589 | 0.138 | 0.363 | 55 | 55/120 | 6.5 |
| proposed | class_weight_mode=off num_clients=10 | 0.551 | 0.188 | 0.388 | 118 | 74/120 | 46.4 |
| fedavg | class_weight_mode=pooled_oracle num_clients=5 | 0.538 | 0.152 | 0.334 | 56 | 56/120 | 6.7 |
| fedavg | class_weight_mode=off num_clients=10 | 0.502 | 0.175 | 0.343 | 70 | 70/120 | 13.7 |
