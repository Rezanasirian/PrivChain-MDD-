# Federated sweep — inner-CV ranking

Protocol-matched centralized reference: **ROC-AUC 0.691±0.102** on the same inner folds.

Ranked on inner-fold mean ROC-AUC over the official train split. Dev is
untouched here. Selecting the maximum over many configurations inflates
the winner's inner score, so treat the top row as a *candidate*: its
unbiased estimate is the single outer dev evaluation, run separately.

``steps/client`` is what a run actually spent; ``rounds`` shows run vs
planned, so a configuration truncated by early stopping is visible.

| arm | configuration | ROC-AUC | ±sd | F1 | steps/client | rounds | seconds |
|---|---|---|:--:|---|---|---|---|
| fedavg | batch_size=4 local_epochs=5 | 0.667 | 0.079 | 0.419 | 120 | 12/12 | 13.6 |
| fedavg | batch_size=32 local_epochs=3 | 0.654 | 0.084 | 0.417 | 120 | 40/40 | 17.3 |
| fedavg | batch_size=8 local_epochs=3 | 0.654 | 0.084 | 0.417 | 120 | 40/40 | 18.3 |
| fedavg | batch_size=32 local_epochs=5 | 0.653 | 0.095 | 0.435 | 120 | 24/24 | 16.2 |
| fedavg | batch_size=8 local_epochs=5 | 0.653 | 0.095 | 0.435 | 120 | 24/24 | 16.5 |
| proposed | batch_size=32 local_epochs=3 | 0.651 | 0.096 | 0.416 | 144 | 40/40 | 34.8 |
| proposed | batch_size=8 local_epochs=3 | 0.651 | 0.096 | 0.416 | 144 | 40/40 | 35.4 |
| proposed | batch_size=32 local_epochs=5 | 0.650 | 0.069 | 0.436 | 134 | 24/24 | 27.5 |
| proposed | batch_size=8 local_epochs=5 | 0.650 | 0.069 | 0.436 | 134 | 24/24 | 28.1 |
| fedavg | batch_size=32 local_epochs=1 | 0.649 | 0.117 | 0.394 | 75 | 75/120 | 10.2 |
| fedavg | batch_size=8 local_epochs=1 | 0.649 | 0.117 | 0.394 | 75 | 75/120 | 15.2 |
| proposed | batch_size=4 local_epochs=3 | 0.649 | 0.084 | 0.386 | 132 | 20/20 | 24.0 |
| fedavg | batch_size=4 local_epochs=3 | 0.645 | 0.107 | 0.414 | 120 | 20/20 | 14.4 |
| proposed | batch_size=32 local_epochs=1 | 0.644 | 0.118 | 0.413 | 111 | 69/120 | 41.5 |
| proposed | batch_size=8 local_epochs=1 | 0.644 | 0.118 | 0.413 | 111 | 69/120 | 43.3 |
| fedavg | batch_size=4 local_epochs=1 | 0.640 | 0.113 | 0.394 | 102 | 51/60 | 16.2 |
| proposed | batch_size=4 local_epochs=5 | 0.639 | 0.092 | 0.430 | 127 | 12/12 | 20.2 |
| proposed | batch_size=4 local_epochs=1 | 0.634 | 0.116 | 0.404 | 141 | 54/60 | 39.2 |
