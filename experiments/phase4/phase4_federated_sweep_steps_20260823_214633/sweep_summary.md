# Federated sweep — inner-CV ranking

Ranked on inner-fold mean ROC-AUC over the official train split. Dev is
untouched here. Selecting the maximum over many configurations inflates
the winner's inner score, so treat the top row as a *candidate*: its
unbiased estimate is the single outer dev evaluation, run separately.

| arm | configuration | ROC-AUC | ±sd | F1 | steps | seconds |
|---|---|---|:--:|---|---|---|
| fedavg | batch_size=32 local_epochs=1 | 0.743 | 0.069 | 0.775 | 369 | 6.2 |
| fedavg | batch_size=8 local_epochs=1 | 0.743 | 0.069 | 0.775 | 369 | 5.4 |
| fedavg | batch_size=4 local_epochs=1 | 0.743 | 0.069 | 0.775 | 369 | 5.9 |
| fedavg | batch_size=32 local_epochs=3 | 0.722 | 0.039 | 0.775 | 1080 | 11.0 |
| fedavg | batch_size=8 local_epochs=3 | 0.722 | 0.039 | 0.775 | 1080 | 12.1 |
| fedavg | batch_size=4 local_epochs=3 | 0.722 | 0.039 | 0.775 | 1080 | 12.1 |
| fedavg | batch_size=32 local_epochs=5 | 0.528 | 0.196 | 0.690 | 1080 | 11.6 |
| fedavg | batch_size=8 local_epochs=5 | 0.528 | 0.196 | 0.690 | 1080 | 12.0 |
| fedavg | batch_size=4 local_epochs=5 | 0.528 | 0.196 | 0.690 | 1080 | 12.2 |
