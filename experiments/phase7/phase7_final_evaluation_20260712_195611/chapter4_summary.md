# Chapter 4 — final evaluation (mock data)

> On mock data the depression label is random; these are placeholder numbers demonstrating the tables. DAIC-WOZ fills them in.


## Main comparison (10-fold CV, mean±std)

| method | F1 | ROC-AUC | accuracy | held-out F1 |
|---|---|---|---|---|
| Centralized (no FL/DP) — cf. Xu et al. 2023 | 0.837±0.075 | 0.333±0.136 | 0.727±0.116 | 0.800 |
| Plain FedAvg | 0.256±0.363 | 0.640±0.178 | 0.356±0.210 | 0.000 |
| Personalized (reputation) — cf. Fan et al. 2025 | 0.256±0.363 | 0.603±0.221 | 0.356±0.210 | 0.000 |
| Proposed (full framework) | 0.256±0.363 | 0.603±0.221 | 0.356±0.210 | 0.000 |
| Proposed − reputation | 0.256±0.363 | 0.621±0.199 | 0.356±0.210 | 0.000 |

## Ablation (proposed vs. component removed)

| variant | F1 | ROC-AUC |
|---|---|---|
| Full framework | 0.256±0.363 | 0.603±0.221 |
| − reputation weighting | 0.256±0.363 | 0.621±0.199 |
| − federated distillation | 0.256±0.363 | 0.603±0.221 |

## DP privacy–utility (centralized DP-SGD, same total ε)

| allocation | F1 | ROC-AUC |
|---|---|---|
| adaptive per-modality | 0.256±0.363 | 0.356±0.300 |
| uniform per-modality | 0.256±0.363 | 0.374±0.300 |

## Inference latency (forward pass)

| batch size | ms/batch | ms/sample |
|---|---|---|
| 1 | 41.915 | 41.915 |
| 2 | 52.387 | 26.193 |
| 4 | 33.116 | 8.279 |
| 8 | 72.935 | 9.117 |
| 16 | 102.319 | 6.395 |
