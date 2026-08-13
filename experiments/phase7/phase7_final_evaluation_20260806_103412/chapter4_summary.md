# Chapter 4 — final evaluation (mock data)

> On mock data the depression label is random; these are placeholder numbers demonstrating the tables. DAIC-WOZ fills them in.


## Main comparison (10-fold CV, mean±std)

| method | F1 | ROC-AUC | accuracy | held-out F1 |
|---|---|---|---|---|
| Centralized (no FL/DP) — cf. Xu et al. 2023 | 0.841±0.035 | 0.500±0.379 | 0.727±0.052 | 0.833 |
| Plain FedAvg | 0.587±0.359 | 0.592±0.251 | 0.593±0.213 | 0.000 |
| Personalized (reputation) — cf. Fan et al. 2025 | 0.587±0.359 | 0.592±0.251 | 0.593±0.213 | 0.000 |
| Proposed (full framework) | 0.587±0.359 | 0.592±0.251 | 0.593±0.213 | 0.000 |
| Proposed − reputation | 0.587±0.359 | 0.567±0.276 | 0.593±0.213 | 0.000 |

## Ablation (proposed vs. component removed)

| variant | F1 | ROC-AUC |
|---|---|---|
| Full framework | 0.587±0.359 | 0.592±0.251 |
| − reputation weighting | 0.587±0.359 | 0.567±0.276 |
| − federated distillation | 0.587±0.359 | 0.592±0.251 |

## DP privacy–utility (centralized DP-SGD, same total ε)

| allocation | F1 | ROC-AUC |
|---|---|---|
| adaptive per-modality | 0.349±0.428 | 0.583±0.360 |
| uniform per-modality | 0.349±0.428 | 0.583±0.360 |

## Inference latency (forward pass)

| batch size | ms/batch | ms/sample |
|---|---|---|
| 1 | 54.512 | 54.512 |
| 2 | 160.965 | 80.483 |
| 4 | 118.181 | 29.545 |
| 8 | 174.712 | 21.839 |
| 16 | 178.321 | 11.145 |
