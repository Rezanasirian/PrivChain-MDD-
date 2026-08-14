# Chapter 4 — final evaluation (mock data)

> On mock data the depression label is random; these are placeholder numbers demonstrating the tables. DAIC-WOZ fills them in.


## Main comparison (10-fold CV, mean±std)

| method | F1 | ROC-AUC | accuracy | held-out F1 |
|---|---|---|---|---|
| Centralized (no FL/DP) — cf. Xu et al. 2023 | 0.494±0.138 | 0.680±0.149 | 0.649±0.125 | 0.538 |
| Plain FedAvg | 0.401±0.166 | 0.584±0.239 | 0.408±0.132 | 0.429 |
| Personalized (reputation) — cf. Fan et al. 2025 | 0.382±0.161 | 0.573±0.256 | 0.355±0.102 | 0.444 |
| Proposed (full framework) | 0.386±0.164 | 0.580±0.260 | 0.364±0.114 | 0.444 |
| Proposed − reputation | 0.386±0.164 | 0.574±0.255 | 0.364±0.114 | 0.444 |

## Ablation (proposed vs. component removed)

| variant | F1 | ROC-AUC |
|---|---|---|
| Full framework | 0.386±0.164 | 0.580±0.260 |
| − reputation weighting | 0.386±0.164 | 0.574±0.255 |
| − federated distillation | 0.382±0.161 | 0.573±0.256 |

## DP privacy–utility (centralized DP-SGD, same total ε)

| allocation | F1 | ROC-AUC |
|---|---|---|
| adaptive per-modality | 0.103±0.207 | 0.539±0.224 |
| uniform per-modality | 0.103±0.207 | 0.539±0.224 |

## Inference latency (forward pass)

| batch size | ms/batch | ms/sample |
|---|---|---|
| 1 | 0.636 | 0.636 |
| 2 | 0.632 | 0.316 |
| 4 | 0.645 | 0.161 |
| 8 | 0.655 | 0.082 |
| 16 | 0.657 | 0.041 |
