# Chapter 4 — final evaluation (real DAIC-WOZ)


## Main comparison (2-fold CV, mean±std)

| method | F1 | ROC-AUC | accuracy | held-out F1 |
|---|---|---|---|---|
| Centralized (no FL/DP) — cf. Xu et al. 2023 | 0.465±0.014 | 0.370±0.024 | 0.327±0.024 | 0.467 |
| Plain FedAvg | 0.429±0.010 | 0.390±0.032 | 0.319±0.038 | 0.375 |
| Personalized (reputation) — cf. Fan et al. 2025 | 0.419±0.019 | 0.391±0.032 | 0.319±0.038 | 0.240 |
| Proposed (full framework) | 0.422±0.016 | 0.393±0.037 | 0.301±0.020 | 0.412 |
| Proposed − reputation | 0.422±0.016 | 0.392±0.038 | 0.301±0.020 | 0.308 |

## Ablation (proposed vs. component removed)

| variant | F1 | ROC-AUC |
|---|---|---|
| Full framework | 0.422±0.016 | 0.393±0.037 |
| − reputation weighting | 0.422±0.016 | 0.392±0.038 |
| − federated distillation | 0.419±0.019 | 0.391±0.032 |

## DP privacy–utility (centralized DP-SGD, same total ε)

| allocation | F1 | ROC-AUC |
|---|---|---|
| adaptive per-modality | 0.358±0.054 | 0.364±0.022 |
| uniform per-modality | 0.358±0.054 | 0.364±0.022 |

## Inference latency (forward pass)

| batch size | ms/batch | ms/sample |
|---|---|---|
| 1 | 0.976 | 0.976 |
| 2 | 0.952 | 0.476 |
| 4 | 1.001 | 0.250 |
| 8 | 0.999 | 0.125 |
| 16 | 0.975 | 0.061 |
