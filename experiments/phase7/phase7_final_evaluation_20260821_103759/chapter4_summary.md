# Chapter 4 — final evaluation (real DAIC-WOZ)


## Main comparison (2-fold CV, mean±std)

| method | F1 | ROC-AUC | accuracy | held-out F1 |
|---|---|---|---|---|
| Centralized (no FL/DP) — cf. Xu et al. 2023 | 0.465±0.014 | 0.370±0.024 | 0.327±0.024 | 0.467 |
| Plain FedAvg | 0.437±0.002 | 0.416±0.038 | 0.292±0.011 | 0.485 |
| Personalized (reputation) — cf. Fan et al. 2025 | 0.437±0.002 | 0.420±0.033 | 0.292±0.011 | 0.485 |
| Proposed (full framework) | 0.437±0.002 | 0.399±0.009 | 0.292±0.011 | 0.357 |
| Proposed − reputation | 0.437±0.002 | 0.398±0.008 | 0.292±0.011 | 0.357 |

## Ablation (proposed vs. component removed)

| variant | F1 | ROC-AUC |
|---|---|---|
| Full framework | 0.437±0.002 | 0.399±0.009 |
| − reputation weighting | 0.437±0.002 | 0.398±0.008 |
| − federated distillation | 0.437±0.002 | 0.420±0.033 |

## DP privacy–utility (centralized DP-SGD, same total ε)

| allocation | F1 | ROC-AUC |
|---|---|---|
| adaptive per-modality | 0.358±0.054 | 0.364±0.022 |
| uniform per-modality | 0.358±0.054 | 0.364±0.022 |

## Inference latency (forward pass)

| batch size | ms/batch | ms/sample |
|---|---|---|
| 1 | 0.955 | 0.955 |
| 2 | 0.947 | 0.473 |
| 4 | 0.975 | 0.244 |
| 8 | 2.215 | 0.277 |
| 16 | 0.976 | 0.061 |
