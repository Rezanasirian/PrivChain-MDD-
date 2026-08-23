# Chapter 4 — final evaluation (real DAIC-WOZ)


## Main comparison (10-fold CV, mean±std)

| method | F1 | ROC-AUC | accuracy | held-out F1 |
|---|---|---|---|---|
| Centralized (no FL/DP) — cf. Xu et al. 2023 | 0.494±0.138 | 0.676±0.151 | 0.649±0.125 | 0.538 |
| Plain FedAvg | 0.294±0.186 | 0.439±0.186 | 0.548±0.157 | 0.375 |
| Personalized (reputation) — cf. Fan et al. 2025 | 0.254±0.189 | 0.402±0.197 | 0.489±0.103 | 0.400 |
| Proposed (full framework) | 0.377±0.161 | 0.440±0.200 | 0.523±0.148 | 0.462 |
| Proposed − reputation | 0.318±0.194 | 0.448±0.207 | 0.507±0.163 | 0.444 |

## Ablation (proposed vs. component removed)

| variant | F1 | ROC-AUC |
|---|---|---|
| Full framework | 0.377±0.161 | 0.440±0.200 |
| − reputation weighting | 0.318±0.194 | 0.448±0.207 |
| − federated distillation | 0.254±0.189 | 0.402±0.197 |

## DP privacy–utility (centralized DP-SGD, same total ε)

| allocation | F1 | ROC-AUC |
|---|---|---|
| adaptive per-modality | 0.469±0.133 | 0.539±0.224 |
| uniform per-modality | 0.472±0.127 | 0.539±0.224 |

## Inference latency (forward pass)

| batch size | ms/batch | ms/sample |
|---|---|---|
| 1 | 0.985 | 0.985 |
| 2 | 0.983 | 0.491 |
| 4 | 1.011 | 0.253 |
| 8 | 1.016 | 0.127 |
| 16 | 1.023 | 0.064 |
