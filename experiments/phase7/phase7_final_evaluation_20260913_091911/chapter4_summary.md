# Chapter 4 — final evaluation (real DAIC-WOZ)


## Main comparison (10-fold CV, mean±std)

| method | F1 | ROC-AUC | accuracy | held-out F1 |
|---|---|---|---|---|
| Centralized (no FL/DP) — cf. Xu et al. 2023 | 0.494±0.138 | 0.676±0.151 | 0.649±0.125 | 0.538 |
| Plain FedAvg | 0.319±0.172 | 0.502±0.208 | 0.495±0.161 | 0.538 |
| Personalized (reputation) — cf. Fan et al. 2025 | 0.306±0.184 | 0.464±0.186 | 0.497±0.125 | 0.538 |
| Proposed (full framework) | 0.289±0.222 | 0.469±0.199 | 0.503±0.165 | 0.538 |
| Proposed − reputation | 0.296±0.227 | 0.475±0.188 | 0.533±0.166 | 0.538 |

## Ablation (proposed vs. component removed)

| variant | F1 | ROC-AUC |
|---|---|---|
| Full framework | 0.289±0.222 | 0.469±0.199 |
| − reputation weighting | 0.296±0.227 | 0.475±0.188 |
| − federated distillation | 0.306±0.184 | 0.464±0.186 |

## DP privacy–utility (centralized DP-SGD, same total ε)

| allocation | F1 | ROC-AUC |
|---|---|---|
| adaptive per-modality | 0.376±0.150 | 0.409±0.220 |
| uniform per-modality | 0.368±0.147 | 0.409±0.220 |

## Inference latency (forward pass)

| batch size | ms/batch | ms/sample |
|---|---|---|
| 1 | 1.390 | 1.390 |
| 2 | 1.381 | 0.690 |
| 4 | 1.404 | 0.351 |
| 8 | 1.425 | 0.178 |
| 16 | 1.414 | 0.088 |
