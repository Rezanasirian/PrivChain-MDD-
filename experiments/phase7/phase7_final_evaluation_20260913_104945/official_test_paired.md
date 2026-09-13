# Official test — paired participant-level comparisons

47 participants (14 positive), 3 seeds averaged per participant before resampling.

| comparison | Δ ROC-AUC | 95% CI | p | significant |
|---|---|---|---|:--:|
| centralized − fedavg | +0.299 | [+0.121, +0.481] | 0.000 | yes |
| centralized − personalized | +0.247 | [+0.088, +0.421] | 0.001 | yes |
| centralized − proposed | +0.106 | [-0.035, +0.248] | 0.150 | no |
| centralized − proposed_no_reputation | +0.123 | [-0.021, +0.270] | 0.100 | no |
| fedavg − personalized | -0.052 | [-0.104, -0.008] | 0.015 | yes |
| fedavg − proposed | -0.193 | [-0.303, -0.103] | 0.000 | yes |
| fedavg − proposed_no_reputation | -0.175 | [-0.276, -0.090] | 0.000 | yes |
| personalized − proposed | -0.141 | [-0.226, -0.071] | 0.000 | yes |
| personalized − proposed_no_reputation | -0.123 | [-0.201, -0.061] | 0.000 | yes |
| proposed − proposed_no_reputation | +0.017 | [+0.002, +0.038] | 0.029 | yes |
