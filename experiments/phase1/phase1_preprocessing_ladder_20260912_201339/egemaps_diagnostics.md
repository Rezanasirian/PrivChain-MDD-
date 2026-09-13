# eGeMAPS post-hoc diagnostics (ADR-0032)

Run on the same server and checkout as the ladder, on the 107-participant
inner-CV pool only (dev/test untouched). Commands were one-off `uv run python`
snippets; the numbers below are copied from their stdout.

## 1. Extraction sanity (all 188 participants)

Per-participant eGeMAPS `F0semitoneFrom27.5Hz_sma3nz_amean` against the median
voiced COVAREP F0 converted to semitones from 27.5 Hz:

| check | value |
|---|---|
| Pearson r, same participant | 0.966 |
| Pearson r, pairing shifted by one participant | -0.032 |
| eGeMAPS F0 (st): mean 27.3, sd 4.9, range 17.7–35.7 | |
| COVAREP F0 (st): mean 27.8, sd 4.9 | |

The features are correctly aligned to participants and on the expected scale.

## 2. Normalized input scale (train split, corpus normalization)

`speech+egemaps` audio matrix (86 x 88): per-dimension sd mean 1.004
(min 0.551, max 1.113), |max| 9.9. No scaling failure.

## 3. Audio-only linear probe (5 inner folds, pooled out-of-fold ROC-AUC)

StandardScaler + LogisticRegression, features = the exact tensor the encoder
receives (COVAREP: the 370 `masked_statistics` functionals; eGeMAPS: 88).

| features | C=0.01 | C=0.1 | C=1 |
|---|---|---|---|
| COVAREP stats (370) | 0.608 | 0.621 | 0.619 |
| eGeMAPS (88) | 0.498 | 0.486 | 0.495 |

## 4. Per-fold multimodal losses (`results.jsonl`)

`speech+egemaps` validation loss reaches 2.66, 2.92, 2.14 on three folds
(committed: 0.94–1.0 everywhere), i.e. confident wrong predictions. With
88 inputs and 86 training sessions the linear 88 -> 294 -> 128 audio branch
can fit the training labels from noise and the fusion follows it.
