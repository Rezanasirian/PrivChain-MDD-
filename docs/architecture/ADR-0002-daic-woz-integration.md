# ADR-0002 — Phase 1 baseline model & real DAIC-WOZ integration

- **Status:** Accepted
- **Date:** 2026-06-30
- **Phase:** 1 (Centralized Multimodal Baseline)
- **Related objectives:** H4 (prototype), H5 (evaluation)

## Context

Phase 1 needs a centralized multimodal model that establishes the diagnostic
accuracy ceiling before federation/DP are added. The real DAIC-WOZ corpus
(~300 GB, behind a Data Use Agreement at <https://dcapswoz.ict.usc.edu/>) is
**not present in the development/CI environment**, so the model and training
loop must be fully exercisable on the mock dataset while remaining drop-in
compatible with real data.

## Decisions

### 1. One tensor contract for mock and real data

Both `MockDaicWozDataset` and `DaicWozDataset` emit the same
`Sample`/`Batch` structure: per-modality float sequences `(T, D)` + true
`lengths`, plus `phq8_score` and binary `label`. Encoders/fusion/heads are
dimension-agnostic (input dims come from config or are inferred from real
features), so **no model code changes between mock and real runs**.

### 2. Modality treatment for real DAIC-WOZ

- **Audio** → COVAREP features (74-dim), one row per ~10 ms frame.
- **Video** → OpenFace `*_CLNF_AUs.txt` (metadata columns `frame/timestamp/
  confidence/success` dropped).
- **Text** → participant transcript turns concatenated and vectorized to a
  single `(1, D_text)` "length-1 sequence" so text reuses the float-sequence
  pipeline. The default vectorizer is a pure-NumPy **hashing** vectorizer
  (`HashingTextVectorizer`) — it needs no network or pretrained model, which
  matters because the environment is offline. TF-IDF (sklearn) is an opt-in
  alternative, consistent with the thesis plan ("TF-IDF + a simple classifier
  to start").

### 3. Sequence length handling

Interviews are ~15 minutes, so raw COVAREP/OpenFace sequences are tens of
thousands of frames. The loader **subsamples** (`frame_stride`) and
**truncates** (`max_frames`), then optionally **z-score standardizes** each
feature over time. These are config knobs in `configs/daic_woz.yaml`.

### 4. Model

`MultimodalDepressionModel` = three `SequenceEncoder`s (projection → optional
bi-GRU → masked mean-pool) → `ConcatFusion` → a binary classification head
(F1/ROC-AUC reported here) plus an optional PHQ-8 regression head (multi-task,
on by default since DAIC-WOZ ships PHQ-8 scores). The fusion module already
accepts a per-sample modality `presence` mask so it survives into Phase 2's
heterogeneous clients without a rewrite.

### 5. Minor structure extension

`src/privchain/training/` and `src/privchain/data/` are documented additions to
the mandated `src/privchain/` layout (CLAUDE.md §2/§9): training/experiment
utilities and dataset code need a testable home in `src`. The baseline model and
fusion live under the existing `fusion/` package; metrics live under `eval/`.

## Assumptions to validate against a real download

- COVAREP files have **no header**; OpenFace AU files **do**. Adjust
  `has_header` / `drop_columns` if your extraction differs (e.g., E-DAIC).
- Split files are `*_split_Depression_AVEC2017.csv` with columns
  `Participant_ID, PHQ8_Binary, PHQ8_Score`. The AVEC2017 **test** labels
  (`full_test_split.csv`) may be withheld in some releases.
- Transcripts are tab-separated with `speaker`/`value` columns and the
  participant labeled `Participant`.

**The real loader has not been run against the 300 GB corpus here.** Verify the
above in `configs/daic_woz.yaml`, then run
`python scripts/train_baseline.py --daic-config configs/daic_woz.yaml`.

## Consequences

- CI/tests run entirely on mock data and a tiny fabricated on-disk DAIC-WOZ
  fixture (`tests/unit/test_daic_woz_parsing.py`) that exercises the real
  parser without the real corpus.
- When real data lands, only `configs/daic_woz.yaml` should need tuning.
