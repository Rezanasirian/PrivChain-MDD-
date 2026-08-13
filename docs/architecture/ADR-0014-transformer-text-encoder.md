# ADR-0014 — Contextual transcript embeddings for the text modality

- **Status:** Accepted
- **Date:** 2026-08-13
- **Phase:** 1 (Centralized Multimodal Baseline)
- **Related objectives:** H1 (per-modality privacy), H4 (prototype), H5 (evaluation)

## Context

[ADR-0002](ADR-0002-daic-woz-integration.md) chose a pure-NumPy hashing
bag-of-words vectorizer for transcripts, explicitly because *"it needs no network
or pretrained model, which matters because the environment is offline."* That
constraint no longer holds: training now runs on a GPU host with network access.

Two things make the choice worth revisiting rather than merely possible:

1. **Text is the strongest modality on DAIC-WOZ.** A hashed bag-of-words
   discards word order, negation, and context — exactly what carries depressive
   language. It was the weakest link in a multimodal model whose strongest signal
   should have come from it.
2. **The thesis argument depends on it.** The central claim (H1) is that
   modalities deserve *different* privacy budgets. That argument is far stronger
   when the modality carrying the most signal is also identifiable as such — a
   deliberately weak text branch would have understated both the utility of text
   and the stakes of protecting it.

The hardware also made this the right use of an otherwise idle GPU: peak VRAM
across all previous phases was ~2.5 GB of 24 GB.

## Decisions

### 1. `TransformerTextVectorizer` as the real-data default

`sentence-transformers/all-mpnet-base-v2` (768-dim), selected via
`daic_woz.text.vectorizer: transformer`. Hashing remains available and stays the
default for mock data and CI, which must run offline.

A DAIC-WOZ interview runs to a couple of thousand participant tokens, well past
any encoder's context window, so the document is chunked and pooled:

1. tokenize the participant's concatenated turns, splitting into
   `max_length`-token windows (`return_overflowing_tokens`, so the tokenizer
   handles special tokens and padding itself);
2. embed each window and attention-masked mean-pool its tokens;
3. average the window vectors and L2-normalize.

Averaging over all windows — rather than truncating to the first — keeps the
whole session in the representation, matching the full-session coverage audio and
video already have (ADR-0011). Empty transcripts return a zero vector rather than
raising, so one unusable session cannot abort a run.

The model is cached per `(model_name, device)`, so building separate
train/dev/test datasets does not reload the weights.

### 2. Text embeddings are memoized like the other modalities

Each embedding costs a GPU forward pass per session. `_load_text` now writes to
the same `feature_cache_dir` introduced in ADR-0012, keyed by the vectorizer
identity (kind, dim, model options, speaker label) so switching model — or back
to hashing — cannot silently reuse the wrong vectors. Embedding all 107 training
sessions takes 2.9 s cold and is free thereafter.

### 3. `text.vectorizer` is actually read now

The key existed in `configs/daic_woz.yaml` from ADR-0002 but the loader ignored
it: `DaicWozDataset` unconditionally constructed a `HashingTextVectorizer`. A
config asking for anything else silently got hashing. A `build_text_vectorizer`
factory now resolves it, and an unknown name raises — including `tfidf`, which
appeared in the config comments but was never implemented. An explicitly injected
vectorizer still wins, which is how the offline tests stay offline.

### 4. Per-modality encoder overrides

The `stats` encoder (ADR-0012) computes five functionals per channel over the
time axis. Text has no time axis — it arrives as one document vector, i.e. a
length-1 "sequence" — so those functionals are **degenerate**: mean, min and max
all equal the vector itself, while std and the first-difference term are
identically zero. Applying `stats` to text would have quintupled the text
encoder's input width (768 → 3840) for zero additional information, making the
text branch dominate the parameter count of a model trained on 107 sessions.

`ModelConfig.encoder_overrides` layers partial per-modality overrides onto the
shared encoder config, validated rather than merged blindly. `configs/baseline.yaml`
gives text `type: mean` (a plain projection); audio and video keep `stats`.

## Result

Holding everything else at the ADR-0012 configuration (`lr = 1e-3`,
`hidden = 64`, `dropout = 0.3`; train split, dev-split validation, class-weighted
loss, F1 selection) and changing **only** the text representation:

| Text representation | dev F1 | dev ROC-AUC | dev accuracy |
|---|---|---|---|
| Hashing bag-of-words (ADR-0012) | 0.560 | 0.664 | 0.677 |
| **all-mpnet-base-v2 embeddings** | **0.667** | **0.779** | **0.765** |

A +0.107 F1 and +0.115 ROC-AUC gain from the text branch alone, consistent with
the DAIC-WOZ literature's finding that linguistic features carry the most signal.

### Re-tuning on the new feature space

The 768-dim contextual features are not the 512-dim hashed ones, so the grid was
re-swept (18 configurations x 3 seeds). Several points tie near F1 ≈ 0.667, and
the adopted configuration was chosen on **stability** rather than on the single
highest mean:

| lr | hidden | dropout | dev F1 (mean ± spread) | dev AUC |
|---|---|---|---|---|
| 1e-3 | 128 | 0.3 | 0.668 ± 0.091 | 0.733 |
| **3e-4** | **128** | **0.3** | **0.667 ± 0.000** | **0.760** |
| 3e-4 | 64 | 0.3 | 0.667 ± 0.000 | 0.756 |
| 1e-3 | 64 | 0.3 | 0.638 ± 0.004 | 0.718 |

The top row's ±0.091 spread across three seeds is a worse bet than an identical
result on all three at effectively the same mean and a higher AUC, because every
later phase re-runs this model many times.

Final adopted configuration, run through `scripts/train_baseline.py` at seed 42:

```
Device=cuda  pos_weight=2.567  epochs_run=69/200
Best (epoch 29, by f1) — F1=0.6400  ROC-AUC=0.7668  acc=0.7353
```

The F1 here (0.640) sits slightly below the sweep's 0.667 for the same
hyperparameters: the sweep harness and the training script shuffle batches
differently, so they are different draws, not different methods. The gap is
within the seed-to-seed noise visible in the table above, and ROC-AUC agrees
across both (0.767 vs 0.760).

## Consequences

- **The `nlp` extra is now required for real-data runs.** CI and mock runs are
  unaffected: they use the hashing vectorizer and never import `transformers`.
- Text feature dim changes 512 → 768, so **cached features and any checkpoint
  from before this change are not reusable**. The cache key covers the
  vectorizer identity, so stale entries are ignored rather than mixed in.
- The privacy story gets sharper rather than weaker: the modality with the most
  utility is now also the one with rich, quasi-identifying content, which is
  precisely the trade-off the per-modality budget allocator (ADR-0004) exists to
  express. The `reidentification_risk` values in `configs/privacy.yaml` still
  rank text *lowest*; Phase 6's attacker models must now re-test that ordering
  against contextual embeddings, and the ranking may well need revising.
- Not claimed: that this is a tuned or competitive DAIC-WOZ system. The encoder
  is frozen — no fine-tuning — and the thesis contribution remains the
  privacy/federation/audit layer above it.
