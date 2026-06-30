# ADR-0001 — Synthetic mock DAIC-WOZ dataset for Phase 0

- **Status:** Accepted
- **Date:** 2026-06-30
- **Phase:** 0 (Environment & Data Setup)
- **Related objective:** H4 (prototype)

## Context

Real DAIC-WOZ access requires a Data Use Agreement with weeks of lead time, and
the data must never enter the repository (CLAUDE.md §7). The pipeline still
needs reproducible, correctly shaped multimodal inputs to develop and test
against in CI before real data arrives.

## Decision

Provide an in-memory, seed-reproducible synthetic dataset
(`privchain.data.mock_daic_woz.MockDaicWozDataset`) that mirrors the three
DAIC-WOZ modalities and PHQ-8 labels:

- **audio** — log-mel features `(num_frames, n_mels)`
- **video** — facial features `(num_frames, n_features)`
- **text** — token embeddings `(num_tokens, embed_dim)`
- **label** — `phq8_score >= depression_cutoff` (DAIC-WOZ binary cutoff = 10)

All dimensions come from `configs/baseline.yaml` (no hardcoded hyperparameters).
Sequence lengths vary per session; `collate_fn` right-pads each batch to its
longest member and returns true length tensors for downstream masking.

### Assumptions documented here (per CLAUDE.md §9)

- **Feature shapes** are placeholders chosen to resemble common DAIC-WOZ feature
  extractions (e.g., 40-d log-mels, ~49 OpenFace AUs). They will be reconciled
  with the real feature pipeline in Phase 1.
- **Binary depression cutoff = PHQ-8 ≥ 10**, the standard DAIC-WOZ convention.
- Features are i.i.d. Gaussian noise — sufficient for shape/plumbing tests, but
  carries **no diagnostic signal**, so accuracy numbers on mock data are
  meaningless by design.

## Consequences

- CI runs without sensitive data; tests assert tensor shapes/dtypes only.
- A disk writer (`write_mock_dataset`) emits a DAIC-WOZ-like per-session layout
  under `data/mock/` (git-ignored) for manual inspection.
- When real data arrives, a parallel `DaicWozDataset` will implement the same
  `Sample`/`Batch` interface so downstream code is unchanged.
