# ADR-0010 — Real DAIC-WOZ corpus ingestion: verified layout, label-column mismatch, and participant 440

- **Status:** Accepted
- **Date:** 2026-08-13
- **Phase:** 1 (Centralized Multimodal Baseline) — retroactively validates ADR-0002
- **Related objectives:** H4 (prototype), H5 (evaluation)

## Context

ADR-0002 built the real-data loader (`src/privchain/data/daic_woz.py`) against
the *documented* DAIC-WOZ layout and closed with an explicit caveat: *"The real
loader has not been run against the 300 GB corpus here."* That caveat is now
discharged. On 2026-08-13 the full corpus was downloaded from
<https://dcapswoz.ict.usc.edu/wwwdaicwoz/> onto the project GPU host and the
loader was executed against it for the first time.

The corpus as published is **196 files / 85.6 GB**: 189 per-participant archives
(`{pid}_P.zip`), 4 split CSVs, `documents.zip`, `util.zip`, and the AVEC2017
documentation PDF. Every file was verified byte-exact against the origin
server's `Content-Length`.

This ADR records three findings from that first real run: two assumptions in
ADR-0002 that turned out **wrong**, and one **data defect at source**.

## Decisions

### 1. Split label columns are not uniform — resolved with a per-split override

ADR-0002 assumed all split files use `Participant_ID, PHQ8_Binary, PHQ8_Score`.
The real headers are:

| File | Header |
|---|---|
| `train_split_Depression_AVEC2017.csv` | `Participant_ID, PHQ8_Binary, PHQ8_Score, Gender, PHQ8_*…` |
| `dev_split_Depression_AVEC2017.csv` | `Participant_ID, PHQ8_Binary, PHQ8_Score, Gender, PHQ8_*…` |
| `full_test_split.csv` | `Participant_ID, **PHQ_Binary**, **PHQ_Score**, Gender` |
| `test_split_Depression_AVEC2017.csv` | `participant_ID, Gender` — **unlabelled**, IDs only |

The test split drops the `8`. Because `_read_split_labels` resolved columns with
`(row.get(col) or "0")`, the absent `PHQ8_Binary` fell through to the default and
**every test-set label would have been read as 0** — 47 sessions, all negative,
with no error raised. Test F1/ROC-AUC in Chapter 4 would have been silently
meaningless.

Two changes:

- **`configs/daic_woz.yaml` gains `split_label_columns`**, a per-split override
  layered over the defaults. `test` maps to `PHQ_Binary`/`PHQ_Score`.
- **A missing configured column is now a `ValueError`**, not a default. Silent
  label corruption is the worst possible failure mode for this project, so the
  loader refuses to guess and names both the missing column and the actual
  header in the message.

Verified after the fix: `train n=107 (30 positive)`, `dev n=34 (11)`,
`test n=47 (14)`.

Note `test_split_Depression_AVEC2017.csv` carries **no labels at all** — it is
the blind AVEC2017 challenge split. `full_test_split.csv` is the labelled one and
remains what `splits.test` points at.

### 2. Participant 440 is excluded — its archive is truncated at source

`440_P.zip` fails to open: *"End-of-central-directory signature not found."* This
is **not** a download error. The file is byte-identical to the origin server's
copy (md5 `77ea835d…`, size 619,905,024 = exactly 591.0 MiB — a round number
consistent with a truncated upload), re-downloading reproduces it exactly, and
all 190 other archives pass `unzip -t` and contain all 8 required files.

Salvaging the intact deflate streams recovers 5 of the 10 entries; truncation
begins mid-`440_CLNF_hog` and everything after it in the archive is simply
absent:

| Recovered | Lost |
|---|---|
| `440_AUDIO.wav`, `440_CLNF_AUs.txt`, `440_CLNF_features.txt`, `440_CLNF_features3D.txt`, `440_CLNF_gaze.txt` | `440_CLNF_hog`, `440_CLNF_pose.txt`, `440_COVAREP.csv`, `440_FORMANT.csv`, **`440_TRANSCRIPT.csv`** |

Participant 440 belongs to the **dev** split (`PHQ8_Binary=1`, `PHQ8_Score=19`).
It has lost its text modality entirely and its configured audio features
(`COVAREP`), leaving only video.

**Decision: exclude 440 from the dev split** via a new `exclude_participants`
config key, taking dev from 35 to 34 sessions (12 → 11 positive).

Alternatives considered and rejected:

- *Keep it as a missing-modality case.* Defensible — Phase 2 clients are already
  modality-heterogeneous and `ConcatFusion` accepts a presence mask — but it
  would make one dev session structurally unlike every other, for a split used
  only for model selection. Not worth the confound.
- *Reconstruct the transcript with ASR (Whisper).* Rejected. DAIC-WOZ transcripts
  are human-produced with speaker turns (`Ellie`/`Participant`) and manual
  timings; the text modality reads **only the participant's** turns. Machine
  transcription without reliable diarization would put a differently-distributed
  document into one session, which is worse than dropping it.

The salvaged 440 files are kept out of `data/daic_woz/440_P/` deliberately: a
partially-populated participant directory could be picked up by a future loader
change without anyone noticing. Exclusion is expressed in config, where it is
visible.

### 3. `CLNF_hog` is not extracted

Extracted per-participant data is ~118 GB, of which **~87 GB is `CLNF_hog`**
alone. The video modality reads `CLNF_AUs` (ADR-0002 §2) and no phase of the
plan uses raw HOG descriptors. Extraction skips `*CLNF_hog*`, bringing on-disk
extracted data to ~34 GB and leaving headroom for checkpoints and experiment
artifacts on the 294 GB host.

The archives are retained in `data/daic_woz/raw/`, so HOG remains recoverable
without re-downloading if a later phase wants it.

## Assumptions from ADR-0002 that were confirmed correct

Verified directly against the corpus, so these are no longer assumptions:

- `*_COVAREP.csv` — no header, comma-separated, **74** features per row. ✅
- `*_CLNF_AUs.txt` — has a header; `frame, timestamp, confidence, success` are
  present and dropped, leaving **20** AU columns. ✅
- `*_TRANSCRIPT.csv` — tab-separated with `start_time / stop_time / speaker /
  value`, participant turns labelled `Participant`. ✅ (Files use CRLF endings;
  the loader's `newline=""` handling reads them correctly.)
- Per-participant directory template `{pid}_P`. ✅

Feature dims inferred from real data: `{audio: 74, video: 20, text: 512}`.
Parsed features are finite, and transcripts are non-empty (e.g. participant 303
yields 1,965 participant words).

## Consequences

- The ADR-0002 caveat is discharged: the loader is now **executed and verified**
  against the real corpus, not just a fabricated fixture.
- CI is unaffected — the new behaviours are covered by three tests in
  `tests/unit/test_daic_woz_parsing.py` using the existing tiny on-disk fixture,
  so no real data is required.
- Any future split whose headers differ needs only a `split_label_columns`
  entry; a mismatch that goes unnoticed now fails loudly at load time.
- Chapter 4 must report dev **n=34**, not 35, and state the 440 exclusion with
  the reason above.
