# ADR-0019 — Per-session z-scoring, and what it did to the audio verdict

- **Status:** Accepted
- **Date:** 2026-08-13
- **Phase:** 1 (data) — with consequences for the Phase 1 ablation and the Phase 6 attack
- **Related objectives:** H4 (prototype), H5 (evaluation), H1 (the risk ordering that drives budget allocation)

## Context

`configs/daic_woz.yaml` had carried `standardize: true` since the loader was
written, and nobody had asked what it did. It z-scored each feature column
**within a session**:

```python
mean = matrix.mean(axis=0, keepdims=True)  # over time, within one participant
std = matrix.std(axis=0, keepdims=True)
matrix = (matrix - mean) / (std + 1e-6)
```

So every participant's audio has per-channel mean 0 and standard deviation 1 *by
construction*. Absolute pitch level, formant positions and overall vocal energy
are removed before any model or attacker sees the data. Those are:

- the cues a speaker-identification attacker relies on — which ADR-0017 was
  measuring; and
- among the best-attested acoustic markers of depression (reduced vocal energy,
  flattened prosody) — which ADR-0016 concluded audio did not carry.

A preprocessing default was therefore bounding both headline findings, and doing
it invisibly.

## Decision

`standardize: bool` becomes `normalization: session | corpus | none`, per
modality:

- **session** — the old behaviour, kept and now named.
- **corpus** — per-channel mean/std fitted on the **train split only** and applied
  to every split, so between-subject differences survive while scale is still
  controlled. Fitting on train alone is what keeps it leakage-free.
- **none** — the values as recorded.

The on-disk cache now stores the **raw** subsampled matrix and normalization is
applied at load, so switching mode costs nothing; re-parsing the 36 MB COVAREP
files per mode would have made this experiment impractical.

## Result

Full 7-arm ablation under each mode. 86 train / 21 selection / 34 dev, three
seeds, with the bootstrap intervals of ADR-0020.

**Regression check first:** under `session`, every arm reproduces ADR-0016's
committed numbers exactly (audio 0.503 ± 0.004, video 0.494 ± 0.026, text
0.710 ± 0.010, all three 0.740 ± 0.016). The refactor added an option without
changing behaviour.

| normalization | audio alone | video alone | text alone | all three |
|---|---|---|---|---|
| session | 0.503 | 0.494 | 0.710 | 0.740 |
| corpus | 0.553 | 0.416 | 0.710 | 0.736 |
| none | **0.619** | 0.335 | 0.710 | 0.718 |

**Text is identical to three decimals in all three rows.** Text embeddings never
went through this code path, so that column is the internal control confirming
the manipulation did what it claimed and nothing else.

### Audio: the "no signal" verdict was partly an artifact

Audio alone moves 0.503 → 0.619 when per-session z-scoring is removed — a swing
of 0.116, in exactly the predicted direction. ADR-0016 reported audio at
"0.503, chance" and read that as a property of the modality. It was in
substantial part a property of a preprocessing default.

The honest statement is **not** "audio carries signal after all": at 34 sessions
audio's own interval under `none` is [0.392, 0.808], which contains chance. It is
that **we cannot measure what audio carries on this corpus**, and the earlier
claim that it carries nothing was not supported by an experiment that had
normalized the relevant information away.

### Video moves the other way

Video alone falls 0.494 → 0.416 → 0.335, ending well *below* chance. OpenFace
action-unit intensities are already calibrated quantities on a common scale, so
removing standardization does not restore information — it lets between-subject
scale differences in, and those are noise here. Below chance means the arm is
anti-predictive, which on 34 sessions is itself within noise, but the direction
is consistent across both non-session modes.

That the two modalities respond in **opposite directions** is the clearest
evidence that this parameter is doing real work rather than adding jitter.

### The combined model does not care

0.740 / 0.736 / 0.718 across the three modes — indistinguishable. Text dominates
the fused model regardless of how audio and video are scaled, which is consistent
with ADR-0016's one surviving finding.

## The default stays `session`

Deliberately, for three reasons:

1. The full model is insensitive, so there is no performance argument either way.
2. It is what every committed result used, so prior numbers remain valid and
   comparable rather than needing an asterisk.
3. Picking normalization per modality by whichever scored best on dev would be
   another dev-fitted decision, which is precisely what ADR-0020 warns against.
   Audio prefers `none`, video prefers `session`; choosing both by looking would
   be fitting the preprocessing to the evaluation split.

What changes is not the default but what may be **claimed**: normalization is now
a stated, tested variable rather than a silent one.

## Consequences

- **ADR-0016 is amended.** "Audio and video carry no usable signal on their own"
  is withdrawn as stated; it holds under `session` normalization and does not
  survive the change to `none` for audio.
- **ADR-0017 is amended, and the prediction that motivated this work was wrong.**
  The critical review argued that per-session z-scoring removes speaker-identity
  cues, so ADR-0017's rates had to be *underestimates* for audio and video, with
  the ordering safe because the bias was conservative. Re-running the attack
  under `none` says the opposite:

  | representation (raw_pca) | session | none |
  |---|---|---|
  | audio | 22.2× chance | **10.6×** |
  | video | 21.6× chance | **5.1×** |
  | text | 6.6× chance | 6.6× (unchanged — the control) |

  Removing normalization *lowered* measured leakage. The mechanism is the
  attacker's metric, not the data: the attacker scores by cosine similarity,
  which is scale-invariant, and raw COVAREP channels share a large common offset
  across all subjects. Unnormalized, every subject's vector points in nearly the
  same direction and discriminability collapses. Session z-scoring strips that
  shared offset and leaves the between-subject *directions* the attacker can use.

  The consequence is worse than a wrong sign. **The measured ordering is not
  stable**: audio is the most re-identifying under both modes, but video moves
  from second (21.6×) to last (5.1×, below text). So `configs/privacy.yaml`'s
  1.00 / 0.97 / 0.29 rests on one preprocessing choice, and the audio-vs-video
  part of it does not survive changing that choice. Text's invariance across
  modes is what confirms this is a real effect rather than run-to-run noise.
- **A second candidate confound remains untested.** `frame_stride: 30` subsamples
  COVAREP to ~3.3 Hz, which cannot represent prosodic dynamics either. It was
  held fixed here so that any movement was attributable to normalization alone;
  it is the obvious next thing to vary before anyone concludes anything about
  what audio can contribute.
