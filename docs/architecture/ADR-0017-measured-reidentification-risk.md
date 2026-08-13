# ADR-0017 — Measuring per-modality re-identification risk on real DAIC-WOZ

- **Status:** Accepted
- **Date:** 2026-08-13
- **Phase:** 6 (measurement) / 3 (the allocation it is supposed to justify)
- **Related objectives:** H5 (empirical privacy evaluation), H1 (per-modality budget allocation)

## Context

`configs/privacy.yaml` hands audio the tightest privacy budget (ε = 2) and text
the loosest (ε = 8), on the strength of three numbers — `reidentification_risk`
0.9 / 0.6 / 0.3 — that were **assumed and never measured**. The config has said
so since Phase 3: *"This ordering is a hypothesis to be validated with attacker
models in Phase 6."*

ADR-0016 turned that open question into a blocker. The modality ablation found
audio alone at ROC-AUC 0.503 and video at 0.494 (both chance) while text alone
reached 0.710 of the full model's 0.740. So the current allocation spends its
strongest protection on the modality contributing nothing and its weakest on the
modality carrying nearly all the signal. Whether that is defensible depends
entirely on the *risk* axis, which nobody had measured.

The risk ordering is also stale on its face. It was set when text was a hashed
bag-of-words. Text is now a 768-dim contextual embedding of what a participant
actually said (ADR-0014) — free-text clinical disclosure, plausibly the most
re-identifying content in the corpus rather than the least.

## Why the existing Phase 6 harness could not answer this

`scripts/run_attack_eval.py` has only ever run on the mock corpus, where it
manufactures several views per subject by adding feature-space **jitter** to one
session and checking whether the attacker can match the noisy copies. It always
can — that is true by construction and says nothing about whether a modality
carries identity.

Real DAIC-WOZ gives exactly one recording per participant, so the jitter had to
be replaced with something real.

## Method

**Views are disjoint contiguous stretches of the one session.** Each session is
cut into `num_segments = 6` spans; the attacker enrols a template from 3 and
re-identifies the other 3 against the whole candidate pool by nearest cosine
centroid (`ReidentificationAttacker`, unchanged from ADR-0007).

- audio / video — spans of the cached frame matrix
- text — spans of the participant's **turn list**, each embedded separately
  through the same transformer (`DaicWozDataset.text_segment_vectors`)

Splitting by item count rather than by timestamp keeps every span non-empty: a
participant can fall silent for minutes, and equal-duration bins would leave some
stretch with no turns at all (`privchain.segmentation.contiguous_spans`).

**Three representations, because width is a confound.** Nearest-centroid accuracy
grows with feature width, and the modalities are not the same width. Reporting
only raw features would partly measure *how many numbers each modality hands the
attacker*:

| row | representation | width |
|---|---|---|
| `raw` | the summary the configured encoder consumes pre-projection — functionals for `stats`, a masked mean for `mean` | audio 370, video 100, text 768 |
| `raw_pca` | the same, projected to a common width, **fitted on the enrollment rows only** so the projection is never shaped by the rows it is scored against | 64 |
| `encoded` | the trained encoder's output — width-matched by construction | `out_dim` |

**The ordering claim rests on the width-matched rows.**

**Pool and protocol.** Candidates are every participant in train ∪ selection ∪ dev
— the test split stays sealed for the Chapter 4 table. Chance is 1 / pool size.
`encoded` rows are broken out by group, because re-identifying a subject the
encoder was fitted on is an easier problem than re-identifying an unseen one.
Repeated over `train.seeds` with a random enrollment subset each time, reported
mean ± std.

**No DP is applied.** This is the *unprotected* risk the allocation claims to be
calibrated against. Attack success versus ε is a different question and stays
with `run_attack_eval.py`.

**Negative control.** The same attack with shuffled probe labels is printed
beside every table. If it does not collapse to chance, the pipeline is measuring
itself rather than the data.

## The limitation, stated plainly

Enrolling and probing inside one recording shares channel, room acoustics,
lighting, clothing and topic. It therefore **overstates** identity leakage
relative to real cross-session re-identification, where an adversary matches a
person against a recording made elsewhere.

DAIC-WOZ has one session per participant, so no cross-session measurement is
possible on this corpus at all. The mitigation is that the bias applies
**identically to all three modalities**, so the *ordering* — which is the only
thing the allocation depends on — survives even though the absolute rates do not
transfer. Absolute numbers must be reported as an upper bound, never as "the
re-identification rate of DAIC-WOZ audio".

A second limitation: this measures what the *features we chose* leak, not what
the modality leaks in principle. Audio is subsampled to ~3.3 Hz COVAREP
functionals (ADR-0012); a purpose-built speaker-identification front-end would
almost certainly extract more. The same caveat as ADR-0016's utility result, on
the other axis.

## Result

141 participants (train ∪ selection ∪ dev), none skipped, chance = 1/141 = 0.0071.
Three seeds, mean ± std. `experiments/phase6/phase6_reid_risk_20260813_151757/`.

| Modality | `raw` | `raw_pca` (64) | `encoded` (128) |
|---|---|---|---|
| audio | 0.173 ± 0.003 (24.4×) | **0.158 ± 0.004 (22.2×)** | 0.107 ± 0.011 (15.1×) |
| video | 0.175 ± 0.006 (24.7×) | **0.153 ± 0.010 (21.6×)** | 0.177 ± 0.007 (25.0×) |
| text | 0.051 ± 0.006 (7.2×) | **0.046 ± 0.004 (6.6×)** | 0.015 ± 0.006 (2.1×) |

Negative control (shuffled probe labels): 0.0024 / 0.0071 / 0.0071 — chance, for
all three. The pipeline is measuring the data, not itself.

Seen-vs-unseen is a *small* effect next to the between-modality gaps (audio
`encoded` 0.111 fitted vs 0.098 unseen; text 0.017 vs 0.003), and video actually
runs the other way (0.156 vs 0.219). So this is representation leakage, not
memorisation of the training subjects — which is what makes it a property of the
modality rather than of one fitted model.

### 1. The assumed ordering survives — audio, video ≫ text

On every representation, text is by far the least re-identifying: 6.6× chance at
matched width against 22× for audio and video, and only 2.1× once passed through
the trained encoder. The thesis hypothesis that text is the *lowest*-risk
modality is **confirmed on our own data and our own features**, and the assumed
0.3 is nearly exact — text normalises to 0.29 of the riskiest modality.

### 2. The audio > video gap does not survive

0.158 ± 0.004 versus 0.153 ± 0.010 at matched width: indistinguishable. The
configured 0.9 / 0.6 asserts a separation the data does not support, and the
`encoded` row reverses it outright (video 25.0× vs audio 15.1×) — the
depression-trained audio encoder discards speaker identity that the video encoder
retains. Which of the two is "riskier" therefore depends on the representation,
which is itself a reason not to bake a 1.5× ratio between them into the budget.

Measured normalised risk from the width-matched row: **audio 1.00, video 0.97,
text 0.29** (against the assumed 0.9 / 0.6 / 0.3).

### 3. This resolves ADR-0016's discomfort — in the thesis's favour

ADR-0016 found the allocation "spends its tightest privacy budget protecting the
modality that contributes nothing, and its loosest budget on the modality that
carries essentially all of the signal", and called that uncomfortable. With the
risk axis measured, it is not a defect but the **best case** for per-modality DP:

| Modality | measured risk (× chance) | standalone utility (AUC) | budget it should get |
|---|---|---|---|
| audio | 22.2× | 0.503 (chance) | tight — leaks a lot, contributes nothing |
| video | 21.6× | 0.494 (chance) | tight — same |
| text | 6.6× | **0.710** | loose — leaks least, contributes most |

The risk axis and the utility axis **point the same way**. Protecting what is
exposed costs almost no accuracy here, because what is exposed is not what is
useful. A uniform budget cannot exploit that; per-modality allocation can. This
is the cleanest argument for H1 the project has produced, and it only exists
because both axes were measured rather than assumed.

### 4. What this does not say

The absolute rates are an upper bound (same-session enrol/probe, see above), and
they describe *these* features — 3.3 Hz COVAREP functionals, OpenFace AUs, mpnet
embeddings — not the modalities in principle. A purpose-built speaker-ID front-end
would very likely push audio higher.

## Consequences

- Per the scope agreed before the run, this ADR **measures and reports only**.
  `configs/privacy.yaml` is not modified here: changing `reidentification_risk`
  or switching `allocation.mode` to `inverse_risk` has Chapter 3 consequences and
  is a deliberate decision to take now that the numbers exist. The open choice is
  narrow — the ordering stands, so what is on the table is whether to replace
  0.9 / 0.6 / 0.3 with the measured 1.00 / 0.97 / 0.29 (which would close audio's
  and video's budgets to roughly equal) and whether to derive ε from risk
  automatically rather than pin it.
- **ADR-0016's blocker is cleared.** Phase 3's headline claim now rests on a
  measurement rather than an assumption, and the measurement supports it.
- Mock re-identification numbers are now known to be uninformative under this
  protocol — mock features are i.i.d. per session, so a *different* stretch
  identifies nobody, and the attack lands exactly at chance. This is asserted as
  a unit test (`test_mock_sessions_carry_no_within_session_identity`) so nobody
  reads a mock run as evidence again.
- `ReidentificationAttacker` gained `predict()` so callers can score subsets of
  the probes; `attack()` is now a thin wrapper over it.

---

## Amendment, 2026-08-13 (ADR-0019, ADR-0020) — the ordering is preprocessing-dependent

This ADR measured re-identification with `normalization: session` in force,
without noticing that the setting existed. ADR-0019 re-ran the same attack with
normalization removed:

| raw_pca (matched width) | session | none |
|---|---|---|
| audio | 0.158 (22.2×) | 0.075 (**10.6×**) |
| video | 0.153 (21.6×) | 0.036 (**5.1×**) |
| text | 0.046 (6.6×) | 0.046 (6.6×) |

**What stands.** Audio is the most re-identifying modality under both schemes.
Text sits at 6.6× under both — unchanged, because text embeddings never pass
through this code path, which makes that column an internal control confirming
the manipulation was specific. The negative control remains at chance in both
runs. The statistical basis is also unaffected by ADR-0020: 423 probe trials give
SE ≈ 0.017, so these gaps are real in a way the 34-session utility numbers are not.

**What does not stand.** The claim that audio and video are jointly the
high-risk modalities. Video ranks second under `session` and **last — below
text — under `none`**. Its position is an artifact of a preprocessing choice, not
a property of the modality.

**What that means for the budget.** `configs/privacy.yaml`'s measured risks
(1.00 / 0.97 / 0.29) were read off the `session` run. The audio-versus-text
ordering they encode survives; the audio-versus-video near-equality does not, and
this ADR's own conclusion that "the audio > video gap does not survive" turns out
to have been right for a different reason than stated — video is not
indistinguishable from audio, it is unstable.

Nothing downstream breaks, because ADR-0018 found the allocation makes no
measurable difference to utility either way. But no Chapter 3 argument may rest
on video's rank until the measurement is repeated on a representation chosen for
the attacker rather than inherited from the classifier.

**Also noted:** the `raw_pca` control is transductive — the projection is fitted
on the enrollment rows of the same subjects being identified. It is applied
identically to all three modalities, so the comparison is fair, but the absolute
`raw_pca` rates are mildly optimistic.
