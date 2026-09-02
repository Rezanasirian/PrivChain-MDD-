# ADR-0028 — Capability-conditioned logit fusion, and one pre-registered test

- **Status:** Accepted
- **Date:** 2026-09-02
- **Phase:** 1 (centralized multimodal baseline), objective H4, with direct effects on Phases 2–4
- **Extends** ADR-0027 (segment-aligned architecture), ADR-0015 (shared evaluation protocol),
  ADR-0021 (federated on real data).

## Context

ADR-0027's ladder ran on real DAIC-WOZ at 20 seeds
(`experiments/phase1/phase1_segment_architecture_20260902_185430`) and produced one
statistically significant result. It was negative: session-level audio and video alongside
segmented text are **worse** than the concat baseline, −0.111 ROC-AUC, CI [−0.222, −0.006].
Segment-level text with attention was the only arm that moved upward (+0.107) and its
interval still spanned zero.

The obvious reading — drop audio and video — optimizes the wrong thing. This thesis is about
detecting depression when modalities are **missing**; a text-only model has nothing to say
to a client that holds only audio. The question is not whether the full-modality model beats
text alone. It is whether the model degrades gracefully when a modality is absent.

Two facts constrain any answer:

- **107 participants.** The inner-CV pool is the official train split; the paired participant
  bootstrap resamples those 107. The largest effect the ladder found was not significant, so
  a design that asks seven questions will produce seven inconclusive rows.
- **DP-SGD.** Anything that changes what one participant contributes to one optimizer step
  changes the sensitivity the accountant assumes.

## Decisions

### D1 — Fuse at the logit, not the embedding

Every fusion in this project so far combines modalities in embedding space. An absent
modality is zeroed, and that zero still travels through a projection whose weights were
fitted with the slot occupied, so it shifts the pre-activation. The model can be taught to
tolerate that; it cannot be made to ignore it.

Each modality now produces its own logit and its own gate score, and a masked softmax over
the **present** modalities mixes them. An absent modality's weight is exactly zero and the
survivors renormalize, so an audio-only client is scored by its audio expert alone.

Late fusion is also the cheapest thing that can do this — three scalar heads, no cross-modal
parameters — which matters at this sample size and again under DP noise.

### D2 — Every parameter lands in a modality group, and none in `shared`

Branch, expert head, gate scorer **and the optional PHQ-8 head** all live under
`encoders.<modality>.`, so `map_parameter_groups` and `param_group` classify them correctly
with no new rule.

The PHQ head is the one that nearly went wrong: written as a top-level
`regressors.<modality>` it reads only one modality's session vector but does not match the
prefix, so it was filed under `shared` — averaged across clients that never held the
modality and charged to the wrong epsilon. That is precisely the bug ADR-0027 found in
`fusion.gates.<modality>`, reappearing in a new model within one commit of being written
down, which is why the grouping test is parametrized over `use_phq_regression` rather than
run once with it off.

A consequence worth stating rather than discovering later: a pure late-fusion MoE has **no
shared parameters at all**. Under capability-aware aggregation each modality's weights are
averaged only across clients that hold it, which is the intended behaviour, but it also
means parameter averaging carries no cross-modal transfer. Federated distillation
(ADR-0024) becomes the only channel by which a text-holding client can teach an audio-only
one. That is a property of the architecture, not a defect, and Phase 4 should report it.

### D3 — One capability per participant per optimizer step

Training must show the model the audio-only case. Forwarding each participant under all four
masks every step would do that, and would break the accounting: those four views are not
independent observations, and Opacus would clip a gradient already summed over them, so the
per-participant sensitivity the accountant assumes is not what the optimizer enforces.

So the capability varies across **epochs** instead. Each participant sees the deployment mix
exactly once per cycle — with `4:3:2:1`, a 10-epoch cycle of 4 full, 3 audio+text, 2
audio-only, 1 text-only.

The cycle is a shuffled bag keyed by `derive_seed(seed, participant, cycle)`, not an
independent draw per epoch. Independent draws leave the rarest pattern to chance: over 10
epochs a participant could plausibly never be seen text-only, and the metric this ADR is
about is precisely the rare-pattern one. Fractions become visit counts through exact
rational arithmetic, never rounding, for the same reason.

The epoch is pushed into the dataset by the trainer (`on_epoch_start`) rather than counted
inside it: a DataLoader with workers copies the dataset per process, and a counter would
advance independently in each copy.

### D4 — The four committed patterns, from one file

`configs/federated.yaml` already declares `full`, `audio_text`, `audio_only`, `text_only`
with population fractions. The training schedule, the counterfactual evaluation and the
federated deployment all read that list through
`privchain.federated.capability_patterns`, and a contract test asserts the three agree.

`video_only` and `video_text` are deliberately **not** added. They would widen the claim and
add tests for a client type the deployment scenario does not contain — and since no pattern
trains video alone, `video_only` would become an artificial argmin for the
worst-capability metric. Extending the capability set is future work.

### D5 — One loss, no new coefficients

`BCEWithLogitsLoss` on the fused logit. `phq_loss_weight: 0.0` for these runs.

Three terms were considered and rejected:

- **Consistency** (pull the incomplete-modality prediction toward the full-modality one) is
  backwards given this corpus. ADR-0027 measured the full-modality prediction as *worse*
  than text alone, so the term would distil the strongest capability toward a weaker teacher.
- **Worst-group (DRO)** as a batch-wise max over capabilities is a max of four noisy
  estimates at batch size 32 — biased and high-variance. A per-group EMA would fix that and
  introduce its own hyperparameter.
- **Subset weighting** adds a coefficient that 107 participants cannot support tuning.

The weak experts are not starved: when the mask is `audio_only` the masked softmax puts
weight exactly 1 on audio, so the audio expert is directly supervised.

### D6 — The gate prior is recorded, not swept

Gate score biases start at `text +2, audio 0, video 0` — about `0.79 / 0.11 / 0.11` under
full modality. This encodes what the ladder already measured instead of making the optimizer
rediscover it.

It is derived from inner-CV on the **train** split of a previous run, so it reads no
evaluation data, and it is **not swept**: sweeping it would be a hyperparameter search this
corpus cannot afford, and the resulting number would be selection. The manifest records the
values, the source and `gate_bias_swept: false`.

### D7 — One inferential claim, named before the run

```
Primary estimand:  Δ = AUC_audio_only(MoE) − AUC_audio_only(baseline)
```

`audio_only` is the only committed pattern without text, the strongest modality, so it is
the most direct test of the missing-modality claim. It is fixed in code
(`PRIMARY_CAPABILITY`), chosen from the pre-existing ladder results before this model
existed, and the manifest asserts `primary_capability_fixed_before_run`.

The test is a paired participant-level bootstrap: participants are resampled once and both
arms are re-scored on the same draw, so fold difficulty and class mix cancel.

**`min` over capabilities is secondary and carries no significance claim.** A minimum over
four noisy estimates is biased downward by an amount that depends on each arm's spread, so a
model with more variance across capabilities is penalized regardless of whether it is
better; and the argmin changes between arms and between bootstrap replicates, so the
estimand is not fixed. It is reported with its argmin distribution, as description.

Also secondary, without tests: macro capability AUC, per-capability ROC/PR-AUC, the
full-modality guardrail, and the learned gate weights.

### D8 — Evaluation is counterfactual, not partitioned

Every held-out participant is scored once under **each** capability. Splitting the fold four
ways would shrink each estimate to a quarter of the data; scoring everyone under every mask
keeps all 107 for each capability. The four estimates are then correlated, which the shared
participant-level bootstrap preserves rather than pretends away.

## Consequences

- The comparison is two arms and one test, not a ladder. A null result is informative here
  in a way seven inconclusive rows were not.
- Selection uses the same capability mix as training, so the chosen epoch is not the one that
  happens to suit full-modality clients.
- DP is untouched by construction: one masked view per participant per step. Phase 3 must
  still re-measure clipping rate and gradient norm on the new architecture and report utility
  again — the adjacency definition does not change, but the numbers will.
- The claim this supports is narrower and more defensible than "the model is better":

  > Capability-conditioned logit fusion improves the worst access pattern relative to
  > conventional fusion, without requiring every modality to be present.
