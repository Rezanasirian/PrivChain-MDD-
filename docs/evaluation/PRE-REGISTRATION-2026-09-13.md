# Pre-registration: Chapter-4 official-test campaign (re-lock)

- **Locked:** 2026-09-13, before the official-test campaign
- **Status:** Configuration frozen; campaign not yet executed
- **Supersedes:** `PRE-REGISTRATION-2026-08-23.md`, whose locked commit predates
  the preprocessing investigation, the eGeMAPS control, the membership-inference
  protocol correction, and the federated local-optimization decision

## Why this document replaces the 2026-08-23 lock

That lock froze "the committed values ... at this document's locked commit". Six
things have happened since, all of them on the pooled official train split or on
dev, none of them on official test:

1. **Frame validity and aggregation** (ADR-0030) and the **feature-extraction
   repair ladder** (ADR-0031). Three defects were identified — the acoustic
   branch read the interviewer's speech, decimation discarded frames, invalid
   frames entered the functionals — and every repair measured null on train-only
   inner CV. The committed preprocessing is unchanged and is now frozen.
2. **eGeMAPS acoustic control** (ADR-0032), rejected: 0.531 against 0.728, pooled
   Δ = -0.213, 95% CI [-0.328, -0.106].
3. **Membership-inference protocol** (ADR-0029): balanced pool, repeated over
   seeds. This changes how the attack is measured, not the model.
4. **Federated local optimization** (ADR-0033): the local optimizer, learning
   rate and class-weight mode moved from code into `configs/federated.yaml` at
   defaults that reproduce the previous behaviour exactly, after a train-only
   inner CV found no candidate beat the committed setting.
5. **Privacy semantics** (ADR-0009 amendment): a parameter-group ε is an
   auditable noise-allocation coefficient, not a modality-record guarantee for
   the released fused model. Wording only; no number changed.
6. **A dev read** of the federated arms
   (`experiments/phase4/phase4_federated_comparison_20260913_085150`).

Because code and config text changed — even where behaviour did not — the earlier
document no longer describes what would run, so it is replaced rather than
reinterpreted.

## Disclosure

**Prior exposure, unchanged.** The official DAIC-WOZ test split was processed by
a run now known to be invalid (discarded best federated checkpoint, no
client-side DP-SGD, presence masks not propagated, non-cross-modal
distillation). Those outcomes were observed. They are reported as prior exposure
and were not used to alter this protocol.

**New train-only exposure.** The ladders (ADR-0030/0031), the eGeMAPS control
(ADR-0032) and the federated optimization CV (ADR-0033) were all run on inner
folds of the pooled official train split. Every one of them returned a null or a
rejection, so none of them changed a default. Official test was neither
constructed nor scored in any of them.

**New dev exposure.** The 2026-09-13 federated comparison read the official dev
split once per arm at a threshold chosen on the selection split, as ADR-0020
specifies. Its numbers are reported in Chapter 4 as the dev result. They were not
used to select a hyperparameter for this campaign.

**Known tension to report, not to resolve by tuning.** Federated arms behave
differently on the two splits: the committed federated setting reaches 0.613
against centralized 0.719 on train-only inner folds (paired interval includes
zero), but 0.476 against 0.740 on the 34-session dev split. Both are reported.
No hyperparameter will be changed to reconcile them.

## Locked decisions

- All preprocessing and hyperparameters are the committed values in
  `configs/baseline.yaml`, `configs/federated.yaml`, `configs/privacy.yaml`,
  `configs/evaluation.yaml` and `configs/daic_woz.yaml` at this document's locked
  commit. The ADR-0030/0031 loader options exist but are **off**, which is the
  pipeline every committed result used.
- Federated clients train with Adam at the centralized learning rate and
  per-shard class weights (`class_weight_mode: null`, `local_optimizer: adam`,
  `local_learning_rate: null`), the defaults ADR-0033 confirmed.
- Every arm trains under the same objective, including the class weighting
  (ADR-0026).
- Session-level audio/video normalization remains enabled; the re-identification
  risks in `configs/privacy.yaml` were measured under it.
- The privacy claim is the composed participant ε for the released model; group
  ε values describe noise placement only (ADR-0009 amendment).
- Main federation uses 10 clients, IID partition, and the three seeds in
  `train.seeds`.
- `distill_anchor` is the primary KD mechanism.

## Analysis sequence

1. Train-only inner CV and the dev read are complete and reported above. No
   further result from either may change a hyperparameter, an arm, or this
   campaign.
2. One locked official-test campaign runs every declared method and seed without
   feedback, configuration changes, or selective reruns.
3. Failures may be rerun only for a documented infrastructure error, preserving
   the same config and seed.

## Amendment, 2026-09-13, after the first campaign run

The campaign of `experiments/phase7/phase7_final_evaluation_20260913_091911`
completed and its official-test metrics are reported: centralized 0.705, proposed
0.584, proposed−reputation 0.573, personalized 0.483, FedAvg 0.452 ROC-AUC across
three seeds. It wrote metric summaries only, so no paired comparison between arms
can be computed from it, and this protocol's reporting rules require
participant-level bootstrap.

Under clause 4 the campaign is rerun once, with **identical configuration and
identical seeds**, for the documented infrastructure reason that predictions were
not persisted. The code change is confined to writing
`official_test_scores.json`; no arm, threshold rule, hyperparameter or seed
changes, and the already-observed metrics above stand as the reported numbers.
Any difference between the two runs' metrics would itself be a reportable defect.

## Uncertainty and reporting

- Bootstrap resampling is at participant level; a participant's predictions are
  averaged across seeds before entering the bootstrap.
- Both supportive and non-supportive outcomes are reported. Superseded artifacts
  are marked, not deleted.
- The campaign is **not** predicted to favour the proposed method. If federation
  or DP fails to match the centralized baseline, that is the reported result, and
  the thesis states it as the measured cost of the privacy architecture.
