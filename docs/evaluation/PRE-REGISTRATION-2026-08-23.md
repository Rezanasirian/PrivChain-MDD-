# Pre-registration: corrected Chapter-4 campaign (re-lock)

- **Locked:** 2026-08-23, before the corrected campaign
- **Status:** Configuration frozen; campaign not yet executed
- **Supersedes:** `PRE-REGISTRATION-2026-08-21.md`, whose locked commit carried
  the class-weighting defect described below
- **Also supersedes:** the invalid Phase-7 run that observed official test outcomes

## Why this document replaces the 2026-08-21 lock

The previous lock froze "the committed values ... at the locked commit"
(`f78dd96`). That commit trains the centralized arm with a class-weighted BCE
and every federated and DP arm with an unweighted one, because
`train.class_weighting` never reached `FederatedClient` or `_eval_centralized_dp`
(ADR-0026). A campaign run against it would compare arms trained under different
objectives and would attribute the difference to federation.

Clause 4 of the previous protocol permits reruns for a documented infrastructure
error. An arm-dependent loss function is such an error. Because the fix changes
code, the earlier document no longer describes what would run, so it is replaced
rather than reinterpreted, and the correction is disclosed here rather than
applied silently.

## Disclosure

**Prior exposure, unchanged from the previous lock.** The official DAIC-WOZ test
split was previously processed by a run now known to have discarded the best
federated checkpoint, omitted client-side DP-SGD, failed to propagate presence
masks, and used a non-cross-modal distillation mechanism. Those outcomes have
already been observed. They are reported as prior exposure and were not used to
alter this protocol.

**New exposure from cross-validation.** The 10-fold CV of 2026-08-23
(`experiments/phase7/phase7_final_evaluation_20260823_190036`) has been observed.
It touched only the pooled official train+dev split; the official test split was
not read. It is what revealed the class-weighting defect. Its numbers were used
to diagnose a code defect, not to select a hyperparameter, an arm, or a
preprocessing choice. Under the previous protocol's clause 1 that distinction is
what separates a permitted diagnosis from a prohibited adjustment.

## Locked decisions

Unchanged from the 2026-08-21 lock except where stated.

- **New:** Every arm trains under the same objective. `train.class_weighting`
  applies to the centralized arm, to the DP arms, and to every federated client,
  each client measuring its weight on its own shard (ADR-0026). A shard holding
  a single class is left unweighted.
- Session-level audio/video normalization remains enabled. The measured
  re-identification risks in `configs/privacy.yaml` were obtained under this
  normalization; changing it would require re-measuring and re-locking all risks.
  The near-constant mean/std statistic blocks are accepted as a stated limitation.
- The thesis claim is narrowed to three modality-specific evaluations using a
  shared nearest-centroid attacker. No claim of three specialized attacker
  architectures will be made.
- All preprocessing and hyperparameters are the committed values in
  `configs/baseline.yaml`, `configs/federated.yaml`, `configs/privacy.yaml`,
  `configs/evaluation.yaml`, and `configs/daic_woz.yaml` at this document's
  locked commit. No configuration value changed in this re-lock; only the code
  defect was corrected.
- Main federation uses 10 clients, IID and Dirichlet partitions, and three
  predeclared partition seeds. Five- and twenty-client runs are sensitivity
  analyses with fewer predeclared seeds. Byzantine robustness is a separate
  injected-client scenario, not part of the headline table.
- `distill_anchor` is the primary KD mechanism; `distill_random` is its control
  and `distill_proximal` is the legacy comparison.

## Analysis sequence

1. Repeated stratified CV on pooled official train+dev reports internal
   uncertainty only. No CV result may change preprocessing, a hyperparameter, an
   arm, or the official-test campaign. A CV result **may** identify a code defect,
   which is corrected under clause 4 and disclosed in a re-lock such as this one.
2. A one-fold/one-seed smoke run must pass and record runtime and disk estimates.
3. One locked official-test campaign runs every declared method and seed without
   feedback, configuration changes, or selective reruns.
4. Failures may be rerun only for a documented infrastructure error and must
   preserve the same config and random seed.

## Uncertainty and reporting

- Bootstrap resampling is at participant level.
- Predictions for the same participant are averaged across seeds before that
  participant enters the bootstrap; seeds are not treated as independent samples.
- Both supportive and non-supportive outcomes are reported. Old artifacts are
  marked superseded, not deleted.
- The corrected campaign is **not** predicted to favour the proposed method. The
  fix removes a confound; if federation still fails to beat the centralized
  baseline, that is the reported result.
