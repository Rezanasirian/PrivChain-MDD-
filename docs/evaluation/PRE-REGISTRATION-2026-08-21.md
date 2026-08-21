# Pre-registration: corrected Chapter-4 campaign

- **Locked:** 2026-08-21, before the corrected campaign
- **Status:** Configuration frozen; campaign not yet executed
- **Supersedes:** the invalid Phase-7 run that observed official test outcomes

## Disclosure

The official DAIC-WOZ test split was previously processed by a run now known to
have discarded the best federated checkpoint, omitted client-side DP-SGD,
failed to propagate presence masks, and used a non-cross-modal distillation
mechanism. Its outcomes have therefore already been observed. They will be
reported as prior exposure and will not be used to alter this protocol.

## Locked decisions

- Session-level audio/video normalization remains enabled. The measured
  re-identification risks in `configs/privacy.yaml` were obtained under this
  normalization; changing it would require re-measuring and re-locking all
  risks. The near-constant mean/std statistic blocks are accepted as a stated
  limitation for this campaign.
- The thesis claim is narrowed to three modality-specific evaluations using a
  shared nearest-centroid attacker. No claim of three specialized attacker
  architectures will be made.
- All preprocessing and hyperparameters are the committed values in
  `configs/baseline.yaml`, `configs/federated.yaml`, `configs/privacy.yaml`,
  `configs/evaluation.yaml`, and `configs/daic_woz.yaml` at the locked commit.
- Main federation uses 10 clients, IID and Dirichlet partitions, and three
  predeclared partition seeds. Five- and twenty-client runs are sensitivity
  analyses with fewer predeclared seeds. Byzantine robustness is a separate
  injected-client scenario, not part of the headline table.
- `distill_anchor` is the primary KD mechanism; `distill_random` is its control
  and `distill_proximal` is the legacy comparison.

## Analysis sequence

1. Repeated stratified CV on pooled official train+dev reports internal
   uncertainty only. No CV result may change preprocessing, a hyperparameter,
   an arm, or the official-test campaign.
2. A one-fold/one-seed smoke run must pass and record runtime and disk estimates.
3. One locked official-test campaign runs every declared method and seed without
   feedback, configuration changes, or selective reruns.
4. Failures may be rerun only for a documented infrastructure error and must
   preserve the same config and random seed.

## Uncertainty and reporting

- Bootstrap resampling is at participant level.
- Predictions for the same participant are averaged across seeds before that
  participant enters the bootstrap; seeds are not treated as independent
  samples.
- Both supportive and non-supportive outcomes are reported. Old artifacts are
  marked superseded, not deleted.
