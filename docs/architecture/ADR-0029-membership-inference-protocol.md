# ADR-0029 — Membership-inference protocol: balanced pool, repeated over seeds

- **Status:** Accepted
- **Date:** 2026-09-04
- **Phase:** 6 (attacker models for privacy evaluation) — objective H5
- **Supersedes:** the membership-inference half of ADR-0007's evaluation protocol. The
  re-identification half is unchanged and still governed by ADR-0007 and ADR-0017.

## Context

The 2026-09-04 re-audit (`docs/evaluation/REPORT-2026-09-04-FA.md`, §6) left one
claim with evidence too weak to state either way: *"DP controls the membership-
inference attack."* The measured numbers were a non-private reference at
`AUC = 0.562` (advantage `0.124`) against `AUC ≈ 0.461–0.464` at every ε from
0.5 to 512. Three defects made that unusable as evidence:

1. **One seed.** Every other claim in the thesis is reported over 5–20 seeds with
   a bootstrap interval. A single membership-inference number has no spread
   attached, so "0.562 against 0.462" cannot be separated from run-to-run noise.
2. **No dose-response.** The AUC is flat across three orders of magnitude of ε.
   Under a working mechanism the attack should weaken as ε shrinks. A flat curve
   is what you would also see if the DP-SGD models had simply collapsed to a
   near-constant predictor, which would make the result a statement about
   underfitting rather than about privacy.
3. **Confounded, unbalanced groups.** `scripts/run_attack_eval.py` used the
   corpus's own splits: members were the 86 training participants, non-members
   the 34 dev participants. Those two groups differ by more than membership —
   they are DAIC-WOZ's own train/dev partition, with their own class and
   demographic composition — so any measured separation partly reflects the
   split, not the model's memorisation. The 86 : 34 imbalance also makes the
   attacker's raw `accuracy` (reported as 0.70) a statement about the prior.

## Decision

**1. Members and non-members are drawn from one pooled corpus, in equal halves,
redrawn per seed.** The pool is the union of the three protocol splits
(`train + selection + report`, 141 participants). A stratified coin-flip on the
depression label assigns half to be trained on and half to be held out. After
that draw, membership is the *only* systematic difference between the groups,
and the attacker's accuracy is measured against a 50/50 prior.

The previous behaviour is kept as `--membership-split official` so the earlier
run stays reproducible; `balanced` is what the thesis reports.

**2. The evaluation is repeated over seeds.** `--seeds` reruns the whole
pipeline — member draw, model init, DP noise, and the attacker's own calibration
split — and the report carries `mean ± std` plus a normal-approximation 95%
interval on the seed-to-seed standard error, alongside every per-seed row. The
committed protocol is 10 seeds, matching ADR-0020.

**3. Membership inference can be run without the re-identification sweep.**
`--mia-only` skips the three re-identification attackers and the adaptive-
allocation section. Re-identification risk is already measured at 10 seeds by
`scripts/run_reid_risk.py` (ADR-0017), so re-running it inside the attack
harness spends GPU time on a number that is not the one under test.

## Consequences

- The non-private reference and every ε point are now measured on models trained
  on ~70 participants rather than 86. The absolute AUCs are therefore not
  comparable to the 2026-09-04 single-seed run; only the new run's own
  ε-to-reference contrast is.
- The interval is over seeds, not over participants. It describes how stable the
  estimate is across repetitions of the whole procedure, which is the question
  the single-seed run could not answer. It is not a per-participant confidence
  bound and is not presented as one.
- If the flat ε curve survives the repetition, the honest reading is that this
  model does not memorise enough for the attack to have a signal to remove, and
  the claim "DP controls membership inference" is **not demonstrated** on
  DAIC-WOZ — not that DP failed. The pre-registered read is the contrast between
  the non-private reference and the smallest ε; a flat curve with overlapping
  intervals is a null, and gets reported as one.
- `data/users`-style leakage is not a concern here: the pool is assembled from
  datasets already loaded under the ADR-0015 protocol, and the report split's
  role as the never-selected-on utility split is untouched — the membership
  attack neither trains nor selects on the utility numbers.

## Alternatives considered

- **Keep the official splits and just add seeds.** Rejected: repeating a
  confounded comparison ten times produces a tight interval around a biased
  estimate.
- **Shadow-model attack (Shokri et al.).** The stronger attacker, and it would
  give a better upper bound on leakage. Rejected for now on cost: it needs tens
  of shadow models per ε per seed, and with 141 participants the shadow models
  would each be fit on a few dozen samples. Recorded as a limitation for
  Chapter 5.
- **Report balanced accuracy on the official splits instead of rebalancing.**
  Fixes the prior but not the split confound, which is the larger of the two.
