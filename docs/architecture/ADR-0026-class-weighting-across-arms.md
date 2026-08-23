# ADR-0026 — One loss for every arm: class weighting reaches federated clients

- **Status:** Accepted
- **Date:** 2026-08-23
- **Phase:** 7 (final evaluation), with effects on Phases 2 and 4
- **Supersedes nothing.** Corrects a defect present in every Chapter-4 run to date.

## Context

`configs/baseline.yaml` sets `train.class_weighting: true`. DAIC-WOZ is roughly
30% positive, so the weight is around 2.3.

The centralized arm honoured that flag. `scripts/run_final_evaluation.py` measured
`positive_class_weight` on each fold's training split and handed it to
`CentralizedTrainer`, which built `BCEWithLogitsLoss(pos_weight=...)`.

No other arm did. `FederatedClient.__init__` constructed
`DepressionObjective(phq8_max, phq_loss_weight)` with no third argument, so
`pos_weight` defaulted to `None` and every client trained under plain,
unweighted BCE. `build_federated_clients` had no parameter through which a weight
could have been passed. `_eval_centralized_dp` never computed one either.

The headline Chapter-4 table therefore compared **one class-weighted model
against six unweighted ones**. The 10-fold CV run of 2026-08-23 shows what that
produced on real DAIC-WOZ:

| arm | ROC-AUC | loss |
|---|---|---|
| centralized | 0.676 | weighted |
| fedavg | 0.439 | unweighted |
| personalized | 0.402 | unweighted |
| proposed | 0.440 | unweighted |
| proposed − reputation | 0.448 | unweighted |
| DP adaptive / uniform | 0.539 | unweighted |

Every federated arm sat at or below chance while the centralized arm cleared it
comfortably. The per-round logs rule out the obvious alternatives: the federated
models do learn (selection ROC-AUC climbs from 0.42 to 0.70 over 120 rounds),
early stopping fires on schedule, and the best checkpoint is restored correctly
by `run_final_evaluation.py`. What differed between the arms was the objective.

This is the same class of defect ADR-0013 found in the DP arm and ADR-0021 found
in the round budget: an experimental asymmetry that charges a method for a
difference in training setup rather than in method. The comment inside
`_BestRoundTracker` warns against precisely this, for the schedule. Nothing was
guarding the loss.

That `positive_class_weight` was already documented as staying correct "across
splits, folds, and **federated client shards**" indicates the wiring was an
oversight rather than a deliberate design choice.

## Decision

Every arm trains under the same objective. `train.class_weighting` now reaches
the federated clients and the DP arms.

**Each client measures its weight on its own shard.** `build_federated_clients`
gains `class_weighting: bool = False`; when set, it counts labels over that
client's own partition and passes the result to `FederatedClient(pos_weight=...)`.
A client in a real federation can observe its local class balance and nothing
more, so a per-shard weight is both the realistic and the leak-free choice. A
pooled weight computed across clients would be a small but genuine violation of
the federated boundary.

**A single-class shard stays unweighted.** `positive_class_weight` returns `None`
when either class is absent, where the ratio is undefined. With ten clients over
141 participants this is reachable, so it is handled rather than guarded against.

**The default is off.** Existing callers — `run_federated.py`,
`run_capability_federated.py`, `run_federated_with_ledger.py`,
`run_flower_parity.py`, the Flower adapter, and the integration tests — keep
their current behaviour unless they opt in. `run_final_evaluation.py` and
`run_federated_comparison.py` pass `class_weighting=train.class_weighting`, since
both compare federated arms against a weighted centralized baseline.

## Consequences

Every previously published Chapter-4 federated and DP number is not comparable
to its centralized counterpart and is superseded. Per the pre-registration's
policy, the superseded artifacts are marked, not deleted.

The pre-registration is re-locked as `PRE-REGISTRATION-2026-08-23.md`. The
original locked configuration and code at commit `f78dd96`, and that commit
contains this defect, so a campaign run against it would produce a confounded
headline comparison. Clause 4 of the original protocol permits reruns for a
documented infrastructure error; an arm-dependent loss function is such an error.
The correction is disclosed in the new document rather than applied silently.

This does **not** predict that federation will now beat chance. It removes a
confound so the comparison measures the method. A negative result after the fix
is a real finding and will be reported as one.

`test_class_weighting_reaches_clients_from_their_own_shard` pins both halves of
the invariant: the flag reaches the clients, and the weight is per-shard rather
than shared.

## Alternatives rejected

**Disable class weighting everywhere.** Symmetric and a one-line change, but it
discards a correction that measurably helps on an imbalanced corpus, and it would
change the centralized baseline that Phases 1 through 6 already report.

**Pool the weight across clients.** Simpler and lower variance on small shards,
but it hands each client a statistic computed from data it does not hold, which
weakens the federated claim the thesis is making.
