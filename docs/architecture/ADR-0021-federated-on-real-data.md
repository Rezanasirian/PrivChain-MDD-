# ADR-0021 — Federated learning on real DAIC-WOZ (H2)

- **Status:** Accepted
- **Date:** 2026-08-13
- **Phase:** 2 and 4
- **Related objectives:** H2 (capability-aware federation), H5 (evaluation)

## Context

H2 is an entire thesis objective, and until now every federated number in this
project came from the **mock corpus**, where the depression label is random noise.
The claim under test is that when clients hold *different subsets of the
modalities*, plain FedAvg is damaged by zero-imputing what a client lacks, and
that capability-aware subgraph aggregation — plus reputation weighting and
federated distillation — repairs it.

The existing harness (`run_capability_federated.py`) also predates ADR-0015. It
selected the best global model on the same loader it reported, ran a single seed,
chose no decision threshold, and quoted no interval. Its numbers were not
comparable to anything else in the project.

## Method

`scripts/run_federated_comparison.py` runs every arm under the shared protocol:
splits from `build_splits`, clients partitioning **train only**, per-round model
selection on **selection**, the untouched dev split read once at a threshold
chosen on selection, three seeds, bootstrap CIs and **paired** differences
(ADR-0020). All arms start each seed from the *same* initial global state, so
they differ in aggregation and nothing else.

Two additions were needed:

- **Label-skewed partitioning.** The population was IID, which gives every client
  the corpus-wide prevalence — the easiest case for averaging and the least like
  real clinical federation. `partition_indices_dirichlet` draws each client's
  class mix from a Dirichlet, and every arm runs under both regimes.
- **Early stopping on the selection split.** Federated arms ran a fixed 20-round
  budget while the centralized baseline stopped at its best epoch, which charged
  federation for a difference in schedule rather than in method — the same trap
  ADR-0013 found in the DP arm. Rounds are now 120 with patience 40.

## Result

86 train / 21 selection / 34 dev, 10 clients, three seeds.

| arm | IID | Dirichlet (α=0.5) | 5 clients (IID) |
|---|---|---|---|
| centralized | 0.740 ± 0.016 | 0.740 ± 0.016 | 0.740 ± 0.016 |
| fedavg | 0.555 ± 0.019 | 0.560 ± 0.026 | 0.557 ± 0.036 |
| capability | **0.582 ± 0.011** | **0.598 ± 0.035** | 0.561 ± 0.009 |
| capability+reputation | 0.580 ± 0.010 | 0.594 ± 0.007 | 0.568 ± 0.015 |
| capability+reputation+distillation | 0.578 ± 0.013 | 0.592 ± 0.004 | 0.561 ± 0.009 |

Paired bootstrap against `fedavg`:

| comparison | IID | Dirichlet | 5 clients |
|---|---|---|---|
| centralized − fedavg | +0.178, p=0.148 | +0.162, p=0.124 | +0.142, p=0.275 |
| capability − fedavg | +0.024, p=0.567 | +0.036, p=0.208 | **−0.043**, p=0.488 |
| +reputation − fedavg | +0.016, p=0.721 | +0.016, p=0.719 | +0.000, p=1.000 |
| +distillation − fedavg | +0.020, p=0.629 | +0.008, p=0.872 | −0.040, p=0.516 |

**Nothing separates.** Not one comparison, in any configuration.

### 1. Federation is expensive — and even that cannot be proven here

Centralized 0.740 against federated ~0.56: a drop of roughly 0.17 ROC-AUC, by far
the largest effect in the table. It is still **not statistically resolvable**
(p = 0.12–0.28). That is the clearest possible illustration of ADR-0020's point:
on 34 dev sessions this corpus cannot resolve an effect of 0.17. Any claim about
the cost of federation must be stated as a point estimate with that caveat
attached, not as a finding.

### 2. Capability-aware aggregation: right direction, no evidence

`capability − fedavg` is positive under both 10-client regimes (+0.024 IID,
+0.036 skewed) and slightly larger when the partition is skewed — which is the
direction H2 predicts, since heterogeneity is what the protocol exists for.

But it **reverses to −0.043 with 5 clients**, and no comparison approaches
significance. A mechanism whose measured effect changes sign with the client
count has not been demonstrated. The honest reading: the direction is weakly
consistent at 10 clients and the evidence is absent.

### 3. Reputation and distillation contribute nothing measurable

Both increments sit at or below +0.016 in every configuration, and the full
protocol is never better than capability-aware aggregation alone. On this corpus
they are cost without return.

### 4. The per-pattern breakdown is where the story actually is

ROC-AUC of the final global model evaluated with only each pattern's modalities
available at inference (IID):

| arm | full | audio+text | audio only | text only |
|---|---|---|---|---|
| centralized | 0.740 | 0.715 | 0.532 | 0.738 |
| fedavg | 0.555 | 0.519 | 0.522 | 0.685 |
| capability | 0.582 | 0.511 | 0.511 | 0.689 |

Two things stand out. **A federated model given only text scores 0.685–0.689 —
far better than the same model given everything (0.555).** And the centralized
model shows the same shape (text-only 0.738 ≈ full 0.740). This is ADR-0016's
finding reappearing from a different direction: text carries the signal, and the
audio and video encoders are contributing noise that the fusion layer has to
overcome. Capability-aware aggregation improves the `full` view (0.555 → 0.582)
without improving the modality-poor views, which is the opposite of the
zero-imputation story it was designed around.

## Flower: executed at last, and it was broken

The thesis names Flower as the orchestration layer, but the adapter had been
written and never run (ADR-0003). `scripts/run_flower_parity.py` puts both
backends on the same splits, partitions, initial parameters and seed, and
compares their final selection-split metrics against a tolerance fixed in advance.

Running it took four attempts, and each failure is worth recording because none
of them would have been found by reading the code:

1. **flwr 1.12 is incompatible with NumPy 2.** It uses `np.float_`, removed in
   NumPy 2.0, while torch 2.12 and scipy require NumPy 2.x. flwr 1.33 works.
   Installing it into the main environment also downgraded `typer`, `pathspec`
   and NumPy, breaking mypy and the transformer stack — so `flwr` is installed
   into a separate directory and put on `PYTHONPATH` only for this check.
2. **Ray workers get no GPU unless asked.** Clients built for CUDA could not be
   deserialized in the worker.
3. **Clients were constructed in the parent process**, so Ray had to pickle their
   CUDA tensors into every worker. Clients are now built *inside* `client_fn`,
   which is what that hook is for.
4. **`History` does not carry the final global parameters**, so the check was
   evaluating the untrained initial model and calling it Flower's result.

Defects 2–4 all failed the same way: Flower logs `received 0 results and N
failures`, then **continues with the untrained initial parameters**. A run that
trains nothing looks like a run that converged. That is precisely why an adapter
that is never executed cannot be claimed to work.

With those fixed, 10 rounds, 10 clients, seed 42, all rounds aggregating
`10 results and 0 failures`:

| metric | in-house | Flower | abs delta |
|---|---|---|---|
| ROC-AUC | 0.5222 | 0.5000 | 0.0222 |
| F1 | 0.4800 | 0.4800 | 0.0000 |
| accuracy | 0.3810 | 0.3810 | 0.0000 |

**AGREE** at the 0.05 tolerance. F1 and accuracy are identical; the small ROC-AUC
difference is expected, since Flower samples clients through its own strategy and
RNG rather than the in-house round sampler. The in-house simulator's numbers can
be reported as the protocol's behaviour, not as one implementation's quirk.

## What this means for H2

The protocol is implemented, runs on real data, and is measured honestly. What it
is not is *demonstrated*. On a corpus of 86 training sessions split across 10
clients — roughly 9 sessions each — the differences between aggregation
strategies are far below what 34 dev sessions can resolve, and the one modality
that carries signal is present in 80% of the client patterns anyway, so there is
little for capability-awareness to rescue.

Chapter 4 should report this as: **capability-aware aggregation shows a small
consistent positive direction at 10 clients that does not survive a change in
client count and is not statistically distinguishable from zero; reputation
weighting and federated distillation show no effect at all.** Claiming more than
that is not supportable.

The experiment is not wasted — it establishes the protocol, the harness, and the
bound. But H2's utility claim needs either a substantially larger corpus or a
setting where the modality split actually removes the informative modality from
most clients. The current population mix (80% of patterns include text) is close
to the best case for FedAvg and the worst case for showing that capability-aware
aggregation is necessary.

## Consequences

- `configs/federated.yaml` gains `partition` (mode + `dirichlet_alpha`) and
  `early_stopping_patience`; `num_rounds` raised 20 → 120.
- `run_capability_federated.py` remains for the mock path but must not be cited:
  it selects and reports on the same split.
- The client population mix is worth revisiting as an experimental variable — a
  mix where most clients *lack* text would be a far sharper test of H2 than the
  current one, and is the obvious follow-up.
