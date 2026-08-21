# ADR-0024 — Data-free anchor distillation

- **Status:** Accepted
- **Date:** 2026-08-21
- **Phase:** 4
- **Related objectives:** H2 (capability-aware federated learning), H1 (privacy)

## Context

The original federated-distillation path evaluates the global teacher on the
same capability-masked private batch seen by the client. Consequently the
teacher has no access to the client's missing modalities and the loss cannot
transfer cross-modal knowledge. It acts as a proximal constraint instead.

Putting a knowledge-distillation term on private examples inside DP-SGD would
also change the per-sample mechanism whose privacy cost is accounted for.

## Decision

The primary distillation arm uses data-free anchors synthesized only from the
current global model. Anchor input tensors are optimized for confident and
diverse teacher predictions without reading DAIC-WOZ records. The full-modality
global teacher evaluates each anchor; a client student evaluates the same
anchor with its capability mask and matches the teacher logits.

Each round has an explicit order:

1. run the fixed number of private DP-SGD steps using only the depression
   objective;
2. run a fixed number of KD steps over data-free anchors.

The KD step is post-processing of a DP model and does not add an accountant
event. The experiment exposes three distinct arms:

- `distill_anchor`: optimized data-free anchors (primary mechanism);
- `distill_random`: unoptimized random anchors (control);
- `distill_proximal`: the former masked-private-batch mechanism (legacy
  comparison only).

## Privacy boundary

This decision is valid only while anchor synthesis has no direct access to
private records. If corpus examples, corpus statistics, cached embeddings, or
any other DAIC-WOZ-derived value is used to build anchors, the operation is no
longer post-processing and must be included in the privacy mechanism and its
accounting before results may be reported.

## Consequences

- The teacher can supply information from modalities absent at the client.
- The private per-sample objective and `dp_train_steps` contract remain
  unchanged.
- Acceptance tests inspect the mechanism (full teacher mask, client student
  mask, non-zero KD gradient, and unchanged accountant state), not a stochastic
  improvement in a metric.
