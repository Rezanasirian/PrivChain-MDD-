"""Record a federated round to the audit ledger (Phase 5, objective H3).

A small backend-agnostic bridge that writes the per-round artifacts of the
capability-aware protocol — the published subgraph, per-modality consumed
privacy budget (H1), and per-modality reputation (H2) — through any
:class:`~privchain.chain_client.ledger.LedgerClient`. Kept free of any
``federated`` import (it takes plain ``(client_id, capability)`` data) so the
ledger layer never depends on the training layer.
"""

from __future__ import annotations

from privchain.chain_client.ledger import Capability, LedgerClient
from privchain.config import CAPABILITY_MODALITIES


def record_round(
    ledger: LedgerClient,
    *,
    round_num: int,
    participants: list[tuple[str, Capability]],
    reputation: dict[str, dict[str, float]] | None = None,
    consumed_epsilon: dict[str, float] | None = None,
    registered: set[str],
) -> None:
    """Write one round's registration, subgraph, budget, and reputation.

    Args:
        ledger: The target ledger client.
        round_num: The federated round number.
        participants: ``(client_id, capability)`` for the round's selected
            clients, with ``capability`` the ``[audio, video, text]`` flags.
        reputation: Optional ``{client_id: {modality: score}}`` to log per
            modality the client possesses.
        consumed_epsilon: Optional ``{modality: epsilon}`` consumed this round
            (from the DP budget allocator), logged append-only per client.
        registered: Set of already-registered client IDs, updated in place so a
            client is registered exactly once across rounds.
    """
    for client_id, capability in participants:
        if client_id in registered:
            continue
        # The `registered` set only knows what *this process* has done, but a real
        # ledger outlives the process: a client registered by an earlier run is
        # still registered, and the chaincode rightly refuses to register it
        # twice. Checking the ledger makes the audit trail re-runnable instead of
        # failing on the second run against a live network (ADR-0022).
        if ledger.get_client(client_id) is None:
            ledger.register_client(client_id, capability)
        registered.add(client_id)

    ledger.publish_subgraph(round_num, [client_id for client_id, _ in participants])

    for client_id, capability in participants:
        for index, modality in enumerate(CAPABILITY_MODALITIES):
            if capability[index] != 1:
                continue
            if consumed_epsilon is not None and modality in consumed_epsilon:
                ledger.log_privacy_budget(
                    client_id, modality, round_num, consumed_epsilon[modality]
                )
            if reputation is not None:
                score = reputation.get(client_id, {}).get(modality)
                if score is not None:
                    ledger.update_reputation(client_id, modality, float(score), round_num)
