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
    consumed_epsilon: dict[str, dict[str, dict[str, float]]] | None = None,
    subgraph_client_ids: list[str] | None = None,
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
        consumed_epsilon: Optional ``{client_id: {incremental/cumulative:
            {group: epsilon}}}`` from executed client accountants.
        subgraph_client_ids: Optional IDs whose updates were actually aggregated.
            Defaults to every participant. This can differ from ``participants``
            when a Byzantine filter rejects an update after its DP mechanism ran.
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

    if subgraph_client_ids is None:
        subgraph_client_ids = [client_id for client_id, _ in participants]
    ledger.publish_subgraph(round_num, subgraph_client_ids)

    for client_id, capability in participants:
        client_spend = consumed_epsilon.get(client_id) if consumed_epsilon is not None else None
        if client_spend is not None:
            incremental = client_spend["incremental"]
            cumulative = client_spend["cumulative"]
            for group, value in incremental.items():
                ledger.log_privacy_budget(client_id, group, round_num, value, cumulative[group])
        for index, modality in enumerate(CAPABILITY_MODALITIES):
            if capability[index] != 1:
                continue
            if reputation is not None:
                score = reputation.get(client_id, {}).get(modality)
                if score is not None:
                    ledger.update_reputation(client_id, modality, float(score), round_num)
