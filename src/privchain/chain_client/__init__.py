"""Python <-> Hyperledger Fabric bridge (Phase 5, objective H3).

Exposes a backend-agnostic :class:`LedgerClient` (mirroring the ``privchain-cc``
chaincode), an offline :class:`MockLedger`, the live :class:`FabricRestLedger`,
a :func:`build_ledger` factory, and :func:`record_round` to write a federated
round's audit trail. See ADR-0006.
"""

from __future__ import annotations

from privchain.chain_client.ledger import (
    BudgetRecord,
    Capability,
    ClientRecord,
    LedgerClient,
    LedgerError,
    MockLedger,
    ReputationRecord,
    SubgraphRecord,
)
from privchain.chain_client.recording import record_round
from privchain.config import LedgerConfig

__all__ = [
    "BudgetRecord",
    "Capability",
    "ClientRecord",
    "LedgerClient",
    "LedgerError",
    "MockLedger",
    "ReputationRecord",
    "SubgraphRecord",
    "build_ledger",
    "record_round",
]


def build_ledger(config: LedgerConfig) -> LedgerClient:
    """Construct the ledger client selected by ``config.backend``.

    Args:
        config: Validated ledger configuration.

    Returns:
        A :class:`MockLedger` (offline) or :class:`FabricRestLedger` (live).

    Raises:
        ValueError: If the configured backend is unknown.
    """
    if config.backend == "mock":
        return MockLedger()
    if config.backend == "fabric_rest":
        from privchain.chain_client.fabric_gateway import FabricRestLedger

        return FabricRestLedger(config)
    raise ValueError(f"unknown ledger backend: {config.backend!r}")
