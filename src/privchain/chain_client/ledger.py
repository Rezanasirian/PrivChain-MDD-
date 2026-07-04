"""Auditable ledger bridge (Phase 5, objective H3).

Defines the :class:`LedgerClient` protocol mirroring the four ``privchain-cc``
chaincode functions (plus reads), and an in-memory :class:`MockLedger` that
enforces the **same invariants** as the Go chaincode so the whole federated
pipeline can be exercised offline:

* client registration is one-shot (no duplicates);
* consumed privacy budget is **append-only** — a second write for the same
  ``(client, modality, round)`` is refused (CLAUDE.md §7);
* a round's aggregation **subgraph is immutable** once published;
* reputation is updatable by design.

The live Fabric path lives in :mod:`privchain.chain_client.fabric_gateway`; both
implement this protocol so callers are backend-agnostic. See ADR-0006.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from privchain.config import CAPABILITY_MODALITIES

Capability = tuple[int, int, int]


class LedgerError(RuntimeError):
    """Raised when a ledger operation violates an invariant or fails."""


@dataclass(frozen=True)
class ClientRecord:
    """A registered federated client and its declared capability vector."""

    client_id: str
    capability: Capability


@dataclass(frozen=True)
class BudgetRecord:
    """One append-only per-modality privacy-budget entry for a round."""

    client_id: str
    modality: str
    round: int
    epsilon_spent: float


@dataclass(frozen=True)
class ReputationRecord:
    """A client's latest per-modality reputation score."""

    client_id: str
    modality: str
    score: float
    round: int


@dataclass(frozen=True)
class SubgraphRecord:
    """The set of clients aggregated together in a round (immutable)."""

    round: int
    client_ids: tuple[str, ...]


@runtime_checkable
class LedgerClient(Protocol):
    """Backend-agnostic interface to the audit ledger (chaincode-mirroring)."""

    def register_client(self, client_id: str, capability: Capability) -> None:
        """Register ``client_id`` with its ``[audio, video, text]`` capability."""
        ...

    def log_privacy_budget(
        self, client_id: str, modality: str, round_num: int, epsilon_spent: float
    ) -> None:
        """Append a consumed-ε entry (append-only; never overwritten)."""
        ...

    def update_reputation(
        self, client_id: str, modality: str, score: float, round_num: int
    ) -> None:
        """Set a client's latest per-modality reputation (updatable)."""
        ...

    def publish_subgraph(self, round_num: int, client_ids: list[str]) -> None:
        """Publish a round's aggregation subgraph (immutable once set)."""
        ...

    def get_client(self, client_id: str) -> ClientRecord | None:
        """Return a client's record, or ``None`` if unregistered."""
        ...

    def get_reputation(self, client_id: str, modality: str) -> ReputationRecord | None:
        """Return a client's latest reputation for ``modality``, or ``None``."""
        ...

    def get_subgraph(self, round_num: int) -> SubgraphRecord | None:
        """Return a round's published subgraph, or ``None``."""
        ...

    def budget_history(self, client_id: str, modality: str) -> list[BudgetRecord]:
        """Return every budget entry for ``client_id``+``modality`` (round order)."""
        ...


def _validate_capability(capability: Capability) -> None:
    """Raise :class:`LedgerError` if a capability vector is malformed."""
    if len(capability) != len(CAPABILITY_MODALITIES):
        raise LedgerError(f"capability must have length {len(CAPABILITY_MODALITIES)}")
    if any(flag not in (0, 1) for flag in capability):
        raise LedgerError("capability entries must be 0 or 1")
    if sum(capability) == 0:
        raise LedgerError("a client must declare at least one modality")


def _validate_modality(modality: str) -> None:
    """Raise :class:`LedgerError` for an unknown modality name."""
    if modality not in CAPABILITY_MODALITIES:
        raise LedgerError(f"unknown modality: {modality!r}")


class MockLedger:
    """In-memory ledger mirroring the ``privchain-cc`` chaincode invariants.

    Used for offline development and testing (Risk #3 in the implementation
    plan): a drop-in :class:`LedgerClient` that behaves like the real chaincode
    without a running Fabric network.
    """

    def __init__(self) -> None:
        """Create an empty ledger."""
        self._clients: dict[str, ClientRecord] = {}
        self._budgets: dict[tuple[str, str, int], BudgetRecord] = {}
        self._reputation: dict[tuple[str, str], ReputationRecord] = {}
        self._subgraphs: dict[int, SubgraphRecord] = {}

    def register_client(self, client_id: str, capability: Capability) -> None:
        """Register a client; raise if empty ID, bad capability, or duplicate."""
        if not client_id:
            raise LedgerError("client_id must not be empty")
        _validate_capability(capability)
        if client_id in self._clients:
            raise LedgerError(f"client {client_id!r} is already registered")
        self._clients[client_id] = ClientRecord(client_id, capability)

    def log_privacy_budget(
        self, client_id: str, modality: str, round_num: int, epsilon_spent: float
    ) -> None:
        """Append a consumed-ε entry; raise on overwrite (append-only)."""
        _validate_modality(modality)
        if round_num < 0:
            raise LedgerError("round must be non-negative")
        if not (epsilon_spent >= 0.0) or epsilon_spent == float("inf"):
            raise LedgerError("epsilon_spent must be finite and non-negative")
        if client_id not in self._clients:
            raise LedgerError(f"client {client_id!r} is not registered")
        key = (client_id, modality, round_num)
        if key in self._budgets:
            raise LedgerError(
                f"privacy budget already logged for {client_id!r}/{modality}/round {round_num}; "
                "consumed epsilon is append-only and must not be overwritten"
            )
        self._budgets[key] = BudgetRecord(client_id, modality, round_num, epsilon_spent)

    def update_reputation(
        self, client_id: str, modality: str, score: float, round_num: int
    ) -> None:
        """Set a client's latest reputation; raise if score is out of [0, 1]."""
        _validate_modality(modality)
        if round_num < 0:
            raise LedgerError("round must be non-negative")
        if not 0.0 <= score <= 1.0:
            raise LedgerError("score must be in [0, 1]")
        if client_id not in self._clients:
            raise LedgerError(f"client {client_id!r} is not registered")
        self._reputation[(client_id, modality)] = ReputationRecord(
            client_id, modality, score, round_num
        )

    def publish_subgraph(self, round_num: int, client_ids: list[str]) -> None:
        """Publish a round's subgraph; raise on empty list or re-publish."""
        if round_num < 0:
            raise LedgerError("round must be non-negative")
        if not client_ids:
            raise LedgerError("a subgraph must contain at least one client")
        if any(not cid for cid in client_ids):
            raise LedgerError("subgraph client IDs must not be empty")
        if round_num in self._subgraphs:
            raise LedgerError(f"subgraph for round {round_num} is already published (immutable)")
        self._subgraphs[round_num] = SubgraphRecord(round_num, tuple(client_ids))

    def get_client(self, client_id: str) -> ClientRecord | None:
        """Return a client's record, or ``None``."""
        return self._clients.get(client_id)

    def get_reputation(self, client_id: str, modality: str) -> ReputationRecord | None:
        """Return a client's latest reputation for ``modality``, or ``None``."""
        return self._reputation.get((client_id, modality))

    def get_subgraph(self, round_num: int) -> SubgraphRecord | None:
        """Return a round's subgraph, or ``None``."""
        return self._subgraphs.get(round_num)

    def budget_history(self, client_id: str, modality: str) -> list[BudgetRecord]:
        """Return budget entries for a client+modality, ordered by round."""
        entries = [
            record
            for (cid, mod, _), record in self._budgets.items()
            if cid == client_id and mod == modality
        ]
        return sorted(entries, key=lambda record: record.round)
