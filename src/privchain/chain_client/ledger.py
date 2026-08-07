"""Auditable ledger bridge (Phase 5, objective H3).

Defines the :class:`LedgerClient` protocol mirroring the four ``privchain-cc``
chaincode functions (plus reads), and an in-memory :class:`MockLedger` that
enforces the **same invariants** as the Go chaincode so the whole federated
pipeline can be exercised offline:

* client registration is one-shot (no duplicates) and **binds the client to the
  identity that registered it**;
* consumed privacy budget is **append-only** — a second write for the same
  ``(client, modality, round)`` is refused (CLAUDE.md §7) — and only the owning
  client or the coordinator may write it;
* a round's aggregation **subgraph is immutable** once published, and only the
  coordinator may publish it;
* reputation is updatable by design, but **coordinator-only**: a client that
  could write its own reputation could set its own aggregation weight.

The live Fabric path lives in :mod:`privchain.chain_client.fabric_gateway`; both
implement this protocol so callers are backend-agnostic. See ADR-0006.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from privchain.config import CAPABILITY_MODALITIES

Capability = tuple[int, int, int]


class LedgerError(RuntimeError):
    """Raised when a ledger operation violates an invariant or fails."""


@dataclass(frozen=True)
class LedgerIdentity:
    """A submitting identity, mirroring Fabric's ``SerializedIdentity``.

    Args:
        msp_id: The submitter's membership service provider (its organization).
        name: A stable per-identity label standing in for the certificate
            fingerprint the chaincode hashes.
    """

    msp_id: str
    name: str = ""


DEFAULT_COORDINATOR_MSP = "CoordinatorMSP"


@dataclass(frozen=True)
class ClientRecord:
    """A registered federated client, its capability vector, and its owner."""

    client_id: str
    capability: Capability
    owner: LedgerIdentity | None = None


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
    without a running Fabric network — including its **access control**, so a
    permission bug cannot pass offline and only surface against real Fabric.

    The caller identity defaults to the coordinator, which is what the federated
    server is; :meth:`acting_as` switches it to exercise the client-side rules.

    Args:
        coordinator_msp: MSP allowed to update reputation and publish subgraphs.
        caller: Identity submitting operations (defaults to the coordinator).
    """

    def __init__(
        self,
        coordinator_msp: str = DEFAULT_COORDINATOR_MSP,
        caller: LedgerIdentity | None = None,
    ) -> None:
        """Create an empty ledger."""
        self._clients: dict[str, ClientRecord] = {}
        self._budgets: dict[tuple[str, str, int], BudgetRecord] = {}
        self._reputation: dict[tuple[str, str], ReputationRecord] = {}
        self._subgraphs: dict[int, SubgraphRecord] = {}
        self.coordinator_msp = coordinator_msp
        self.caller = caller or LedgerIdentity(coordinator_msp, "coordinator")

    @contextmanager
    def acting_as(self, identity: LedgerIdentity) -> Iterator[MockLedger]:
        """Temporarily submit operations as ``identity``.

        Args:
            identity: The identity to act as inside the block.

        Yields:
            This ledger, with ``caller`` swapped for the duration.
        """
        previous = self.caller
        self.caller = identity
        try:
            yield self
        finally:
            self.caller = previous

    def _require_coordinator(self, operation: str) -> None:
        """Raise unless the caller belongs to the coordinator MSP."""
        if self.caller.msp_id != self.coordinator_msp:
            raise LedgerError(
                f"only the coordinator MSP {self.coordinator_msp!r} may {operation} "
                f"(caller is {self.caller.msp_id!r})"
            )

    def _require_owner_or_coordinator(self, client_id: str) -> None:
        """Raise unless the caller owns ``client_id`` or is the coordinator."""
        if self.caller.msp_id == self.coordinator_msp:
            return
        record = self._clients.get(client_id)
        if record is None:
            raise LedgerError(f"client {client_id!r} is not registered")
        if record.owner != self.caller:
            raise LedgerError(f"identity {self.caller.msp_id!r} does not own client {client_id!r}")

    def register_client(self, client_id: str, capability: Capability) -> None:
        """Register a client; raise if empty ID, bad capability, or duplicate."""
        if not client_id:
            raise LedgerError("client_id must not be empty")
        _validate_capability(capability)
        if client_id in self._clients:
            raise LedgerError(f"client {client_id!r} is already registered")
        self._clients[client_id] = ClientRecord(client_id, capability, self.caller)

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
        self._require_owner_or_coordinator(client_id)
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
        self._require_coordinator("update reputation")
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
        self._require_coordinator("publish a subgraph")
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
