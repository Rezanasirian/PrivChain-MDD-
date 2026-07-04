"""Live Hyperledger Fabric ledger client via a REST gateway (Phase 5, H3).

Implements :class:`~privchain.chain_client.ledger.LedgerClient` by translating
each call into an invoke/query against a REST gateway that fronts the
``privchain-cc`` chaincode. It uses only the standard library (``urllib``) so it
adds no dependency, but — like the Flower and Opacus backends in earlier phases
— it is **not exercised in the offline environment**: it needs a running Fabric
network + gateway. The in-memory :class:`~privchain.chain_client.ledger.MockLedger`
is the offline stand-in. See ADR-0006.

Assumed gateway contract (JSON over HTTP):

* ``POST {gateway_url}/invoke`` and ``POST {gateway_url}/query`` with body
  ``{"channel", "chaincode", "function", "args": [str, ...]}``;
* success → ``200`` with body ``{"payload": <chaincode JSON string>}``;
* failure → non-2xx, or a body containing ``{"error": <message>}``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from privchain.chain_client.ledger import (
    BudgetRecord,
    Capability,
    ClientRecord,
    LedgerError,
    ReputationRecord,
    SubgraphRecord,
)
from privchain.config import LedgerConfig


class FabricRestLedger:
    """A :class:`LedgerClient` backed by a Fabric REST gateway.

    Args:
        config: Validated ledger configuration (gateway URL, channel, chaincode,
            request timeout).
    """

    def __init__(self, config: LedgerConfig) -> None:
        """Store the gateway endpoint and channel/chaincode routing."""
        self._url = config.gateway_url.rstrip("/")
        self._channel = config.channel
        self._chaincode = config.chaincode
        self._timeout = config.timeout_seconds

    # ── writes ────────────────────────────────────────────────────────────────

    def register_client(self, client_id: str, capability: Capability) -> None:
        """Invoke ``RegisterClient``."""
        self._invoke("RegisterClient", [client_id, *(str(flag) for flag in capability)])

    def log_privacy_budget(
        self, client_id: str, modality: str, round_num: int, epsilon_spent: float
    ) -> None:
        """Invoke ``LogPrivacyBudget``."""
        self._invoke(
            "LogPrivacyBudget", [client_id, modality, str(round_num), repr(epsilon_spent)]
        )

    def update_reputation(
        self, client_id: str, modality: str, score: float, round_num: int
    ) -> None:
        """Invoke ``UpdateReputation``."""
        self._invoke("UpdateReputation", [client_id, modality, repr(score), str(round_num)])

    def publish_subgraph(self, round_num: int, client_ids: list[str]) -> None:
        """Invoke ``PublishSubgraph``."""
        self._invoke("PublishSubgraph", [str(round_num), *client_ids])

    # ── reads ─────────────────────────────────────────────────────────────────

    def get_client(self, client_id: str) -> ClientRecord | None:
        """Query ``GetClient``."""
        payload = self._query_optional("GetClient", [client_id])
        if payload is None:
            return None
        capability = payload["capability"]
        return ClientRecord(payload["clientId"], (capability[0], capability[1], capability[2]))

    def get_reputation(self, client_id: str, modality: str) -> ReputationRecord | None:
        """Query ``GetReputation``."""
        payload = self._query_optional("GetReputation", [client_id, modality])
        if payload is None:
            return None
        return ReputationRecord(
            payload["clientId"], payload["modality"], payload["score"], payload["round"]
        )

    def get_subgraph(self, round_num: int) -> SubgraphRecord | None:
        """Query ``GetSubgraph``."""
        payload = self._query_optional("GetSubgraph", [str(round_num)])
        if payload is None:
            return None
        return SubgraphRecord(payload["round"], tuple(payload["clientIds"]))

    def budget_history(self, client_id: str, modality: str) -> list[BudgetRecord]:
        """Query ``GetBudgetHistory``."""
        payload = self._query("GetBudgetHistory", [client_id, modality])
        records = json.loads(payload) if payload else []
        return [
            BudgetRecord(r["clientId"], r["modality"], r["round"], r["epsilonSpent"])
            for r in records
        ]

    # ── transport ──────────────────────────────────────────────────────────────

    def _invoke(self, function: str, args: list[str]) -> str:
        """POST an invoke transaction, returning the raw chaincode payload."""
        return self._post("invoke", function, args)

    def _query(self, function: str, args: list[str]) -> str:
        """POST a query, returning the raw chaincode payload."""
        return self._post("query", function, args)

    def _query_optional(self, function: str, args: list[str]) -> dict[str, Any] | None:
        """Query a single record, mapping a not-found error to ``None``."""
        try:
            payload = self._query(function, args)
        except LedgerError as exc:
            if "not" in str(exc).lower():  # "not registered" / "no ... for" / "not found"
                return None
            raise
        parsed: dict[str, Any] = json.loads(payload)
        return parsed

    def _post(self, endpoint: str, function: str, args: list[str]) -> str:
        """Send one request to the gateway and return the payload string."""
        body = json.dumps(
            {
                "channel": self._channel,
                "chaincode": self._chaincode,
                "function": function,
                "args": args,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self._url}/{endpoint}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LedgerError(f"{function} failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise LedgerError(f"{function} transport error: {exc.reason}") from exc

        parsed = json.loads(raw) if raw else {}
        if isinstance(parsed, dict) and parsed.get("error"):
            raise LedgerError(f"{function} failed: {parsed['error']}")
        payload = parsed.get("payload", "") if isinstance(parsed, dict) else ""
        return str(payload)
