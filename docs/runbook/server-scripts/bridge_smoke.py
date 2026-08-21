"""Exercise every FabricRestLedger method against the real gateway."""

import sys
import uuid

from privchain.chain_client import LedgerError, build_ledger
from privchain.config import LedgerConfig

cfg = LedgerConfig(
    backend="fabric_rest",
    channel="privchain-channel",
    chaincode="privchain-cc",
    gateway_url="http://127.0.0.1:8801",
    timeout_seconds=30.0,
)
ledger = build_ledger(cfg)
cid = f"bridge-{uuid.uuid4().hex[:8]}"
print("backend:", type(ledger).__name__)

ledger.register_client(cid, (1, 0, 1))
print("register_client   OK")
rec = ledger.get_client(cid)
print("get_client        ", rec)
ledger.log_privacy_budget(cid, "audio", 1, 0.5, 0.5)
print("log_privacy_budget OK")
ledger.log_privacy_budget(cid, "text", 1, 0.25, 0.25)
print("log_privacy_budget OK (text)")
print("budget_history    ", ledger.budget_history(cid, "audio"))
ledger.update_reputation(cid, "audio", 0.8, 1)
print("update_reputation OK")
print("get_reputation    ", ledger.get_reputation(cid, "audio"))
ledger.publish_subgraph(7, [cid])
print("publish_subgraph  OK")
print("get_subgraph      ", ledger.get_subgraph(7))

# The append-only invariant must surface as a LedgerError, not a silent success.
try:
    ledger.log_privacy_budget(cid, "audio", 1, 0.9)
    print("APPEND_ONLY_NOT_ENFORCED")
    sys.exit(1)
except LedgerError as exc:
    print("append-only rejected via bridge:", str(exc)[:90])
print("BRIDGE_OK")
