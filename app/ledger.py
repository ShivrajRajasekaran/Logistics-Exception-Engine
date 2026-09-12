"""
Transit custody records, and the integrity check over them.

WHAT THIS IS, AND WHAT IT IS NOT
These are the custody facts the reasoning layer reconciles detections against:
what condition a parcel was in when it left origin, and who was carrying it.
The service never writes to them. An inspection at a delivery hub adjudicates a
parcel's condition; it does not get to revise what origin recorded.

This module was previously a bare dict in main.py named
`IMMUTABLE_TRANSIT_LEDGER`. That name overclaimed. A Python dict is mutable,
`.get()` returned the live nested object by reference, and any code in the
process could have rewritten a custody record with a plain assignment.

What is provided here is **tamper-evidence, not tamper-proofing**:

  * `lookup()` returns a deep copy, so a caller cannot reach back through the
    returned object and mutate the source.
  * `DIGEST` is a SHA-256 over the canonical JSON form, computed once at import.
  * `verify()` recomputes that digest on demand, so an in-process modification
    becomes *detectable*. `/health` reports it and every adjudication records it.

That is meaningfully weaker than a smart contract, and the difference is worth
stating plainly rather than implying otherwise. A contract gives cryptographic
integrity, an append-only history, and verification by a party that does not
have to trust the operator. This gives the first of those three, against
accidental or in-process modification only. Anyone who can edit this file can
also recompute the digest. Real custody integrity belongs in an append-only
store outside the serving process; `app/store.py` is the first half of that.
"""

import copy
import hashlib
import json
from typing import Dict, Optional

TRANSIT_LEDGER: Dict[str, dict] = {
    "PKG-8821": {
        "package_id": "PKG-8821",
        "carrier": "Apex Logistics",
        "origin_hub": "HUB-01 Chennai Sorting Center",
        "origin_label_status": "INTACT",
        "origin_seal_status": "INTACT",
        "dispatched_at": "2026-09-04T06:12:00Z",
        "sku_manifest": "SKU-9901",
        "declared_value_inr": 48500,
        "transit_history": [
            {"hub": "HUB-01 Chennai", "event": "DISPATCH_SCAN", "condition": "INTACT"},
            {"hub": "HUB-04 Bengaluru", "event": "TRANSFER_SCAN", "condition": "NOT_INSPECTED"},
            {"hub": "HUB-09 Pune", "event": "ARRIVAL_SCAN", "condition": "PENDING_INSPECTION"},
        ],
    },
    "PKG-9940": {
        "package_id": "PKG-9940",
        "carrier": "Meridian Freight",
        "origin_hub": "HUB-02 Coimbatore Consolidation",
        "origin_label_status": "ALREADY_DAMAGED",
        "origin_seal_status": "INTACT",
        "dispatched_at": "2026-09-05T21:47:00Z",
        "sku_manifest": "SKU-2274",
        "declared_value_inr": 12300,
        "transit_history": [
            {"hub": "HUB-02 Coimbatore", "event": "DISPATCH_SCAN", "condition": "LABEL_TORN_AT_ORIGIN"},
            {"hub": "HUB-09 Pune", "event": "ARRIVAL_SCAN", "condition": "PENDING_INSPECTION"},
        ],
    },
}


def _digest(records: Dict[str, dict]) -> str:
    """SHA-256 over the canonical JSON form.

    sort_keys makes the encoding independent of dict ordering, so the digest
    changes only when a value actually changes.
    """
    blob = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


DIGEST: str = _digest(TRANSIT_LEDGER)


def verify() -> bool:
    """True while the records still hash to the digest taken at import."""
    return _digest(TRANSIT_LEDGER) == DIGEST


def lookup(package_id: str) -> Optional[dict]:
    """Return a deep copy of one record, or None.

    The copy is the point. Handing out the live nested dict let a caller mutate
    the source through the object it was given.
    """
    record = TRANSIT_LEDGER.get(package_id)
    return copy.deepcopy(record) if record is not None else None


def known_ids() -> list:
    return sorted(TRANSIT_LEDGER)
