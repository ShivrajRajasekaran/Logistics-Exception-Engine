"""
Append-only record of every adjudication the reasoning layer makes.

WHY THIS EXISTS
The reasoning layer deliberately does not write to the transit ledger: an
inspection at a delivery hub adjudicates a parcel's condition, it does not get
to revise what origin recorded. But without somewhere to put its *output*, a
verdict existed only in the HTTP response and was gone the moment the container
restarted. There was no answer to "what did this system decide about PKG-8821,
and on what evidence".

So the ledger stays read-only and the verdict goes here instead: one JSON object
per line, appended, never rewritten. Each line carries the package, the
question, the status, the guardrail outcome, the detections that drove it, and
the ledger digest that was in force at the time, so an entry can be checked
against the custody records it was reconciled against.

DELIBERATE LIMITS, STATED RATHER THAN IMPLIED
This is append-only *by construction* - the file is opened in "a" mode and
nothing here seeks, truncates or rewrites. It is not tamper-proof: anything with
write access to the file can edit it afterwards. Making that detectable needs a
hash chain over entries and a copy held somewhere the serving process cannot
reach. That is the right next step and is not implemented.

Writing must never break an adjudication. If the path is unwritable - a
read-only mount, a full disk - the failure is logged once and the API answers
normally.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("exception-engine.audit")

LOG_PATH = Path(os.getenv("EXCEPTION_LOG_PATH", "logs/exceptions.jsonl"))
MAX_READ_BYTES = 4 * 1024 * 1024

# One process, one uvicorn worker, but the endpoint is async and a burst of
# concurrent requests can interleave. A lock keeps each line intact.
_lock = threading.Lock()
_warned = False


def _warn_once(exc: Exception) -> None:
    global _warned
    if not _warned:
        log.warning("exception log unwritable at %s (%s); adjudications will still "
                    "be served, just not recorded", LOG_PATH, exc)
        _warned = True


def record(package_id: str, query: str, status: str, guardrail_passed: bool,
           requires_vision_model: bool, max_critical_confidence: Optional[float],
           detections: List[dict], ledger_digest: str,
           ledger_intact: bool) -> None:
    """Append one adjudication. Never raises."""
    entry = {
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "package_id": package_id,
        "query": query,
        "status": status,
        "requires_vision_model": requires_vision_model,
        "guardrail_passed": guardrail_passed,
        "max_critical_confidence": max_critical_confidence,
        "detections": [
            {"label": d["label"], "confidence": round(float(d["confidence"]), 4)}
            for d in (detections or [])
        ],
        "ledger_digest": ledger_digest,
        "ledger_intact": ledger_intact,
    }
    try:
        with _lock:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except Exception as exc:          # noqa: BLE001 - logging must not break serving
        _warn_once(exc)


def tail(limit: int = 20) -> List[dict]:
    """Most recent entries, newest first. Returns [] if nothing is recorded yet."""
    if not LOG_PATH.is_file():
        return []
    try:
        with LOG_PATH.open("r", encoding="utf-8") as fh:
            # Bounded read: the file grows without limit by design, and a demo
            # console asking for 20 entries should not pull megabytes.
            size = LOG_PATH.stat().st_size
            if size > MAX_READ_BYTES:
                fh.seek(size - MAX_READ_BYTES)
                fh.readline()          # discard the partial line seeking landed in
            lines = fh.readlines()
    except Exception as exc:          # noqa: BLE001
        log.warning("could not read exception log: %s", exc)
        return []

    out = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue                   # a truncated tail line is not worth failing over
        if len(out) >= limit:
            break
    return out


def count() -> int:
    if not LOG_PATH.is_file():
        return 0
    try:
        with LOG_PATH.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except Exception:                  # noqa: BLE001
        return 0
