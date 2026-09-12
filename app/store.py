"""
Durable store for everything the engine observes and decides.

WHY SQLITE, AND WHY NOT SOMETHING BIGGER
This replaces an append-only JSONL file. That file was genuinely durable - it
survived restarts and rebuilds through the compose bind mount - but answering
"how many damaged parcels did we see last Tuesday, above 0.8 confidence" meant
reading every line and filtering in Python. The gain here is queryability, not
persistence, and it is worth being precise about that.

sqlite3 is in the standard library, so this adds no dependency. A client/server
database would add a service, a connection pool and a failure mode, to serve a
single-process container that writes a few rows per request. That is the wrong
trade at this size. The migration path if it ever stops being the right one is
Postgres behind the same three functions below.

TWO TABLES, BECAUSE THEY ANSWER DIFFERENT QUESTIONS
  detections     every /detect call: what the model saw in an image
  adjudications  every /reason call: what the system decided, and on what evidence

They are deliberately not one table. A detection is an observation; an
adjudication is a decision with a guardrail outcome and a ledger digest behind
it. Collapsing them would mean a column that is null half the time.

WRITING MUST NEVER BREAK A REQUEST
Every write is wrapped. If the database is unwritable - a read-only mount, a
full disk, a serverless filesystem - the failure is logged once and the API
answers normally. Recording is observability, not part of the contract.

APPEND-ONLY BY CONVENTION, NOT BY ENFORCEMENT
Nothing here updates or deletes a row. SQLite would happily allow both, so this
is a property of the code rather than of the storage, and the distinction is
worth stating plainly rather than implying tamper-resistance that is not there.
"""

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("exception-engine.store")

DB_PATH = Path(os.getenv("ENGINE_DB_PATH", "logs/engine.db"))
LEGACY_JSONL = Path(os.getenv("EXCEPTION_LOG_PATH", "logs/exceptions.jsonl"))

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None
_warned = False

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at       TEXT    NOT NULL,
    filename          TEXT,
    image_width       INTEGER,
    image_height      INTEGER,
    inference_time_ms REAL,
    count             INTEGER NOT NULL,
    max_confidence    REAL,
    has_damage        INTEGER NOT NULL,
    detections_json   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_det_time   ON detections(recorded_at);
CREATE INDEX IF NOT EXISTS idx_det_damage ON detections(has_damage);

CREATE TABLE IF NOT EXISTS adjudications (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at             TEXT    NOT NULL,
    package_id              TEXT    NOT NULL,
    query                   TEXT    NOT NULL,
    status                  TEXT    NOT NULL,
    requires_vision_model   INTEGER NOT NULL,
    guardrail_passed        INTEGER NOT NULL,
    max_critical_confidence REAL,
    detections_json         TEXT    NOT NULL,
    ledger_digest           TEXT    NOT NULL,
    ledger_intact           INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_adj_time    ON adjudications(recorded_at);
CREATE INDEX IF NOT EXISTS idx_adj_status  ON adjudications(status);
CREATE INDEX IF NOT EXISTS idx_adj_package ON adjudications(package_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _warn_once(exc: Exception) -> None:
    global _warned
    if not _warned:
        log.warning("store unwritable at %s (%s); requests are still served, "
                    "just not recorded", DB_PATH, exc)
        _warned = True


def _connect() -> Optional[sqlite3.Connection]:
    """Open once and reuse. Returns None if the path cannot be opened."""
    global _conn
    if _conn is not None:
        return _conn
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because FastAPI serves from a threadpool; the
        # module-level lock below is what actually serialises writes.
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL lets the dashboard read while a request is writing.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        conn.commit()
        _conn = conn
        _migrate_legacy_jsonl(conn)
        return _conn
    except Exception as exc:                      # noqa: BLE001
        _warn_once(exc)
        return None


def _migrate_legacy_jsonl(conn: sqlite3.Connection) -> None:
    """Carry the old JSONL rows in once, so history is not lost on upgrade.

    Guarded on the table being empty rather than on a flag file: re-running is
    then harmless, and a fresh database with an old log still picks it up.
    """
    try:
        if conn.execute("SELECT COUNT(*) FROM adjudications").fetchone()[0]:
            return
        if not LEGACY_JSONL.is_file():
            return
        rows = []
        for line in LEGACY_JSONL.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append((
                e.get("recorded_at") or _now(), e.get("package_id", ""),
                e.get("query", ""), e.get("status", ""),
                int(bool(e.get("requires_vision_model"))),
                int(bool(e.get("guardrail_passed"))),
                e.get("max_critical_confidence"),
                json.dumps(e.get("detections") or []),
                e.get("ledger_digest", ""), int(bool(e.get("ledger_intact", True))),
            ))
        if rows:
            conn.executemany(
                "INSERT INTO adjudications (recorded_at, package_id, query, status,"
                " requires_vision_model, guardrail_passed, max_critical_confidence,"
                " detections_json, ledger_digest, ledger_intact)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
            conn.commit()
            log.info("migrated %d rows from %s into %s", len(rows), LEGACY_JSONL, DB_PATH)
    except Exception as exc:                      # noqa: BLE001
        log.warning("legacy log migration skipped: %s", exc)


def close() -> None:
    """Close the connection. Used on shutdown and by tests.

    Windows will not delete a file that still has an open handle, so leaving
    the connection open leaks into anything that cleans up a temp directory.
    """
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:                         # noqa: BLE001
            pass
        _conn = None


# --------------------------------------------------------------------------
# writes
# --------------------------------------------------------------------------

def record_detection(filename: Optional[str], image_size: List[int],
                     inference_time_ms: float, detections: List[dict]) -> None:
    """Append one /detect observation. Never raises."""
    conn = _connect()
    if conn is None:
        return
    confs = [float(d["confidence"]) for d in (detections or [])]
    has_damage = any(d["label"] == "damaged-package" for d in (detections or []))
    try:
        with _lock:
            conn.execute(
                "INSERT INTO detections (recorded_at, filename, image_width,"
                " image_height, inference_time_ms, count, max_confidence,"
                " has_damage, detections_json) VALUES (?,?,?,?,?,?,?,?,?)",
                (_now(), filename,
                 image_size[0] if image_size else None,
                 image_size[1] if len(image_size or []) > 1 else None,
                 round(float(inference_time_ms), 2), len(detections or []),
                 round(max(confs), 4) if confs else None,
                 int(has_damage),
                 json.dumps([{"label": d["label"],
                              "confidence": round(float(d["confidence"]), 4),
                              "bbox": [round(float(v), 1) for v in d["bbox"]]}
                             for d in (detections or [])])))
            conn.commit()
    except Exception as exc:                      # noqa: BLE001
        _warn_once(exc)


def record_adjudication(package_id: str, query: str, status: str,
                        requires_vision_model: bool, guardrail_passed: bool,
                        max_critical_confidence: Optional[float],
                        detections: List[dict], ledger_digest: str,
                        ledger_intact: bool) -> None:
    """Append one /reason decision. Never raises."""
    conn = _connect()
    if conn is None:
        return
    try:
        with _lock:
            conn.execute(
                "INSERT INTO adjudications (recorded_at, package_id, query, status,"
                " requires_vision_model, guardrail_passed, max_critical_confidence,"
                " detections_json, ledger_digest, ledger_intact)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (_now(), package_id, query, status,
                 int(bool(requires_vision_model)), int(bool(guardrail_passed)),
                 max_critical_confidence,
                 json.dumps([{"label": d["label"],
                              "confidence": round(float(d["confidence"]), 4)}
                             for d in (detections or [])]),
                 ledger_digest, int(bool(ledger_intact))))
            conn.commit()
    except Exception as exc:                      # noqa: BLE001
        _warn_once(exc)


# --------------------------------------------------------------------------
# reads
# --------------------------------------------------------------------------

def recent_detections(limit: int = 25) -> List[dict]:
    conn = _connect()
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"recorded_at": r["recorded_at"], "filename": r["filename"],
                 "image_size": [r["image_width"], r["image_height"]],
                 "inference_time_ms": r["inference_time_ms"], "count": r["count"],
                 "max_confidence": r["max_confidence"],
                 "has_damage": bool(r["has_damage"]),
                 "detections": json.loads(r["detections_json"])} for r in rows]
    except Exception:                             # noqa: BLE001
        return []


def recent_adjudications(limit: int = 25) -> List[dict]:
    conn = _connect()
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT * FROM adjudications ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"recorded_at": r["recorded_at"], "package_id": r["package_id"],
                 "query": r["query"], "status": r["status"],
                 "requires_vision_model": bool(r["requires_vision_model"]),
                 "guardrail_passed": bool(r["guardrail_passed"]),
                 "max_critical_confidence": r["max_critical_confidence"],
                 "detections": json.loads(r["detections_json"]),
                 "ledger_digest": r["ledger_digest"],
                 "ledger_intact": bool(r["ledger_intact"])} for r in rows]
    except Exception:                             # noqa: BLE001
        return []


def stats() -> dict:
    """Counts the dashboard shows. Returns zeros rather than failing."""
    conn = _connect()
    empty = {"detections": 0, "adjudications": 0, "images_with_damage": 0,
             "exceptions_flagged": 0, "refusals": 0, "db_path": str(DB_PATH),
             "available": False}
    if conn is None:
        return empty
    try:
        q = lambda sql: conn.execute(sql).fetchone()[0]      # noqa: E731
        return {
            "detections": q("SELECT COUNT(*) FROM detections"),
            "adjudications": q("SELECT COUNT(*) FROM adjudications"),
            "images_with_damage": q("SELECT COUNT(*) FROM detections WHERE has_damage=1"),
            "exceptions_flagged": q("SELECT COUNT(*) FROM adjudications"
                                    " WHERE status='EXCEPTION_FLAGGED'"),
            "refusals": q("SELECT COUNT(*) FROM adjudications"
                          " WHERE status='INSUFFICIENT_INFORMATION'"),
            "db_path": str(DB_PATH),
            "available": True,
        }
    except Exception:                             # noqa: BLE001
        return empty
