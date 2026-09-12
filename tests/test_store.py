"""
Tests for the durable store.

The module this replaces had no coverage, which is part of why replacing it was
acceptable and also why these exist now. Like the reasoning tests, these need no
GPU, no checkpoint and no API key: the store is our own code and is verifiable
on its own terms.

Every test runs against a throwaway database, so nothing here touches the real
logs/engine.db.
"""

import importlib
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LEGACY_ROW = (
    '{"recorded_at":"2026-01-01T00:00:00.000+00:00","package_id":"PKG-1",'
    '"query":"q","status":"CLEAR","requires_vision_model":true,'
    '"guardrail_passed":true,"max_critical_confidence":0.0,'
    '"detections":[],"ledger_digest":"abc","ledger_intact":true}\n'
)

DET = [{"label": "damaged-package", "confidence": 0.8712, "bbox": [1.0, 2.0, 3.0, 4.0]}]


@contextmanager
def store_in_tmp(legacy_text=None):
    """Yield the store bound to a throwaway database, then close and clean up.

    The module holds its connection and path at module level, so a reimport is
    the honest way to rebind it rather than reaching into private globals. The
    connection must be closed before the directory is removed, because Windows
    refuses to delete a file that still has an open handle.
    """
    tmp = tempfile.mkdtemp()
    try:
        legacy = Path(tmp) / ("old.jsonl" if legacy_text else "absent.jsonl")
        if legacy_text:
            legacy.write_text(legacy_text, encoding="utf-8")
        os.environ["ENGINE_DB_PATH"] = str(Path(tmp) / "t.db")
        os.environ["EXCEPTION_LOG_PATH"] = str(legacy)
        import app.store as store
        importlib.reload(store)
        yield store
    finally:
        try:
            import app.store as store
            store.close()
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)


class TestDetectionRecording(unittest.TestCase):
    def test_detection_round_trips(self):
        with store_in_tmp() as s:
            s.record_detection("a.jpg", [640, 480], 120.5, DET)
            rows = s.recent_detections()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["filename"], "a.jpg")
            self.assertEqual(rows[0]["count"], 1)
            self.assertTrue(rows[0]["has_damage"])
            self.assertAlmostEqual(rows[0]["max_confidence"], 0.8712, places=4)

    def test_empty_detection_is_recorded_not_skipped(self):
        """An image with nothing in it is a real observation and must be kept."""
        with store_in_tmp() as s:
            s.record_detection("empty.jpg", [640, 480], 90.0, [])
            rows = s.recent_detections()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["count"], 0)
            self.assertFalse(rows[0]["has_damage"])
            self.assertIsNone(rows[0]["max_confidence"])

    def test_has_damage_false_for_intact_only(self):
        with store_in_tmp() as s:
            s.record_detection("i.jpg", [640, 480], 80.0,
                               [{"label": "package", "confidence": 0.95,
                                 "bbox": [0, 0, 1, 1]}])
            self.assertFalse(s.recent_detections()[0]["has_damage"])

    def test_newest_first(self):
        with store_in_tmp() as s:
            for name in ("one.jpg", "two.jpg", "three.jpg"):
                s.record_detection(name, [10, 10], 1.0, [])
            self.assertEqual([r["filename"] for r in s.recent_detections()],
                             ["three.jpg", "two.jpg", "one.jpg"])


class TestAdjudicationRecording(unittest.TestCase):
    def test_adjudication_round_trips(self):
        with store_in_tmp() as s:
            s.record_adjudication("PKG-8821", "Is it damaged?", "EXCEPTION_FLAGGED",
                                  True, True, 0.8712, DET, "deadbeef", True)
            row = s.recent_adjudications()[0]
            self.assertEqual(row["package_id"], "PKG-8821")
            self.assertEqual(row["status"], "EXCEPTION_FLAGGED")
            self.assertTrue(row["guardrail_passed"])
            self.assertEqual(row["ledger_digest"], "deadbeef")

    def test_refusal_keeps_null_confidence(self):
        """A capability refusal never measured a confidence, so it must stay
        null rather than becoming 0.0, which would read as 'looked, saw nothing'."""
        with store_in_tmp() as s:
            s.record_adjudication("PKG-8821", "How many people?",
                                  "UNSUPPORTED_CAPABILITY", False, False,
                                  None, [], "deadbeef", True)
            self.assertIsNone(s.recent_adjudications()[0]["max_critical_confidence"])

    def test_tampered_ledger_flag_is_preserved(self):
        with store_in_tmp() as s:
            s.record_adjudication("PKG-8821", "q", "CLEAR", True, True, 0.0,
                                  [], "digest", False)
            self.assertFalse(s.recent_adjudications()[0]["ledger_intact"])


class TestTablesAreSeparate(unittest.TestCase):
    def test_detections_do_not_appear_as_adjudications(self):
        with store_in_tmp() as s:
            s.record_detection("a.jpg", [1, 1], 1.0, DET)
            self.assertEqual(len(s.recent_adjudications()), 0)
            self.assertEqual(len(s.recent_detections()), 1)


class TestStats(unittest.TestCase):
    def test_counts_are_accurate(self):
        with store_in_tmp() as s:
            s.record_detection("a.jpg", [1, 1], 1.0, DET)        # has damage
            s.record_detection("b.jpg", [1, 1], 1.0, [])         # nothing found
            s.record_adjudication("P", "q", "EXCEPTION_FLAGGED", True, True,
                                  0.9, DET, "x", True)
            s.record_adjudication("P", "q", "INSUFFICIENT_INFORMATION", True,
                                  False, 0.6, DET, "x", True)
            st = s.stats()
            self.assertEqual(st["detections"], 2)
            self.assertEqual(st["images_with_damage"], 1)
            self.assertEqual(st["adjudications"], 2)
            self.assertEqual(st["exceptions_flagged"], 1)
            self.assertEqual(st["refusals"], 1)
            self.assertTrue(st["available"])


class TestFailureIsNeverFatal(unittest.TestCase):
    """Recording is observability. It must not be able to break a request."""

    def test_unwritable_path_does_not_raise(self):
        tmp = tempfile.mkdtemp()
        try:
            blocker = Path(tmp) / "a.txt"
            blocker.write_text("not a directory", encoding="utf-8")
            os.environ["ENGINE_DB_PATH"] = str(blocker / "nested" / "t.db")
            os.environ["EXCEPTION_LOG_PATH"] = str(Path(tmp) / "absent.jsonl")
            import app.store as store
            importlib.reload(store)
            store.record_detection("a.jpg", [1, 1], 1.0, DET)
            store.record_adjudication("P", "q", "CLEAR", True, True, 0.0, [],
                                      "x", True)
            self.assertEqual(store.recent_detections(), [])
            self.assertEqual(store.recent_adjudications(), [])
            self.assertFalse(store.stats()["available"])
            store.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestLegacyMigration(unittest.TestCase):
    def test_jsonl_rows_are_carried_in_once(self):
        text = LEGACY_ROW + "\n" + "{not json}\n"   # blank and corrupt lines skipped
        with store_in_tmp(legacy_text=text) as s:
            rows = s.recent_adjudications()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["package_id"], "PKG-1")

    def test_migration_does_not_duplicate_on_reopen(self):
        with store_in_tmp(legacy_text=LEGACY_ROW) as s:
            self.assertEqual(len(s.recent_adjudications()), 1)
            s.close()
            importlib.reload(s)                    # simulate a restart
            self.assertEqual(len(s.recent_adjudications()), 1)


class TestAppendOnlyByConvention(unittest.TestCase):
    def test_module_exposes_no_update_or_delete(self):
        """Nothing should offer a way to rewrite history through this API."""
        with store_in_tmp() as s:
            names = [n for n in dir(s) if not n.startswith("_")]
            for forbidden in ("delete", "update", "remove", "purge", "clear"):
                offenders = [n for n in names if forbidden in n.lower()]
                self.assertFalse(offenders,
                                 "store exposes a %s-like function: %s"
                                 % (forbidden, offenders))


if __name__ == "__main__":
    unittest.main(verbosity=2)
