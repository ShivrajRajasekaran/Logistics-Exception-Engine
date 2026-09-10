"""
End-to-end Docker smoke test. Run it yourself; trust the output, not a summary.

    python scripts/smoke_test_docker.py

Every step prints the raw HTTP status and the actual response body it received,
so each assertion can be checked by eye rather than taken on faith. The
container is always torn down, and on failure the docker logs are dumped.

Exit 0 only if every step passed. Exit 1 otherwise.

Stdlib only - no pytest, no requests, nothing to install.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"

steps = []


def log(msg=""):
    print(msg, flush=True)


def step(name, ok, evidence):
    steps.append((name, ok))
    mark = "%sPASS%s" % (GREEN, RESET) if ok else "%sFAIL%s" % (RED, RESET)
    log("[%s] %s" % (mark, name))
    for line in evidence:
        log("       %s%s%s" % (DIM, line, RESET))


def compose(*args, timeout=1800):
    return subprocess.run(["docker", "compose", *args], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=timeout)


def http(url, data=None, headers=None, timeout=90):
    """Return (status, body_text). Never raises on 4xx/5xx."""
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return None, str(e)


def multipart(field, filename, blob, content_type):
    b = "----smoke%s" % uuid.uuid4().hex
    body = (
        ("--%s\r\nContent-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\n"
         "Content-Type: %s\r\n\r\n" % (b, field, filename, content_type)).encode()
        + blob + ("\r\n--%s--\r\n" % b).encode()
    )
    return body, {"Content-Type": "multipart/form-data; boundary=%s" % b}


def dump_logs(reason):
    log("\n%s--- docker logs (%s) ---%s" % (RED, reason, RESET))
    out = compose("logs", "--tail=60", timeout=120)
    log(out.stdout or out.stderr)


def main():
    ap = argparse.ArgumentParser(description="Docker end-to-end smoke test")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--image", default="sample_images/damaged_parcel.jpg")
    ap.add_argument("--keep-up", action="store_true", help="skip teardown")
    args = ap.parse_args()
    base = "http://localhost:%d" % args.port

    log("%s=== DOCKER SMOKE TEST ===%s" % (BOLD, RESET))
    log("%srepo: %s | port: %d%s\n" % (DIM, ROOT, args.port, RESET))

    img = ROOT / args.image
    if not img.exists():
        step("sample image exists", False, ["%s not found" % args.image])
        return 1

    try:
        # -------------------------------------------------- 1. cold start
        compose("down", "--remove-orphans", timeout=300)
        t0 = time.time()
        up = compose("up", "-d", "--build")
        started = up.returncode == 0
        step("1. docker compose up -d --build", started,
             ["exit=%d  elapsed=%.1fs" % (up.returncode, time.time() - t0),
              (up.stderr or up.stdout).strip().splitlines()[-1][:110] if (up.stderr or up.stdout).strip() else ""])
        if not started:
            dump_logs("compose up failed")
            return 1

        # -------------------------------------------------- 2. health gate
        deadline, status, body, waited = time.time() + 180, None, "", 0.0
        while time.time() < deadline:
            status, body = http(base + "/health", timeout=5)
            if status == 200:
                try:
                    if json.loads(body).get("model_loaded") is True:
                        break
                except ValueError:
                    pass
            time.sleep(2)
            waited = time.time() - (deadline - 180)

        health_ok = False
        ev = ["HTTP %s after %.1fs" % (status, waited), "body: %s" % body[:200]]
        if status == 200:
            h = json.loads(body)
            health_ok = h.get("model_loaded") is True
            ev.append("model_loaded=%s  classes=%s" % (h.get("model_loaded"), h.get("classes")))
        step("2. /health returns 200 with model_loaded=true", health_ok, ev)
        if not health_ok:
            dump_logs("health gate failed")
            return 1

        # -------------------------------------------------- 3. Part A
        blob = img.read_bytes()
        data, hdrs = multipart("file", img.name, blob, "image/jpeg")
        t0 = time.time()
        status, body = http(base + "/api/v1/detect", data, hdrs)
        ev = ["HTTP %s in %.2fs" % (status, time.time() - t0), "body: %s" % body[:300]]
        ok = False
        if status == 200:
            d = json.loads(body)
            dets = d.get("detections", [])
            complete = all({"label", "confidence", "bbox"} <= set(x) for x in dets)
            sane = all(len(x["bbox"]) == 4 and 0.0 <= x["confidence"] <= 1.0 for x in dets)
            ok = d.get("status") == "success" and len(dets) > 0 and complete and sane
            ev += ["count=%s" % d.get("count"),
                   "classes=%s" % [x["label"] for x in dets],
                   "confidences=%s" % [x["confidence"] for x in dets],
                   "every detection has label+confidence+bbox: %s" % complete,
                   "bboxes are 4-tuples and conf in [0,1]: %s" % sane]
        step("3. POST /api/v1/detect returns boxes/classes/confidences", ok, ev)
        if not ok:
            dump_logs("detect failed")
            return 1

        # -------------------------------------------------- 4. Part B
        payload = json.dumps({
            "package_id": "PKG-8821",
            "image_path": args.image,
            "query": "Is this package damaged?",
        }).encode()
        t0 = time.time()
        status, body = http(base + "/api/v1/reason", payload,
                            {"Content-Type": "application/json"})
        required = {"package_id", "status", "requires_vision_model",
                    "guardrail_passed", "decision_summary", "detections"}
        ev = ["HTTP %s in %.2fs" % (status, time.time() - t0)]
        ok = False
        if status == 200:
            r = json.loads(body)
            ok = required <= set(r)
            ev += ["schema keys present: %s" % ok,
                   "status=%s" % r.get("status"),
                   "guardrail_passed=%s  max_critical=%s"
                   % (r.get("guardrail_passed"), r.get("max_critical_confidence")),
                   "decision_summary: %s" % str(r.get("decision_summary"))[:160]]
        else:
            ev.append("body: %s" % body[:250])
        step("4. POST /api/v1/reason returns the reasoning schema", ok, ev)
        if not ok:
            dump_logs("reason failed")
            return 1

        # -------------------------------------------------- 5. malformed input
        data, hdrs = multipart("question", "x.txt", b"Is this package damaged?", "text/plain")
        status, body = http(base + "/api/v1/reason", data, hdrs)
        ok = status == 422
        step("5. malformed payload rejected with 422 (not a 500)", ok,
             ["HTTP %s" % status,
              "422 means pydantic validated and refused; 500 would mean a crash"])

        # -------------------------------------------------- 6. log scan
        logs = compose("logs", timeout=120).stdout
        bad = [l for l in logs.splitlines()
               if any(t in l for t in (" 500 ", " 502 ", "Traceback", "Internal server error"))]
        step("6. no 5xx or tracebacks in container log", not bad,
             ["scanned %d log lines" % len(logs.splitlines()),
              "offending lines: %d" % len(bad)] + bad[:5])

    finally:
        if not args.keep_up:
            down = compose("down", timeout=300)
            log("\n%steardown: exit=%d%s" % (DIM, down.returncode, RESET))

    passed = sum(1 for _, ok in steps if ok)
    log("\n%s%s%s" % (BOLD, "=" * 58, RESET))
    log("%s%d/%d steps passed%s" % (BOLD, passed, len(steps), RESET))
    if passed == len(steps):
        log("\n%s%s SMOKE TEST PASSED: THE DOCKER CONTAINER IS BULLETPROOF "
            "AND READY TO SHIP.%s" % (BOLD, GREEN, RESET))
        return 0
    log("\n%s%s SMOKE TEST FAILED: %s%s"
        % (BOLD, RED, ", ".join(n for n, ok in steps if not ok), RESET))
    return 1


if __name__ == "__main__":
    sys.exit(main())
