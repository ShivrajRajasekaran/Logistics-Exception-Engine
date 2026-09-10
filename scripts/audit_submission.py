"""
Adversarial self-audit against the RAP screening rubric.

Run this before submitting. It grades the repository as a hostile reviewer
would, and every verdict prints the evidence it was derived from so you can
verify the checker rather than trusting it.

    python scripts/audit_submission.py            # static audit
    python scripts/audit_submission.py --live     # also boot the API and probe it
    python scripts/audit_submission.py --docker   # also build and run the container

Exit code 0 if no hard constraint failed and the score clears --min-score
(default 95). Exit code 1 otherwise, so this can gate a commit hook or CI.

Design rule: no check may hardcode an expected answer. Each one reads the repo,
runs the code, or executes a subprocess, then reports what it actually found.
"""

import argparse
import ast
import json
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FRAMEWORKS = ["langchain", "llama_index", "llamaindex", "crewai",
              "autogen", "semantic_kernel", "haystack", "langgraph"]

# The 80 standard COCO classes. A taxonomy drawn only from this list scores
# zero on Part A per the brief.
COCO80 = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
}

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m")

_DUMMY_IMAGE_ARG = "/nonexistent/probe.jpg"

results = []
hard_failures = []


def record(phase, name, earned, possible, evidence, hard=False):
    verdict = "PASS" if earned == possible else ("FAIL" if earned == 0 else "PARTIAL")
    results.append((phase, name, earned, possible, verdict, evidence))
    if hard and verdict == "FAIL":
        hard_failures.append(name)
    colour = GREEN if verdict == "PASS" else (RED if verdict == "FAIL" else YELLOW)
    score = "HARD" if possible == 0 else "%2d/%-3d" % (earned, possible)
    print("%s[%s]%s %-46s %s" % (colour, verdict.center(7), RESET, name, score))
    for line in evidence:
        print("        %s%s%s" % (DIM, line, RESET))


def read(path):
    p = ROOT / path
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def git(*args):
    try:
        out = subprocess.run(["git", "-C", str(ROOT), *args],
                             capture_output=True, text=True, timeout=30)
        return out.stdout
    except Exception:
        return ""


def py_files():
    return sorted(
        p for d in ("app", "scripts", "tests")
        for p in (ROOT / d).rglob("*.py")
        if "__pycache__" not in str(p)
    )


# ---------------------------------------------------------------- PHASE 1

def check_frameworks():
    """Distinguish a real import from prose mentioning the ban.

    Parses the AST of every file and inspects only actual import statements.
    A grep alone cannot tell a functional dependency from a docstring, and
    grading them the same way is how a compliant repo gets failed.
    """
    offenders, prose = [], []
    for f in py_files():
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            record("1", "Hard: frameworks", 0, 0,
                   ["%s does not parse: %s" % (f.name, exc)], hard=True)
            return
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            for m in mods:
                if any(fw in m.lower().replace("-", "_") for fw in FRAMEWORKS):
                    offenders.append("%s:%d imports %s" % (f.name, node.lineno, m))

    for path in ("requirements.txt", "requirements-cuda.txt",
                 "requirements-data.txt", "Dockerfile"):
        for i, line in enumerate(read(path).splitlines(), 1):
            bare = line.split("#")[0].strip().lower()
            if bare and any(fw in bare for fw in FRAMEWORKS):
                offenders.append("%s:%d declares %s" % (path, i, line.strip()))

    for f in py_files():
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if any(fw in line.lower() for fw in FRAMEWORKS):
                if not any(o.startswith("%s:%d" % (f.name, i)) for o in offenders):
                    prose.append("%s:%d (prose/docstring, not an import)" % (f.name, i))

    ev = ["scanned %d py files + 4 dependency manifests via AST" % len(py_files())]
    ev += ["non-functional mention: " + p for p in prose[:4]]
    if offenders:
        ev = ["FUNCTIONAL DEPENDENCY FOUND:"] + offenders
    record("1", "Hard: no agentic frameworks", 0 if offenders else 1, 1, ev, hard=True)


def check_taxonomy():
    import yaml
    spec = yaml.safe_load(read("dataset/data.yaml") or "{}") or {}
    names = spec.get("names", {})
    names = list(names.values()) if isinstance(names, dict) else list(names)
    non_coco = [n for n in names if n.lower() not in COCO80]

    detector = read("app/detector.py")
    m = re.search(r"CRITICAL_CLASSES\s*=\s*\{([^}]*)\}", detector)
    critical = re.findall(r'"([^"]+)"', m.group(1)) if m else []
    drift = [c for c in critical if c not in names]

    ev = ["data.yaml nc=%s names=%s" % (spec.get("nc"), names),
          "non-COCO classes: %s" % (non_coco or "NONE"),
          "detector CRITICAL_CLASSES=%s" % critical]
    if drift:
        ev.append("DRIFT: critical class(es) %s absent from data.yaml" % drift)
    ok = bool(non_coco) and not drift
    record("1", "Hard: >=1 non-COCO class, no taxonomy drift",
           1 if ok else 0, 1, ev, hard=True)


def check_reproducibility():
    text = read("REPRODUCIBILITY.md")
    required = {
        "GPU model": r"RTX\s*\d{4}",
        "VRAM": r"\d+(\.\d+)?\s*GB",
        "CUDA/torch build": r"cu128|2\.7\.\d+",
        "training time": r"\d+(\.\d+)?\s*(hours|hrs|h\b|min)",
        "batch size": r"[Bb]atch\s*size",
        "seed": r"\b42\b",
        "hyperparameters": r"lr0|AdamW|weight.decay",
        "train command": r"scripts/train\.py",
    }
    missing = [k for k, pat in required.items() if not re.search(pat, text)]
    ev = ["REPRODUCIBILITY.md: %d chars" % len(text),
          "present: %s" % ", ".join(k for k in required if k not in missing)]
    if missing:
        ev.append("MISSING: %s  -> Part A capped at 50%%" % ", ".join(missing))
    record("1", "Hard: reproducibility documented",
           1 if not missing else 0, 1, ev, hard=True)


# ---------------------------------------------------------------- PHASE 2

def check_weights_deliverable():
    """The brief demands weights OR a direct working download path.

    Graded on what a reviewer gets from `git clone`, not what sits in your
    working directory.
    """
    tracked = [l for l in git("ls-files").splitlines() if l.endswith(".pt")]
    on_disk = list((ROOT / "weights").glob("*.pt")) if (ROOT / "weights").exists() else []
    readme = read("README.md")
    urls = re.findall(r"https?://[^\s)\]]+(?:releases/download|\.pt)[^\s)\]]*", readme)
    ignored = bool(re.search(r"^\s*weights/\*?\.?pt", read(".gitignore"), re.M))

    ev = ["git-tracked .pt files: %s" % (tracked or "NONE"),
          "weights/ on local disk: %s" % ([p.name for p in on_disk] or "none"),
          ".gitignore excludes weights: %s" % ignored,
          "download URL in README: %s" % (urls or "NONE FOUND")]
    if tracked:
        earned = 8
        ev.append("weights committed -> reviewer can run the hidden set")
    elif urls:
        earned = 8
        ev.append("direct download documented -> acceptable per brief")
    else:
        earned = 0
        ev.append("BLOCKER: a fresh clone has no model and no way to get one.")
        ev.append("Hidden-set score (25%) is unobtainable without retraining.")
    record("2", "Weights deliverable (brief item 2)", earned, 8, ev)


def check_api_robustness():
    src = read("app/main.py")
    wanted = {
        "415 wrong content type": r"status_code=415",
        "400 undecodable/empty": r"status_code=400",
        "413 oversize": r"status_code=413",
        "503 model missing": r"status_code=503",
        "500 catch-all": r"status_code=500",
        "UploadFile signature": r"UploadFile\s*=\s*File\(",
        "latency logging": r"perf_counter\(\)",
    }
    found = {k: bool(re.search(p, src)) for k, p in wanted.items()}
    earned = round(9 * sum(found.values()) / len(found))
    ev = ["present: %s" % ", ".join(k for k, v in found.items() if v)]
    miss = [k for k, v in found.items() if not v]
    if miss:
        ev.append("MISSING: %s" % ", ".join(miss))
    record("2", "API robustness (static)", earned, 9, ev)


def check_partb_behaviour():
    """Execute the decision layer rather than reading it."""
    sys.path.insert(0, str(ROOT))
    try:
        from app import reasoning
    except Exception as exc:
        record("2", "Part B behaviour (executed)", 0, 15,
               ["import failed: %s" % exc])
        return

    ev, earned = [], 0

    route, why = reasoning.route_intent("Which carrier handled this in transit?",
                                        _DUMMY_IMAGE_ARG)
    ok1 = route == reasoning.ROUTE_LEDGER
    earned += 5 if ok1 else 0
    ev.append("[%s] ledger-only query bypasses vision -> %s" % ("OK" if ok1 else "XX", route))

    passed, why2 = reasoning.evaluate_guardrail(
        [{"label": "damaged-package", "confidence": 0.64,
          "bbox": [0, 0, 1, 1]}], 0.64)
    ok2 = passed is False
    earned += 5 if ok2 else 0
    ev.append("[%s] 0.64 < 0.65 halts -> passed=%s" % ("OK" if ok2 else "XX", passed))

    at = reasoning.evaluate_guardrail(
        [{"label": "damaged-package", "confidence": 0.65, "bbox": [0, 0, 1, 1]}], 0.65)[0]
    ev.append("[%s] 0.65 boundary is inclusive -> passed=%s" % ("OK" if at else "XX", at))

    ledger = {"carrier": "Apex Logistics", "origin_hub": "HUB-01",
              "origin_label_status": "INTACT", "origin_seal_status": "INTACT",
              "dispatched_at": "2026-09-04", "sku_manifest": "SKU-9901",
              "transit_history": [{"hub": "A"}]}
    import os
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        out = reasoning.synthesize("Which carrier?", [], {}, ledger)
    finally:
        if saved:
            os.environ["OPENAI_API_KEY"] = saved
    ok3 = "Apex Logistics" in out
    earned += 5 if ok3 else 0
    ev.append("[%s] offline fallback reconciles ledger -> %s"
              % ("OK" if ok3 else "XX", out[:70]))

    overlap = reasoning.VISION_TOKENS & reasoning.UNSUPPORTED_TOKENS
    ev.append("[%s] routing vocabularies disjoint (%d overlaps)"
              % ("OK" if not overlap else "XX", len(overlap)))
    record("2", "Part B behaviour (executed)", earned, 15, ev)


def check_metric_honesty():
    """Cross-check every number the memo quotes against metrics.json.

    This is the check that catches an inflated memo, so it compares values
    rather than looking for the presence of a metrics section.
    """
    mpath = ROOT / "runs" / "eval" / "metrics.json"
    if not mpath.exists():
        record("2", "Metric honesty (memo vs metrics.json)", 0, 10,
               ["runs/eval/metrics.json absent - no measured evidence to verify against"])
        return
    metrics = json.loads(mpath.read_text())
    memo = read("MEMO.md")

    truth = {"overall mAP50": metrics["overall"]["mAP50"],
             "overall mAP50-95": metrics["overall"]["mAP50_95"]}
    for cls, v in metrics["per_class"].items():
        truth["%s mAP50" % cls] = v["mAP50"]

    every_measured = set()
    for group in [metrics["overall"]] + list(metrics["per_class"].values()):
        for val in group.values():
            every_measured.add(round(float(val), 3))
            every_measured.add(round(float(val), 2))

    matched, mismatched = [], []
    for label, val in truth.items():
        if re.search(r"\b%.3f\b|\b%.2f\b" % (val, val), memo) or ("%.3f" % val) in memo:
            matched.append("%s=%.3f" % (label, val))
        else:
            mismatched.append("%s=%.3f not quoted in memo" % (label, val))

    inflated = [n for n in re.findall(r"0\.9\d\d", memo)
                if round(float(n), 3) not in every_measured]
    hedged = bool(re.search(r"should not be believed|inflated|honest|do not|misleading", memo, re.I))

    earned = round(10 * len(matched) / max(len(truth), 1))
    if hedged:
        earned = min(10, earned + 2)
    ev = ["metrics.json overall mAP50=%.4f" % metrics["overall"]["mAP50"],
          "memo quotes: %s" % (", ".join(matched) or "none")]
    if mismatched:
        ev += ["UNVERIFIED: " + m for m in mismatched]
    if inflated:
        ev.append("suspicious unbacked 0.9xx figures: %s" % inflated)
    ev.append("memo qualifies its own headline number: %s" % hedged)
    record("2", "Metric honesty (memo vs metrics.json)", min(earned, 10), 10, ev)


def check_failure_cases():
    memo = read("MEMO.md")
    section = re.split(r"##\s*\d*\.?\s*Five failure cases", memo, flags=re.I)
    body = section[1] if len(section) > 1 else memo
    body = re.split(r"\n##\s", body)[0]
    cases = re.findall(r"^\d+\.\s+\*\*(.+?)\*\*", body, re.M)
    with_cause = len(re.findall(r"\*Cause:\*|root.cause|because", body, re.I))
    with_fix = len(re.findall(r"\*Fix:\*|mitigat", body, re.I))
    quantified = len(re.findall(r"\d+(?:\.\d+)?%|\b\d{2,}\b", body))

    ev = ["found %d numbered cases: %s" % (len(cases), [c[:34] for c in cases]),
          "cause statements: %d | fix statements: %d" % (with_cause, with_fix),
          "quantified figures in section: %d" % quantified]
    earned = 0
    if len(cases) >= 5:
        earned = 9
        if with_cause >= 5:
            earned += 3
        if with_fix >= 5:
            earned += 3
    else:
        ev.append("brief requires FIVE; a model with none is a red flag")
    record("2", "Failure-case analysis", min(earned, 15), 15, ev)


def check_dataset_justification():
    memo = read("MEMO.md")
    signals = {
        "sources enumerated": len(re.findall(r"\|\s*\S+/\S+\s*\|", memo)) >= 3,
        "pivot justified": bool(re.search(r"pivot|cut|dropped", memo, re.I)),
        "split strategy": bool(re.search(r"70\s*/\s*15\s*/\s*15|split", memo, re.I)),
        "leakage controls": bool(re.search(r"leak|dedup|grouped", memo, re.I)),
        "evidence for cuts": bool(re.search(r"\d+%\s|0\.\d{3}", memo)),
    }
    earned = round(15 * sum(signals.values()) / len(signals))
    ev = ["satisfied: %s" % ", ".join(k for k, v in signals.items() if v)]
    absent = [k for k, v in signals.items() if not v]
    if absent:
        ev.append("MISSING: %s" % ", ".join(absent))
    if (ROOT / "dataset" / "split_report.json").exists():
        rep = json.loads(read("dataset/split_report.json"))
        ev.append("split_report.json committed: %s" % rep.get("split_counts"))
    else:
        ev.append("split_report.json NOT committed - class balance unauditable")
    record("2", "Dataset sourcing & justification", earned, 15, ev)


def check_code_quality():
    ev, earned = [], 10
    cpu = read("requirements.txt")
    cuda = read("requirements-cuda.txt")
    if "cu128" in cuda and "cu128" not in cpu:
        ev.append("dependency split correct: cu128 isolated to requirements-cuda.txt")
    else:
        earned -= 3
        ev.append("PROBLEM: CPU/GPU requirement split is not clean")

    broken = []
    for f in py_files():
        src = f.read_text(encoding="utf-8")
        for lit in re.findall(r'["\'](sample_images/[^"\']+)["\']', src):
            if not (ROOT / lit).exists():
                broken.append("%s -> %s (missing)" % (f.name, lit))
    if broken:
        earned -= 2
        ev.append("BROKEN FIXTURE PATHS: %s" % "; ".join(broken))
    else:
        ev.append("all sample_images/ paths referenced in code exist")

    bad = [f.name for f in py_files()
           if not _parses(f)]
    if bad:
        earned -= 5
        ev.append("FILES DO NOT PARSE: %s" % bad)
    record("2", "Code quality", max(earned, 0), 10, ev)


def _parses(f):
    try:
        ast.parse(f.read_text(encoding="utf-8"))
        return True
    except SyntaxError:
        return False


def check_container_static():
    df, dc = read("Dockerfile"), read("docker-compose.yml")
    checks = {
        "slim base image": bool(re.search(r"FROM python:3\.\d+-slim", df)),
        "CPU torch index": "whl/cpu" in df,
        "non-root USER": bool(re.search(r"^USER (?!root)", df, re.M)),
        "HEALTHCHECK": "HEALTHCHECK" in df,
        "compose port map": bool(re.search(r"\d+:8000", dc)),
        "health endpoint in app": "/health" in read("app/main.py"),
    }
    earned = round(10 * sum(checks.values()) / len(checks))
    ev = ["satisfied: %s" % ", ".join(k for k, v in checks.items() if v)]
    absent = [k for k, v in checks.items() if not v]
    if absent:
        ev.append("MISSING: %s" % ", ".join(absent))

    # Fresh-clone viability: compose declaring a gitignored env_file is a
    # documented one-command path that fails for every reviewer.
    if re.search(r"env_file", dc):
        envs = re.findall(r"^\s*-\s*(\S+)$", dc.split("env_file")[1][:120], re.M)
        for e in envs:
            ignored = bool(re.search(r"^\s*%s\s*$" % re.escape(e), read(".gitignore"), re.M))
            tracked = e in git("ls-files").split()
            if ignored and not tracked:
                earned = max(earned - 2, 0)
                ev.append("BLOCKER: compose requires '%s' which is gitignored; "
                          "`docker compose up` fails on a fresh clone" % e)
    record("2", "Bonus: containerisation & ops", earned, 10, ev)


# ---------------------------------------------------------------- OPTIONAL

def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def check_live_api():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = "http://127.0.0.1:%d" % port
    ev, earned = [], 0
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(base + "/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        else:
            record("2", "Live API probe", 0, 8, ["server never became ready"])
            return

        health = json.loads(urllib.request.urlopen(base + "/health", timeout=5).read())
        ev.append("/health model_loaded=%s classes=%s"
                  % (health.get("model_loaded"), health.get("classes")))
        earned += 2 if health.get("model_loaded") else 0
        if not health.get("model_loaded"):
            ev.append("model NOT loaded - hidden-set evaluation would fail here")

        # Malformed payload handling, checked by status code.
        for name, body, ctype, expect in [
            ("text/plain upload", b"not an image", "text/plain", 415),
            ("empty image body", b"", "image/jpeg", 400),
            ("garbage jpeg bytes", b"\x00\x01\x02\x03", "image/jpeg", 400),
        ]:
            boundary = "----auditboundary"
            payload = (
                "--%s\r\nContent-Disposition: form-data; name=\"file\"; "
                "filename=\"x\"\r\nContent-Type: %s\r\n\r\n" % (boundary, ctype)
            ).encode() + body + ("\r\n--%s--\r\n" % boundary).encode()
            req = urllib.request.Request(
                base + "/api/v1/detect", data=payload,
                headers={"Content-Type": "multipart/form-data; boundary=%s" % boundary})
            try:
                urllib.request.urlopen(req, timeout=20)
                got = 200
            except urllib.error.HTTPError as e:
                got = e.code
            except Exception as e:
                got = str(e)
            ok = got == expect
            earned += 2 if ok else 0
            ev.append("[%s] %-20s expected %s got %s"
                      % ("OK" if ok else "XX", name, expect, got))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    record("2", "Live API probe", min(earned, 8), 8, ev)


def check_docker_live():
    if not shutil.which("docker"):
        record("2", "Docker build & run", 0, 0, ["docker not on PATH - skipped"])
        return
    tag = "rap-audit:selftest"
    ev = []
    build = subprocess.run(["docker", "build", "-t", tag, str(ROOT)],
                           capture_output=True, text=True, timeout=2400)
    if build.returncode != 0:
        record("2", "Docker build & run", 0, 5,
               ["BUILD FAILED", build.stderr.strip().splitlines()[-1][:120]])
        return
    ev.append("image built")
    port = _free_port()
    run = subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", "rap-audit-run",
         "-p", "%d:8000" % port,
         "-v", "%s:/app/weights:ro" % str(ROOT / "weights"), tag],
        capture_output=True, text=True, timeout=120)
    if run.returncode != 0:
        record("2", "Docker build & run", 2, 5, ev + ["RUN FAILED: " + run.stderr[:120]])
        return
    try:
        ok = False
        for _ in range(60):
            try:
                h = json.loads(urllib.request.urlopen(
                    "http://127.0.0.1:%d/health" % port, timeout=2).read())
                ev.append("container /health model_loaded=%s" % h.get("model_loaded"))
                ok = bool(h.get("model_loaded"))
                break
            except Exception:
                time.sleep(2)
        else:
            ev.append("container never answered /health")
        record("2", "Docker build & run", 5 if ok else 3, 5, ev)
    finally:
        subprocess.run(["docker", "stop", "rap-audit-run"], capture_output=True, timeout=60)
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True, timeout=120)


# ---------------------------------------------------------------- MAIN

def main():
    ap = argparse.ArgumentParser(description="Adversarial self-audit for the RAP submission")
    ap.add_argument("--live", action="store_true", help="boot the API and probe endpoints")
    ap.add_argument("--docker", action="store_true", help="build and run the container")
    ap.add_argument("--min-score", type=float, default=95.0)
    args = ap.parse_args()

    print("%s%s RAP SUBMISSION AUDIT %s" % (BOLD, "=" * 26, RESET))
    print("%srepo: %s%s\n" % (DIM, ROOT, RESET))

    print("%sPHASE 1 - HARD CONSTRAINTS (instant zero)%s" % (BOLD, RESET))
    check_frameworks()
    check_taxonomy()
    check_reproducibility()

    print("\n%sPHASE 2 - RUBRIC%s" % (BOLD, RESET))
    check_weights_deliverable()
    check_api_robustness()
    check_dataset_justification()
    check_metric_honesty()
    check_failure_cases()
    check_partb_behaviour()
    check_code_quality()
    check_container_static()

    if args.live:
        print("\n%sLIVE PROBES%s" % (BOLD, RESET))
        check_live_api()
    if args.docker:
        print("\n%sCONTAINER PROBES%s" % (BOLD, RESET))
        check_docker_live()

    scored = [r for r in results if r[3] > 0]
    earned = sum(r[2] for r in scored)
    possible = sum(r[3] for r in scored)
    pct = 100.0 * earned / possible if possible else 0.0

    print("\n%s%s%s" % (BOLD, "=" * 60, RESET))
    if hard_failures:
        print("%sHARD CONSTRAINT FAILED: %s%s" % (RED, ", ".join(hard_failures), RESET))
        print("%sSubmission would be disqualified or zeroed on Part A.%s" % (RED, RESET))
    print("%sSCORE: %d/%d = %.1f%%%s"
          % (BOLD, earned, possible, pct, RESET))

    gaps = sorted((r for r in scored if r[2] < r[3]),
                  key=lambda r: r[2] - r[3])
    if gaps:
        print("\n%sPRIORITISED GAPS (largest point loss first)%s" % (BOLD, RESET))
        for phase, name, e, p, _, evidence in gaps:
            print("  %s-%d%s  %s" % (RED, p - e, RESET, name))
            for line in evidence:
                if re.match(r"^(BLOCKER|MISSING|PROBLEM|BROKEN|UNVERIFIED|FILES)", line):
                    print("        %s" % line)

    ok = not hard_failures and pct >= args.min_score
    print("\n%sVERDICT: %s%s" % (GREEN if ok else RED,
                                 "READY TO SUBMIT" if ok else "DO NOT SUBMIT YET", RESET))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
