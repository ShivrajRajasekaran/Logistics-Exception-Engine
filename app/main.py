"""
Logistics Exception Engine - FastAPI application.

Two endpoints:
  POST /api/v1/detect  - Part A. Image in, RT-DETR detections out.
  POST /api/v1/reason  - Part B. Question + package id in, adjudication out.

FRAMEWORK DECLARATION
This module imports fastapi, pydantic, opencv, numpy, and our own two modules.
It does not import LangChain, LlamaIndex, CrewAI, AutoGen, Haystack, or any
other orchestration framework. Part B's control flow is the plain sequence of
function calls in `reason()` below, written by hand and readable top to bottom.
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import detector, ledger, reasoning, store
from app.schemas import DetectResponse, ReasonRequest, ReasonResponse

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("exception-engine.api")

STATIC_DIR = Path(__file__).resolve().parent / "static"

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the checkpoint once at startup.

    A missing checkpoint is logged and tolerated rather than fatal: the service
    still starts, /health reports not-ready, and the endpoints return a clear
    503. A crash-on-boot would tell a reviewer far less than that.
    """
    try:
        detector.load_model()
        log.info("startup complete | model ready")
    except detector.ModelNotLoadedError as exc:
        log.error("startup: model unavailable | %s", exc)
    yield
    log.info("shutdown")


app = FastAPI(
    title="Logistics Exception Engine",
    description="RT-DETR parcel defect detection with a hand-written reasoning layer.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Per-request latency logging, and a last-resort handler so an unexpected
    exception returns JSON instead of an HTML traceback page."""
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed = (time.perf_counter() - start) * 1000
        log.exception("%s %s | unhandled error after %.1f ms",
                      request.method, request.url.path, elapsed)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": "Internal server error."},
        )
    elapsed = (time.perf_counter() - start) * 1000
    log.info("%s %s -> %d | %.1f ms",
             request.method, request.url.path, response.status_code, elapsed)
    return response


def _decode_upload(raw: bytes) -> np.ndarray:
    """Bytes to BGR ndarray, with the failure modes separated out."""
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds %d MB limit." % (MAX_UPLOAD_BYTES // (1024 * 1024)),
        )
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. Supply a valid JPEG, PNG, BMP, or WebP.",
        )
    return image


# Roots /reason is allowed to read from. Anything else is refused before the
# filesystem is touched: the endpoint previously accepted any path, so a caller
# could tell which files existed by comparing a 404 against a 400. That leaks
# the filesystem layout even though cv2 would never have decoded the contents.
READABLE_ROOTS = [
    (Path(__file__).resolve().parent.parent / "sample_images").resolve(),
    (Path(__file__).resolve().parent.parent / "uploads").resolve(),
]


def _resolve_readable(image_path: str) -> Path:
    """Resolve a caller-supplied path, or refuse it.

    Resolution happens first so that `..` segments and symlinks are collapsed
    before the containment test, rather than after.
    """
    try:
        target = Path(image_path).resolve()
    except (OSError, ValueError):
        raise HTTPException(status_code=400, detail="Malformed image path.")
    if not any(target == root or root in target.parents for root in READABLE_ROOTS):
        raise HTTPException(
            status_code=403,
            detail=("Images may only be read from sample_images/ or uploads/. "
                    "Upload one with POST /api/v1/uploads to reason about it."),
        )
    return target


def _load_image_from_path(image_path: str) -> np.ndarray:
    """Read a server-side image for the reasoning endpoint."""
    path = _resolve_readable(image_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found at '%s'." % image_path)
    image = cv2.imread(str(path))
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image at '%s'." % image_path)
    return image


def _require_model() -> None:
    if not detector.is_ready():
        raise HTTPException(
            status_code=503,
            detail=("Detection model is not loaded. Train with scripts/train.py or place "
                    "a checkpoint at the path given by WEIGHTS_PATH."),
        )


class RevalidatingStaticFiles(StaticFiles):
    """StaticFiles that always sends `Cache-Control: no-cache`.

    Starlette sets ETag and Last-Modified but no Cache-Control. With no
    freshness directive a browser falls back to heuristic caching and may reuse
    a stored copy WITHOUT revalidating. Because these assets have stable names
    (`style.css`, `app.js`), that showed up as a stale console after a rebuild:
    the server had the new file and the browser never asked for it.

    `no-cache` does not mean "do not store", it means "revalidate before use".
    Paired with the ETag the normal case is a 304 with an empty body, so this
    costs one conditional request and cannot serve stale UI. Content-hashed
    filenames would allow immutable long-lived caching instead, but that needs
    a build step this project deliberately does not have.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


if (STATIC_DIR / "assets").is_dir():
    # Stylesheet and script live on disk as separate files rather than inlined
    # into index.html, so they are cacheable, diffable and editable on their own.
    app.mount("/assets", RevalidatingStaticFiles(directory=STATIC_DIR / "assets"),
              name="assets")


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def ui():
    """Single-page demo console.

    Served from this app rather than a second container so `docker compose up`
    still brings up the whole system. Returns 404 rather than 500 if the asset
    is missing, since the API must stay usable without it.
    """
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI asset not found.")
    # Revalidate every load: the markup names the asset files, so a stale
    # index.html would keep pointing at whatever it was built against.
    return FileResponse(index, headers={"Cache-Control": "no-cache"})


@app.get("/samples/{name}", include_in_schema=False)
def sample_image(name: str):
    """Serve a bundled sample image so the demo console can show thumbnails.

    Resolves the path and confirms it is inside sample_images/ before serving:
    accepting a bare name from the URL would otherwise let `../` escape the
    directory and read arbitrary files.
    """
    root = (Path(__file__).resolve().parent.parent / "sample_images").resolve()
    target = (root / name).resolve()
    if root not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="No such sample image.")
    return FileResponse(target)


@app.post("/api/v1/uploads", include_in_schema=False)
async def upload_for_reasoning(file: UploadFile = File(...)):
    """Store an uploaded image so /reason can be asked about it.

    /reason takes a server-side path, so before this existed you could run
    detection on your own image but never ask a question about it - the
    dropdown could only ever offer the bundled samples. That was the gap.

    The stored name is generated, never taken from the client: a filename is
    attacker-controlled and is the usual way a write escapes its directory.
    """
    if file.content_type and file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415,
                            detail="Unsupported content type '%s'. Send an image."
                                   % file.content_type)
    raw = await file.read()
    image = _decode_upload(raw)          # reuses the 400/413 checks
    height, width = image.shape[:2]

    root = Path(__file__).resolve().parent.parent / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        suffix = ".jpg"
    stored = root / ("u_%s%s" % (uuid.uuid4().hex[:16], suffix))
    stored.write_bytes(raw)

    log.info("upload | %s -> %s | %dx%d", file.filename, stored.name, width, height)
    return {
        "image_path": "uploads/%s" % stored.name,
        "original_name": file.filename,
        "image_size": [width, height],
    }


@app.get("/api/v1/samples", include_in_schema=False)
def list_samples():
    """Names of the bundled samples, so the UI never hardcodes a filename."""
    root = Path(__file__).resolve().parent.parent / "sample_images"
    if not root.is_dir():
        return {"samples": []}
    return {"samples": sorted(p.name for p in root.iterdir()
                              if p.suffix.lower() in {".jpg", ".jpeg", ".png"})}


@app.get("/api/v1/exceptions", include_in_schema=False)
def recent_exceptions(limit: int = 20):
    """Most recent adjudications, newest first.

    The reasoning layer never writes to the transit ledger; its output goes to
    the store instead. Read-only: nothing here edits or deletes a row, because
    both tables are append-only by convention.
    """
    limit = max(1, min(int(limit), 200))
    st = store.stats()
    return {
        "count": st["adjudications"],
        "ledger_digest": ledger.DIGEST[:16],
        "ledger_intact": ledger.verify(),
        "entries": store.recent_adjudications(limit),
    }


@app.get("/api/v1/detections", include_in_schema=False)
def recent_detections(limit: int = 20):
    """Most recent /detect observations, newest first.

    Separate from adjudications on purpose: a detection is an observation, an
    adjudication is a decision with a guardrail outcome behind it.
    """
    limit = max(1, min(int(limit), 200))
    st = store.stats()
    return {
        "count": st["detections"],
        "images_with_damage": st["images_with_damage"],
        "entries": store.recent_detections(limit),
    }


@app.get("/api/v1/stats", include_in_schema=False)
def store_stats():
    """Counts behind the dashboard summary tiles."""
    return store.stats()


@app.get("/health")
def health():
    """Readiness probe. Also surfaces the thresholds, so a reviewer can see
    what the guardrail is actually configured to at runtime."""
    return {
        "status": "ok",
        "model_loaded": detector.is_ready(),
        "weights_path": detector.WEIGHTS_PATH,
        "classes": list(detector.class_names().values()) if detector.is_ready() else [],
        "critical_classes": sorted(detector.CRITICAL_CLASSES),
        "detection_threshold": detector.CONFIDENCE_THRESHOLD,
        "guardrail_threshold": reasoning.GUARDRAIL_THRESHOLD,
        "ledger_digest": ledger.DIGEST[:16],
        "ledger_intact": ledger.verify(),
        "adjudications_recorded": store.stats()["adjudications"],
        "detections_recorded": store.stats()["detections"],
    }


@app.post("/api/v1/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)):
    """Part A. Single image upload in, detections out."""
    _require_model()

    if file.content_type and file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported content type '%s'. Send an image." % file.content_type,
        )

    image = _decode_upload(await file.read())
    height, width = image.shape[:2]
    detections, elapsed_ms = detector.run_inference(image)

    # Observations were previously discarded the moment the response was sent,
    # so the system could not answer what it had seen. Recording never raises.
    store.record_detection(file.filename, [width, height], elapsed_ms, detections)

    return DetectResponse(
        filename=file.filename,
        image_size=[width, height],
        inference_time_ms=elapsed_ms,
        count=len(detections),
        detections=detections,
    )


@app.post("/api/v1/reason", response_model=ReasonResponse)
async def reason(payload: ReasonRequest):
    """
    Part B. The whole decision layer is the ordered sequence below.

      1. route_intent      - does this question need pixels at all?
      2. run_inference     - only if it does
      3. evaluate_guardrail- halt before the LLM on weak evidence
      4. ledger lookup     - ground truth to reconcile against
      5. synthesize        - one direct LLM call

    Steps 3 and 5 are deliberately adjacent so it is obvious by reading that
    no model is consulted when the guardrail fails.
    """
    response = _adjudicate(payload)
    store.record_adjudication(
        package_id=payload.package_id,
        query=payload.query,
        status=response.status,
        guardrail_passed=response.guardrail_passed,
        requires_vision_model=response.requires_vision_model,
        max_critical_confidence=response.max_critical_confidence,
        detections=[d.model_dump() for d in response.detections],
        ledger_digest=ledger.DIGEST,
        ledger_intact=ledger.verify(),
    )
    return response


def _adjudicate(payload: ReasonRequest) -> ReasonResponse:
    """The decision layer itself. Split out so `reason` has one exit point to
    record, and so this stays readable top to bottom."""
    ledger_record: Optional[dict] = ledger.lookup(payload.package_id)

    route, rationale = reasoning.route_intent(payload.query, payload.image_path)
    log.info("reason | package=%s | route=%s | %s", payload.package_id, route, rationale)

    if route == reasoning.ROUTE_UNSUPPORTED:
        return ReasonResponse(
            package_id=payload.package_id,
            status="UNSUPPORTED_CAPABILITY",
            requires_vision_model=False,
            guardrail_passed=False,
            decision_summary=(
                "Cannot answer from this model. The question %s. The detector is "
                "trained on package and damaged-package only. "
                "Answering from those classes would be a confident wrong answer, "
                "so this is routed to manual inspection instead." % rationale
            ),
            ledger_record=ledger_record,
        )

    if route == reasoning.ROUTE_OUT_OF_SCOPE:
        return ReasonResponse(
            package_id=payload.package_id,
            status="ANSWERED_WITHOUT_VISION",
            requires_vision_model=False,
            guardrail_passed=True,
            decision_summary=(
                "This question does not concern the physical condition of the parcel, so "
                "the detection model was not invoked. Routing rationale: %s. Ask about "
                "label integrity, seal integrity, or parcel condition to trigger inspection."
                % rationale
            ),
            ledger_record=ledger_record,
        )

    if route == reasoning.ROUTE_LEDGER:
        if not ledger_record:
            return ReasonResponse(
                package_id=payload.package_id,
                status="INSUFFICIENT_INFORMATION",
                requires_vision_model=False,
                guardrail_passed=False,
                decision_summary=(
                    "No transit record exists for package '%s', and the question does not "
                    "require image analysis. Nothing can be answered from available data."
                    % payload.package_id
                ),
            )
        summary = reasoning.synthesize(payload.query, [], {}, ledger_record)
        return ReasonResponse(
            package_id=payload.package_id,
            status="ANSWERED_WITHOUT_VISION",
            requires_vision_model=False,
            guardrail_passed=True,
            decision_summary=summary,
            ledger_record=ledger_record,
        )

    _require_model()
    image = _load_image_from_path(payload.image_path)
    detections, elapsed_ms = detector.run_inference(image)
    max_critical = detector.max_critical_confidence(detections)
    log.info("reason | detections=%d | peak_critical=%.3f | %.1f ms",
             len(detections), max_critical, elapsed_ms)

    passed, explanation = reasoning.evaluate_guardrail(detections, max_critical)
    if not passed:
        log.info("reason | GUARDRAIL HALT | no LLM call | %s", explanation)
        return ReasonResponse(
            package_id=payload.package_id,
            status="INSUFFICIENT_INFORMATION",
            requires_vision_model=True,
            guardrail_passed=False,
            max_critical_confidence=max_critical,
            decision_summary=(
                "INSUFFICIENT_INFORMATION. %s Routing this parcel to manual inspection "
                "rather than guessing." % explanation
            ),
            detections=detections,
            ledger_record=ledger_record,
        )

    counts = detector.summarize(detections)
    summary = reasoning.synthesize(payload.query, detections, counts, ledger_record)

    defect_found = max_critical > 0.0
    return ReasonResponse(
        package_id=payload.package_id,
        status="EXCEPTION_FLAGGED" if defect_found else "CLEAR",
        requires_vision_model=True,
        guardrail_passed=True,
        max_critical_confidence=max_critical,
        decision_summary=summary,
        detections=detections,
        ledger_record=ledger_record,
    )
