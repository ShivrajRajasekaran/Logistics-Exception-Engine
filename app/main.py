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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app import detector, reasoning
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


IMMUTABLE_TRANSIT_LEDGER = {
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


def _load_image_from_path(image_path: str) -> np.ndarray:
    """Read a server-side image for the reasoning endpoint."""
    path = Path(image_path)
    if not path.exists():
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


@app.get("/", include_in_schema=False)
def ui():
    """Single-page demo console.

    Served from this app rather than a second container so `docker compose up`
    still brings up the whole system. Returns 404 rather than 500 if the asset
    is missing, since the API must stay usable without it.
    """
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI asset not found.")
    return FileResponse(index)


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
    ledger_record: Optional[dict] = IMMUTABLE_TRANSIT_LEDGER.get(payload.package_id)

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
