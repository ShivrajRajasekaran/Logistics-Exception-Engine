"""
RT-DETR inference wrapper.

The model is loaded once at process start and reused; reloading per request
would dominate latency. Both endpoints call run_inference(), so Part B is
genuinely calling the Part A model rather than a second copy of it.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

log = logging.getLogger("exception-engine.detector")

WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "weights/best.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))

# RT-DETR is NMS-free: it relies on one-to-one Hungarian matching during
# training rather than suppressing overlaps at inference. When two decoder
# queries lock onto the same parcel, both survive. Measured on the v3 test
# split, 12.4% of images emitted more boxes than there were objects, e.g.
#     package 0.959  [210.2, 200.7, 271.6, 301.5]
#     package 0.575  [210.2, 200.5, 271.7, 301.5]
# Suppressing the weaker of two heavily-overlapping SAME-class boxes cost
# nothing and removed a fifth of all false positives (scripts/tune_dedup.py):
#     none      FP 420  precision 0.638  recall 0.749
#     IoU>0.7   FP 330  precision 0.692  recall 0.749   <- chosen
#     IoU>0.5   FP 321  precision 0.697  recall 0.748   (loses a true positive)
# 0.7 is the conservative pick: it only removes near-identical boxes, which
# matters on crowded scenes where parcels legitimately touch.
DEDUP_IOU = float(os.getenv("DEDUP_IOU", "0.7"))

CRITICAL_CLASSES = {"damaged-package"}

CONTEXT_CLASSES = {"package"}

ANCHOR_CLASS = "package"

_model = None
_class_names: Dict[int, str] = {}


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is requested but no checkpoint is available."""


def load_model(weights: str = None):
    """Idempotent loader. Called on startup; safe to call again."""
    global _model, _class_names
    if _model is not None:
        return _model

    path = Path(weights or WEIGHTS_PATH)
    if not path.exists():
        raise ModelNotLoadedError(
            f"Checkpoint not found at '{path}'. Train with scripts/train.py or "
            "fetch the released weights (see README) before serving."
        )

    from ultralytics import RTDETR

    _model = RTDETR(str(path))
    _class_names = dict(_model.names)
    log.info("RT-DETR loaded from %s | classes=%s", path, list(_class_names.values()))
    return _model


def is_ready() -> bool:
    return _model is not None


def class_names() -> Dict[int, str]:
    return dict(_class_names)


def _iou(a: List[float], b: List[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def deduplicate(detections: List[dict], iou_threshold: float = None) -> List[dict]:
    """Drop the weaker of two heavily-overlapping detections of the SAME class.

    Class-aware deliberately: a parcel overlapping a different-class object is a
    real observation and must survive. Expects `detections` sorted by confidence
    descending, which run_inference guarantees.
    """
    threshold = DEDUP_IOU if iou_threshold is None else iou_threshold
    if threshold >= 1.0:
        return detections
    kept: List[dict] = []
    for det in detections:
        if all(det["label"] != k["label"] or _iou(det["bbox"], k["bbox"]) < threshold
               for k in kept):
            kept.append(det)
    return kept


def run_inference(image: np.ndarray, conf: float = None) -> Tuple[List[dict], float]:
    """Run RT-DETR on one BGR/RGB ndarray. Returns (detections, latency_ms)."""
    if _model is None:
        raise ModelNotLoadedError("Model is not loaded. Call load_model() first.")

    threshold = CONFIDENCE_THRESHOLD if conf is None else conf

    start = time.perf_counter()
    results = _model.predict(source=image, conf=threshold, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    detections: List[dict] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            class_id = int(box.cls.item())
            detections.append({
                "label": _class_names.get(class_id, f"class_{class_id}"),
                "confidence": round(float(box.conf.item()), 4),
                "bbox": [round(float(v), 2) for v in box.xyxy[0].tolist()],
            })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    detections = deduplicate(detections)
    return detections, round(elapsed_ms, 2)


def max_critical_confidence(detections: List[dict]) -> float:
    """Highest confidence among defect classes only. 0.0 when none present."""
    scores = [d["confidence"] for d in detections if d["label"] in CRITICAL_CLASSES]
    return max(scores) if scores else 0.0


def summarize(detections: List[dict]) -> Dict[str, int]:
    """Class -> count. This is the structured object the LLM reasons over."""
    counts: Dict[str, int] = {}
    for d in detections:
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    return counts
