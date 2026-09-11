"""
Measure whether de-duplicating overlapping detections improves serving quality.

WHY THIS EXISTS
RT-DETR is NMS-free: it relies on one-to-one Hungarian matching during training
rather than suppressing overlaps at inference. When two decoder queries lock
onto the same object, both survive. Measured on the v3 test split, 12.4% of
images emit more predictions than there are ground-truth objects, and the
highest-confidence "false positive" found during failure analysis was a second
box on an already-detected parcel:

    package 0.959  [210.2, 200.7, 271.6, 301.5]
    package 0.575  [210.2, 200.5, 271.7, 301.5]

Those cost precision without costing recall, so a same-class IoU filter should
be close to free. This script measures that rather than assuming it.

Inference runs ONCE and predictions are cached; each threshold is then scored
from cache. Re-running the model per threshold took over ten minutes.

    python scripts/tune_dedup.py
"""

import argparse
import json
from pathlib import Path

from ultralytics import RTDETR

TEST_IMAGES = Path("dataset/test/images")
TEST_LABELS = Path("dataset/test/labels")


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def collect(weights: str, conf: float) -> list:
    """One inference pass. Returns [(ground_truths, predictions)] per image."""
    model = RTDETR(weights)
    names = model.names
    rows = []
    for image_path in sorted(TEST_IMAGES.glob("*")):
        label_path = TEST_LABELS / (image_path.stem + ".txt")
        if not label_path.exists():
            continue
        result = model.predict(str(image_path), conf=conf, verbose=False)[0]
        height, width = result.orig_shape
        truths = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:5])
            truths.append((cid, [(cx - w / 2) * width, (cy - h / 2) * height,
                                 (cx + w / 2) * width, (cy + h / 2) * height]))
        preds = [(int(b.cls.item()), b.xyxy[0].tolist(), float(b.conf.item()))
                 for b in result.boxes]
        rows.append((truths, preds))
    return rows, names


def dedup(preds, threshold):
    """Drop a lower-confidence box that overlaps a kept box of the SAME class.

    Class-aware on purpose: a parcel legitimately overlapping a different-class
    object must survive.
    """
    kept = []
    for det in sorted(preds, key=lambda d: -d[2]):
        if all(det[0] != k[0] or iou(det[1], k[1]) < threshold for k in kept):
            kept.append(det)
    return kept


def score(rows, threshold):
    tp = fp = fn = 0
    for truths, preds in rows:
        if threshold is not None:
            preds = dedup(preds, threshold)
        used = set()
        for cid, box in truths:
            hit = False
            for i, (pcid, pbox, _) in enumerate(preds):
                if i in used or pcid != cid:
                    continue
                if iou(box, pbox) >= 0.5:
                    used.add(i)
                    hit = True
                    break
            tp += hit
            fn += not hit
        fp += len(preds) - len(used)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Tune same-class detection de-duplication")
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="reports/dedup_sweep.json")
    args = ap.parse_args()

    print("running one inference pass over the test split ...")
    rows, _ = collect(args.weights, args.conf)
    print("cached %d images\n" % len(rows))

    settings = [(None, "none (current)"), (0.9, "IoU>0.9"), (0.8, "IoU>0.8"),
                (0.7, "IoU>0.7"), (0.6, "IoU>0.6"), (0.5, "IoU>0.5")]
    results = {}
    print("  %-18s %6s %6s %6s %9s %9s %9s" %
          ("dedup", "TP", "FP", "FN", "precision", "recall", "F1"))
    print("  " + "-" * 68)
    for threshold, label in settings:
        m = score(rows, threshold)
        results[label] = m
        print("  %-18s %6d %6d %6d %9.3f %9.3f %9.3f" %
              (label, m["tp"], m["fp"], m["fn"], m["precision"], m["recall"], m["f1"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"weights": args.weights, "conf": args.conf,
                               "results": results}, indent=2), encoding="utf-8")
    print("\nwritten -> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
