"""
Evaluate a trained RT-DETR checkpoint on a held-out split.

Writes a per-class metrics table to stdout and metrics.json to disk so the
numbers quoted in MEMO.md can be traced back to a specific artifact instead
of being retyped by hand.

`--per-source` additionally reports recall broken down by the source project
each test image came from. Aggregate mAP hides a model that has learned
"which dataset is this" instead of the actual visual class; a large spread
across sources is the symptom. This was the analysis that exposed the shortcut
in the first trained model, so it belongs in the repository rather than in a
one-off script.
"""

import argparse
import collections
import json
from pathlib import Path

from ultralytics import RTDETR


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def per_source_recall(model, split: str, conf: float, names: dict) -> dict:
    """
    Recall per (source project, class) at the operational threshold.

    Aggregate mAP cannot distinguish a model that learned the visual class from
    one that learned which dataset an image came from. If recall is near-perfect
    on one source and near-zero on another, the model is keying on source
    appearance, and the aggregate is meaningless for unseen data.

    Source is read from the filename prefix that prepare_dataset.py writes
    (`<source>_<stem>_<hash>.jpg`), so no extra bookkeeping is needed.

    Uses the operational confidence threshold, not the mAP sweep threshold:
    this answers "what would the deployed service actually catch".
    """
    image_dir = Path("dataset") / split / "images"
    label_dir = Path("dataset") / split / "labels"
    images = sorted(image_dir.glob("*"))
    tally = collections.defaultdict(lambda: collections.Counter())

    for image_path in images:
        source = image_path.name.split("_")[0]
        label_path = label_dir / (image_path.stem + ".txt")
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

        preds = [(int(b.cls.item()), b.xyxy[0].tolist()) for b in result.boxes]
        matched = set()
        for cid, box in truths:
            best, best_i = 0.0, -1
            for i, (pcid, pbox) in enumerate(preds):
                if i in matched or pcid != cid:
                    continue
                overlap = _iou(box, pbox)
                if overlap > best:
                    best, best_i = overlap, i
            key = (source, names[cid])
            if best >= 0.5:
                matched.add(best_i)
                tally[key]["hit"] += 1
            else:
                tally[key]["miss"] += 1
        tally[(source, "__unmatched_predictions__")]["count"] += len(preds) - len(matched)

    report = {}
    for (source, cls), counts in sorted(tally.items()):
        if cls == "__unmatched_predictions__":
            report.setdefault(source, {})["unmatched_predictions"] = counts["count"]
            continue
        total = counts["hit"] + counts["miss"]
        report.setdefault(source, {})[cls] = {
            "instances": total,
            "hits": counts["hit"],
            "recall": round(counts["hit"] / total, 4) if total else None,
        }
    return report


def evaluate(args: argparse.Namespace) -> dict:
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found. Train first, or download per README.")

    model = RTDETR(str(weights))
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=True,
        save_json=True,
    )

    names = model.names
    per_class = {}
    for row, class_id in enumerate(metrics.box.ap_class_index):
        per_class[names[int(class_id)]] = {
            "precision": round(float(metrics.box.p[row]), 4),
            "recall": round(float(metrics.box.r[row]), 4),
            "mAP50": round(float(metrics.box.ap50[row]), 4),
            "mAP50_95": round(float(metrics.box.ap[row]), 4),
        }

    summary = {
        "weights": str(weights),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "overall": {
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4),
        },
        "per_class": per_class,
        "speed_ms": {k: round(float(v), 2) for k, v in metrics.speed.items()},
    }

    if args.per_source:
        summary["per_source"] = per_source_recall(
            model, args.split, args.conf_operational, names)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"\n{'class':<22}{'P':>8}{'R':>8}{'mAP50':>9}{'mAP50-95':>11}")
    print("-" * 58)
    for name, m in per_class.items():
        print(f"{name:<22}{m['precision']:>8.3f}{m['recall']:>8.3f}"
              f"{m['mAP50']:>9.3f}{m['mAP50_95']:>11.3f}")
    o = summary["overall"]
    print("-" * 58)
    print(f"{'ALL':<22}{o['precision']:>8.3f}{o['recall']:>8.3f}"
          f"{o['mAP50']:>9.3f}{o['mAP50_95']:>11.3f}")
    if args.per_source:
        print(f"\nper-source recall @ conf={args.conf_operational} "
              f"(spread across sources is the shortcut tell)")
        print(f"{'source':<26}{'class':<18}{'inst':>6}{'recall':>9}{'unmatched':>11}")
        print("-" * 70)
        spread = {}
        for source, classes in summary["per_source"].items():
            unmatched = classes.get("unmatched_predictions", 0)
            first = True
            for cls, m in classes.items():
                if cls == "unmatched_predictions":
                    continue
                spread.setdefault(cls, []).append(m["recall"] or 0.0)
                print(f"{source if first else '':<26}{cls:<18}{m['instances']:>6}"
                      f"{(m['recall'] or 0):>9.3f}{unmatched if first else '':>11}")
                first = False
        print("-" * 70)
        for cls, values in spread.items():
            if len(values) > 1:
                print(f"{cls:<26}recall spread across sources: "
                      f"{min(values):.3f} to {max(values):.3f}  "
                      f"(range {max(values) - min(values):.3f})")

    print(f"\n[eval] metrics written -> {out}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate RT-DETR on a held-out split")
    p.add_argument("--weights", default="weights/best.pt")
    p.add_argument("--data", default="dataset/data.yaml")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--conf", type=float, default=0.001, help="low conf is correct for mAP")
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", default="0")
    p.add_argument("--out", default="runs/eval/metrics.json")
    p.add_argument("--per-source", dest="per_source", action="store_true",
                   help="break recall down by source project; a large spread means "
                        "the model keyed on source appearance rather than the class")
    p.add_argument("--conf-operational", dest="conf_operational", type=float, default=0.25,
                   help="threshold for the per-source pass; matches the serving "
                        "default rather than the mAP sweep threshold")
    return p


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
