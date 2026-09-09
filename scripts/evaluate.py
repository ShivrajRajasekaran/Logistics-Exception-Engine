"""
Evaluate a trained RT-DETR checkpoint on a held-out split.

Writes a per-class metrics table to stdout and metrics.json to disk so the
numbers quoted in MEMO.md can be traced back to a specific artifact instead
of being retyped by hand.
"""

import argparse
import json
from pathlib import Path

from ultralytics import RTDETR


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
    # Ultralytics exposes per-class arrays indexed by ap_class_index, not by
    # raw class id - mapping through it avoids silently mislabelling rows.
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
    return p


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
