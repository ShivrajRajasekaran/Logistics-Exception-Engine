"""
Measure how much of the label a model could get for free, without looking at
the parcel.

WHY THIS EXISTS
The first trained model reached 0.601 mAP50 with a `package` recall of exactly
1.000, and turned out to be a source classifier. Aggregate mAP cannot detect
that. These diagnostics can, and they run on the DATASET alone - no model, no
GPU - so they can gate a build before any training time is spent.

Three shortcuts are measured, each as "accuracy of a rule that ignores the
parcel entirely":

  1. source-only      guess each source's majority class
  2. resolution-only  guess each image resolution's majority class
  3. box geometry     ratio of median box area between classes

A high number means the class is predictable from provenance rather than from
visual condition, and any headline metric is inflated by exactly that much.

    python scripts/diagnose_shortcuts.py
    python scripts/diagnose_shortcuts.py --out reports/shortcuts_v2.json
    python scripts/diagnose_shortcuts.py --compare reports/shortcuts_v1.json

Exit code 0 always: this reports, it does not gate. `prepare_dataset.py` gates.
"""

import argparse
import collections
import json
import statistics
from pathlib import Path

import yaml
from PIL import Image

DATASET = Path("dataset")
SPLITS = ("train", "val", "test")


def load_class_names() -> dict:
    spec = yaml.safe_load((DATASET / "data.yaml").read_text(encoding="utf-8"))
    names = spec.get("names", {})
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return dict(enumerate(names))


def scan(names: dict) -> dict:
    """One pass over the built dataset collecting everything the rules need."""
    rows = []
    for split in SPLITS:
        image_dir, label_dir = DATASET / split / "images", DATASET / split / "labels"
        if not label_dir.exists():
            continue
        for label_path in label_dir.glob("*.txt"):
            source = label_path.name.split("_")[0]
            classes, areas = [], []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                classes.append(names[int(parts[0])])
                areas.append(float(parts[3]) * float(parts[4]))

            size = None
            matches = list(image_dir.glob(label_path.stem + ".*"))
            if matches:
                try:
                    size = Image.open(matches[0]).size
                except Exception:
                    size = None
            rows.append({"split": split, "source": source, "size": size,
                         "classes": classes, "areas": areas})
    return rows


def majority_rule(rows, key_fn, label_fn):
    """Accuracy of: group by key, always answer that group's majority label.

    This is the cleanest way to express "how much does knowing only <key> tell
    you about the class". It needs no training and cannot overfit.
    """
    groups = collections.defaultdict(collections.Counter)
    for row in rows:
        key, label = key_fn(row), label_fn(row)
        if key is None or label is None:
            continue
        groups[key][label] += 1
    total = sum(sum(c.values()) for c in groups.values())
    correct = sum(max(c.values()) for c in groups.values())
    return (correct / total if total else 0.0), total, groups


def single_class(row):
    unique = set(row["classes"])
    return unique.pop() if len(unique) == 1 else None


def diagnose(names: dict) -> dict:
    rows = scan(names)
    report = {"images_scanned": len(rows)}

    # --- baseline: ignore everything -------------------------------------
    labels = collections.Counter(c for r in rows for c in r["classes"])
    baseline = max(labels.values()) / sum(labels.values()) if labels else 0.0
    report["class_instances"] = dict(labels)
    report["majority_class_baseline"] = round(baseline, 4)

    # --- 1. source-only ---------------------------------------------------
    acc, n, groups = majority_rule(rows, lambda r: r["source"], single_class)
    report["source_only"] = {
        "accuracy": round(acc, 4), "images": n,
        "lift_over_baseline": round(acc - baseline, 4),
        "per_source": {s: dict(c) for s, c in sorted(groups.items())},
    }

    # --- 2. resolution-only ----------------------------------------------
    acc, n, groups = majority_rule(rows, lambda r: r["size"], single_class)
    report["resolution_only"] = {
        "accuracy": round(acc, 4), "images": n,
        "lift_over_baseline": round(acc - baseline, 4),
        "per_resolution": {("%dx%d" % k): dict(c) for k, c in sorted(groups.items())},
    }

    # --- 3. box geometry --------------------------------------------------
    by_class = collections.defaultdict(list)
    for row in rows:
        for cls, area in zip(row["classes"], row["areas"]):
            by_class[cls].append(area)
    medians = {c: round(statistics.median(v), 4) for c, v in by_class.items() if v}
    ratio = None
    if len(medians) == 2:
        lo, hi = sorted(medians.values())
        ratio = round(hi / lo, 2) if lo else None
    report["box_geometry"] = {"median_area_by_class": medians, "ratio": ratio}

    return report


def show(report: dict, previous: dict = None) -> None:
    def delta(path_now, path_before):
        if not previous:
            return ""
        try:
            before = previous
            for key in path_before:
                before = before[key]
        except (KeyError, TypeError):
            return ""
        change = path_now - before
        arrow = "better" if change < 0 else ("worse" if change > 0 else "same")
        return "   was %.4f  (%+.4f, %s)" % (before, change, arrow)

    print("=" * 74)
    print("SHORTCUT DIAGNOSTICS  -  how much label is free without seeing the parcel")
    print("=" * 74)
    print("images scanned            : %d" % report["images_scanned"])
    print("class instances           : %s" % report["class_instances"])
    print("majority-class baseline   : %.4f" % report["majority_class_baseline"])
    print()

    s = report["source_only"]
    print("1. SOURCE-ONLY RULE       : %.4f%s" % (
        s["accuracy"], delta(s["accuracy"], ["source_only", "accuracy"])))
    print("   lift over baseline     : %+.4f" % s["lift_over_baseline"])
    for src, counts in s["per_source"].items():
        print("      %-26s %s" % (src, counts))
    print()

    r = report["resolution_only"]
    print("2. RESOLUTION-ONLY RULE   : %.4f%s" % (
        r["accuracy"], delta(r["accuracy"], ["resolution_only", "accuracy"])))
    print("   lift over baseline     : %+.4f" % r["lift_over_baseline"])
    for res, counts in r["per_resolution"].items():
        print("      %-26s %s" % (res, counts))
    print()

    g = report["box_geometry"]
    print("3. BOX GEOMETRY")
    print("   median area by class   : %s" % g["median_area_by_class"])
    if g["ratio"]:
        was = ""
        if previous and previous.get("box_geometry", {}).get("ratio"):
            was = "   was %.2fx" % previous["box_geometry"]["ratio"]
        print("   separation ratio       : %.2fx%s" % (g["ratio"], was))
    print()
    print("Read these as ceilings on how much of the headline metric is free.")
    print("A value near the baseline means the class must be learned visually.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure dataset shortcut leakage")
    ap.add_argument("--out", default="reports/shortcuts.json")
    ap.add_argument("--compare", help="a previous report to diff against")
    args = ap.parse_args()

    previous = None
    if args.compare and Path(args.compare).exists():
        previous = json.loads(Path(args.compare).read_text(encoding="utf-8"))

    report = diagnose(load_class_names())
    show(report, previous)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwritten -> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
