"""
Build the unified two-class logistics dataset from Roboflow Universe exports.

WHY THIS SCRIPT EXISTS
No single public dataset covers both intact and damaged parcels. Damage
datasets annotate only damaged parcels; package datasets annotate only intact
ones. We merge six sources, each with its own label vocabulary, into one
contiguous two-class scheme. That merge is the part of
the pipeline most likely to silently corrupt a dataset, so every decision here
is explicit and logged: which source class became which target class, and which
were deliberately discarded.

USAGE
  # Phase 1 - see what the sources actually contain before mapping anything:
  python scripts/prepare_dataset.py --inspect

  # Phase 2 - build the merged dataset:
  python scripts/prepare_dataset.py

Requires ROBOFLOW_API_KEY in .env (free account, universe.roboflow.com).
"""

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv

load_dotenv()

SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Unified target scheme. Indices must match dataset/data.yaml.
#
# Data-driven, not convenient. Three classes from earlier designs were cut,
# each for a measured reason (full narrative in MEMO.md):
#
#   `person`           - no downloadable source carries person boxes alongside
#                        parcels.
#   `compromised-seal` - 35 instances across every source combined. A class
#                        that size is memorised, not learned. Folded into
#                        `damaged-package`, which triggers the same operational
#                        path: divert to the exception bay.
#   `printed-label`    - its one real source was food-packaging expiry codes,
#                        a different domain from logistics routing slips. In a
#                        3-class run it reached 0.053 mAP50 while dragging the
#                        whole model to 0.243.
#
# Both surviving classes are outside COCO, which has backpack, handbag and
# suitcase but no box, package, parcel, or carton category.
TARGET_CLASSES = {
    "package": 0,
    "damaged-package": 1,
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# Source registry.
#
# `remap` maps a SOURCE class name (lowercased, as it appears in that project's
# data.yaml) to one of our target class names. A source class absent from this
# dict is DROPPED, and every drop is counted and reported. Do not add a
# catch-all: silently absorbing unknown classes is how label sets rot.
#
# Run --inspect first. Roboflow projects rename classes between versions, so
# these keys must be confirmed against the actual export, not assumed.
# ---------------------------------------------------------------------------
SOURCES = [
    {
        # Largest damage source. damage_A/B/C are SEVERITY tiers of the same
        # defect, not distinct defect types, so all three collapse to one class.
        # Keeping them separate would train three classes to reproduce a
        # judgement call that the source annotators applied inconsistently.
        "name": "parcel-box-damage",
        "workspace": "project-hffml",
        "project": "parcel-box-damage-classification",
        "version": 1,
        "remap": {
            "damage_a": "damaged-package",
            "damage_b": "damaged-package",
            "damage_c": "damaged-package",
        },
    },
    {
        # Same severity scheme, different institution. Useful precisely because
        # it is a different capture setup: it stops the model from learning one
        # lab's lighting as a proxy for damage.
        "name": "moratuwa-parcel-damage",
        "workspace": "university-of-moratuwa-ztkqd",
        "project": "parcel-damage-detection",
        "version": 2,
        "remap": {
            "damage_a": "damaged-package",
            "damage_b": "damaged-package",
            "damage_c": "damaged-package",
        },
    },
    {
        # Primary source of INTACT parcels, and the class counterweight to the
        # damage sources above.
        "name": "haw-packages",
        "workspace": "haw-odap3",
        "project": "packages-iw8aw",
        "version": 3,
        "remap": {
            "package": "package",
        },
    },
    {
        "name": "packages2-damaged",
        "workspace": "damaged-package",
        "project": "packages2-8qxzc",
        "version": 1,
        "remap": {
            "damaged": "damaged-package",
            "package2": "package",
        },
    },
    {
        # "Open box", "wet Package" and "Package with hole" all fold into
        # damaged-package: operationally they trigger the same exception path.
        # "Invoice" is dropped - it is a document, not a parcel condition.
        "name": "biradar-damage",
        "workspace": "bhagyashri-biradar",
        "project": "damage-package-detection",
        "version": 7,
        "remap": {
            "damagepackage": "damaged-package",
            "package with hole": "damaged-package",
            "wet package": "damaged-package",
            "open box": "damaged-package",
            "box": "package",
        },
    },
    {
        # "label" and "paint" are dropped: `printed-label` was cut as a class
        # (see the pivot note in MEMO.md), and "paint" has 3 instances.
        "name": "box-damage-open",
        "workspace": "box-prfzk",
        "project": "box-a2mjf",
        "version": 3,
        "remap": {
            "damage": "damaged-package",
            "open": "damaged-package",
        },
    },
]

# DROPPED SOURCES, and why. Kept here rather than deleted, because the reason
# they were removed is the most useful thing this file records.
#
# object-detection-5pf5v/packaging-defect-detection (1806 images)
#   Supplied 1592 `package` and all 1502 `printed-label` instances. It is FOOD
#   packaging with printed expiry-date codes, not logistics parcels. In the
#   first 3-class run it was 36% of training data and reached 17% recall,
#   while the logistics source reached 99%. The model was being asked to learn
#   one label class spanning expiry codes and routing slips. Removing it also
#   removed the `printed-label` class.
#
# mohamed-traore-2ekkp/boxes-on-a-conveyer-belt (240 images)
#   All 240 images are frames of a SINGLE candy-factory video clip, so they
#   carry roughly one clip's worth of information, and they are food packaging
#   for the same reason as above. Correctly grouped into one split by the
#   sequence grouping below, which is what exposed how little they add.

RAW_DIR = Path("dataset/_raw")
OUT_DIR = Path("dataset")


def log(msg: str) -> None:
    print(msg, flush=True)


def download_sources(api_key: str) -> List[Tuple[dict, Path]]:
    """Pull each Roboflow project in YOLO format. Ultralytics RT-DETR consumes
    the same normalized `class cx cy w h` layout as YOLO, so no conversion is
    needed beyond class remapping."""
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for src in SOURCES:
        target = RAW_DIR / src["name"]
        if target.exists() and any(target.rglob("*.txt")):
            log("[download] %s already present, skipping" % src["name"])
            downloaded.append((src, target))
            continue
        log("[download] %s/%s v%d ..." % (src["workspace"], src["project"], src["version"]))
        try:
            project = rf.workspace(src["workspace"]).project(src["project"])
            project.version(src["version"]).download("yolov8", location=str(target))
            downloaded.append((src, target))
        except Exception as exc:
            log("[download] FAILED for %s: %s" % (src["name"], exc))
            log("           Check the slug and version at "
                "https://universe.roboflow.com/%s/%s" % (src["workspace"], src["project"]))
    return downloaded


def read_source_classes(root: Path) -> List[str]:
    """Class names in export order, from the project's own data.yaml."""
    for candidate in list(root.glob("data.yaml")) + list(root.rglob("data.yaml")):
        spec = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        names = spec.get("names", [])
        if isinstance(names, dict):
            return [names[k] for k in sorted(names)]
        return list(names)
    return []


def inspect(downloaded: List[Tuple[dict, Path]]) -> None:
    """Report each source's real class vocabulary against our remap, so
    unmapped classes are visible BEFORE a training run consumes them."""
    log("\n" + "=" * 72)
    log("SOURCE CLASS INSPECTION")
    log("=" * 72)
    for src, root in downloaded:
        names = read_source_classes(root)
        log("\n%s  (%d source classes)" % (src["name"], len(names)))
        for idx, name in enumerate(names):
            mapped = src["remap"].get(name.lower().strip())
            marker = "-> %s" % mapped if mapped else "-> DROPPED (no mapping)"
            log("   [%d] %-32s %s" % (idx, name, marker))
        unused = set(src["remap"]) - {n.lower().strip() for n in names}
        if unused:
            log("   WARNING: remap keys matching nothing in this export: %s" % sorted(unused))
    log("\nFix any unintended DROPPED lines in SOURCES before building.\n")


def group_key(stem: str) -> str:
    """
    Group near-duplicate frames so they cannot straddle splits.

    Roboflow exports append augmentation and hash suffixes, e.g.
    `IMG_0412_jpg.rf.9c1a...`. Frames from one capture burst share the prefix
    before that suffix. Splitting on the raw filename would put augmented
    copies of the same photo in both train and test, inflating every metric.
    """
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", stem, flags=re.I)
    stem = re.sub(r"_(jpg|jpeg|png|bmp|webp)$", "", stem, flags=re.I)
    stem = re.sub(r"[-_](aug|augmented)?\d{1,3}$", "", stem, flags=re.I)
    return stem.lower()


def collect_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """Find (image, label) pairs anywhere under an export."""
    pairs = []
    for label_path in root.rglob("labels/*.txt"):
        image_dir = label_path.parent.parent / "images"
        for suffix in IMAGE_SUFFIXES:
            image_path = image_dir / (label_path.stem + suffix)
            if image_path.exists():
                pairs.append((image_path, label_path))
                break
    return pairs


def remap_label_file(label_path: Path, source_names: List[str],
                     remap: Dict[str, str], stats: Counter, drops: Counter) -> Optional[str]:
    """
    Rewrite one label file into target indices.

    Returns the new file content, or None if nothing survived. An image whose
    every annotation was dropped becomes a background image; we keep a bounded
    number of those later rather than discarding them outright, because a
    detector trained without negatives over-predicts.
    """
    lines = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            source_idx = int(float(parts[0]))
        except ValueError:
            continue
        if source_idx >= len(source_names):
            drops["index-out-of-range"] += 1
            continue

        source_name = source_names[source_idx].lower().strip()
        target_name = remap.get(source_name)
        if target_name is None:
            drops[source_name] += 1
            continue

        coords = parts[1:5]
        # Guard against malformed exports: coordinates must be normalized.
        try:
            values = [float(c) for c in coords]
        except ValueError:
            drops["unparseable-coords"] += 1
            continue
        if any(v < 0.0 or v > 1.0 for v in values):
            drops["out-of-bounds-coords"] += 1
            continue

        stats[target_name] += 1
        lines.append("%d %s" % (TARGET_CLASSES[target_name], " ".join(coords)))

    return "\n".join(lines) if lines else None


def build(downloaded: List[Tuple[dict, Path]], max_background_ratio: float) -> dict:
    """Merge, remap, split, and write the unified dataset."""
    random.seed(SEED)

    staged: List[Tuple[Path, str, str]] = []   # (image_path, label_text, group)
    backgrounds: List[Tuple[Path, str]] = []
    class_stats = Counter()
    drop_stats = Counter()
    per_source = defaultdict(int)
    seen_hashes = set()

    for src, root in downloaded:
        source_names = read_source_classes(root)
        if not source_names:
            log("[build] %s: no data.yaml found, skipping" % src["name"])
            continue

        for image_path, label_path in collect_pairs(root):
            # Content hash dedupe: the same photo appears across projects.
            digest = hashlib.md5(image_path.read_bytes()).hexdigest()
            if digest in seen_hashes:
                drop_stats["duplicate-image"] += 1
                continue
            seen_hashes.add(digest)

            content = remap_label_file(label_path, source_names, src["remap"],
                                       class_stats, drop_stats)
            group = "%s::%s" % (src["name"], group_key(image_path.stem))
            if content is None:
                backgrounds.append((image_path, group))
            else:
                staged.append((image_path, content, group))
                per_source[src["name"]] += 1

    if not staged:
        log("\n[build] No usable annotations produced. Run --inspect and fix SOURCES.")
        return {}

    # Keep a bounded number of background images so the model learns what an
    # undamaged scene looks like without drowning in empty frames.
    cap = int(len(staged) * max_background_ratio)
    random.shuffle(backgrounds)
    for image_path, group in backgrounds[:cap]:
        staged.append((image_path, "", group))
    log("[build] kept %d background images (cap %d of %d available)"
        % (min(cap, len(backgrounds)), cap, len(backgrounds)))

    # --- Grouped, source-stratified split ---------------------------------
    # Whole groups move together so augmented copies of one photo cannot
    # straddle splits.
    #
    # The split is stratified PER SOURCE rather than run over one shuffled
    # pool. Assigning from a single pool produced 70/18.6/11.4 with wildly
    # different class mixes per split - validation held 1090 `package`
    # instances against 318 in test - because the sources differ in size and
    # in what they annotate. Splitting each source 70/15/15 independently
    # keeps every domain proportionally represented in all three splits, which
    # is what makes val and test comparable to each other at all.
    groups = defaultdict(list)
    for item in staged:
        groups[item[2]].append(item)

    by_source = defaultdict(list)
    for name in groups:
        by_source[name.split("::")[0]].append(name)

    total = len(staged)
    assignments: Dict[str, str] = {}

    for source_name in sorted(by_source):
        names = sorted(by_source[source_name])
        random.shuffle(names)
        source_total = sum(len(groups[n]) for n in names)

        # Deficit-greedy packing: walk groups largest-first and drop each into
        # whichever split is currently furthest below its target share.
        #
        # The obvious alternative - walk groups in order and switch splits once
        # a running fraction crosses 0.70 / 0.85 - overshoots badly when a
        # source has a few large groups, because the group that crosses the
        # threshold lands entirely on the wrong side of it. That produced
        # 75/14/11 instead of 70/15/15. Largest-first placement means the
        # biggest, most disruptive groups get placed while all three splits
        # still have room to absorb them.
        names.sort(key=lambda n: len(groups[n]), reverse=True)
        assigned = {"train": 0, "val": 0, "test": 0}
        for name in names:
            deficits = {
                split: SPLIT_RATIOS[split] - (assigned[split] / source_total if source_total else 0)
                for split in SPLIT_RATIOS
            }
            split = max(deficits, key=deficits.get)
            assignments[name] = split
            assigned[split] += len(groups[name])

    # --- Write ------------------------------------------------------------
    for split in SPLIT_RATIOS:
        for sub in ("images", "labels"):
            path = OUT_DIR / split / sub
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)

    split_counts = Counter()
    split_class_counts = defaultdict(Counter)
    written_stems = defaultdict(set)

    for image_path, content, group in staged:
        split = assignments[group]
        stem = "%s_%s" % (group.split("::")[0], image_path.stem)
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem)[:110]

        # Truncating to a fixed length collides: several Roboflow exports use
        # long names that are identical in their first 110 characters, and the
        # copy below would silently overwrite the earlier file. That cost 239
        # training images before it was caught by comparing the split report
        # against the file count on disk. The path digest makes the stem unique.
        stem = "%s_%s" % (stem, hashlib.md5(str(image_path).encode()).hexdigest()[:8])

        shutil.copy2(image_path, OUT_DIR / split / "images" / (stem + image_path.suffix))
        (OUT_DIR / split / "labels" / (stem + ".txt")).write_text(content, encoding="utf-8")

        split_counts[split] += 1
        written_stems[split].add(stem)
        for line in content.splitlines():
            if line:
                idx = int(line.split()[0])
                name = [k for k, v in TARGET_CLASSES.items() if v == idx][0]
                split_class_counts[split][name] += 1

    # --- Leakage assertion -------------------------------------------------
    overlap = ((written_stems["train"] & written_stems["test"])
               | (written_stems["train"] & written_stems["val"])
               | (written_stems["val"] & written_stems["test"]))
    if overlap:
        raise RuntimeError("Split leakage: %d filenames appear in multiple splits" % len(overlap))

    # --- Write-completeness assertion --------------------------------------
    # Every staged image must exist on disk. Without this check a filename
    # collision silently drops images: the report claims one count, the trainer
    # sees another, and nothing complains. That happened once here.
    for split in SPLIT_RATIOS:
        on_disk = len(list((OUT_DIR / split / "images").iterdir()))
        if on_disk != split_counts[split]:
            raise RuntimeError(
                "Write mismatch in '%s': staged %d images but %d are on disk. "
                "Output filenames are colliding."
                % (split, split_counts[split], on_disk))

    # --- Invalidate Ultralytics label caches -------------------------------
    # Ultralytics caches its label scan per split directory. A cache left over
    # from a previous build makes the next training run silently consume the
    # OLD split while the new files sit unused.
    for cache in OUT_DIR.rglob("*.cache"):
        cache.unlink()
        log("[build] removed stale label cache %s" % cache)

    report = {
        "seed": SEED,
        "total_images": total,
        "split_counts": dict(split_counts),
        "class_instances_total": dict(class_stats),
        "class_instances_per_split": {k: dict(v) for k, v in split_class_counts.items()},
        "images_per_source": dict(per_source),
        "dropped_annotations": dict(drop_stats),
        "groups": len(groups),
        "split_ratios_requested": SPLIT_RATIOS,
        "stratified_by": "source project, then capture-sequence group",
    }
    Path("dataset/split_report.json").write_text(json.dumps(report, indent=2))

    # --- Report ------------------------------------------------------------
    log("\n" + "=" * 72)
    log("DATASET BUILD REPORT")
    log("=" * 72)
    log("groups: %d | images: %d | leakage check: PASSED" % (len(groups), total))
    log("\nimages per split:")
    for split in ("train", "val", "test"):
        pct = 100.0 * split_counts[split] / total if total else 0
        log("   %-6s %5d  (%.1f%%)" % (split, split_counts[split], pct))

    log("\nannotation instances per class:")
    log("   %-22s %8s %8s %8s %8s" % ("class", "train", "val", "test", "TOTAL"))
    for name in TARGET_CLASSES:
        log("   %-22s %8d %8d %8d %8d" % (
            name,
            split_class_counts["train"][name],
            split_class_counts["val"][name],
            split_class_counts["test"][name],
            class_stats[name],
        ))

    if drop_stats:
        log("\ndropped annotations (unmapped or invalid):")
        for key, count in drop_stats.most_common():
            log("   %-32s %6d" % (key, count))

    thin = [n for n in TARGET_CLASSES if class_stats[n] < 100]
    if thin:
        log("\nWARNING: under 100 instances for %s." % ", ".join(thin))
        log("         Report this honestly in MEMO.md or merge the class.")

    log("\nwrote dataset/split_report.json")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the unified logistics dataset")
    parser.add_argument("--inspect", action="store_true",
                        help="download and print source class vocabularies, then exit")
    parser.add_argument("--background-ratio", type=float, default=0.10,
                        help="max background images as a fraction of annotated images")
    args = parser.parse_args()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        log("ROBOFLOW_API_KEY is not set. Add it to .env "
            "(free key at https://app.roboflow.com/settings/api).")
        return 1

    downloaded = download_sources(api_key)
    if not downloaded:
        log("No sources downloaded. Nothing to do.")
        return 1

    if args.inspect:
        inspect(downloaded)
        return 0

    inspect(downloaded)
    report = build(downloaded, args.background_ratio)
    return 0 if report else 1


if __name__ == "__main__":
    sys.exit(main())
