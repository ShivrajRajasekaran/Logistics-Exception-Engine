"""
Build the unified two-class logistics dataset from Roboflow Universe exports.

WHY THIS SCRIPT EXISTS
Seven public projects are merged here, each with its own label vocabulary, into
one contiguous two-class scheme. That merge is the part of the pipeline most
likely to silently corrupt a dataset, so every decision is explicit and logged:
which source class became which target class, and which were deliberately
discarded.

Five of the seven are single-class, which is what made the first attempt fail.
Damage projects annotate only damaged parcels and package projects only intact
ones, so a model could separate the classes by recognising which project an
image came from. The first trained model did exactly that, scoring 99.97% at
guessing the class from source identity alone.

Two sources carry both classes and are what fixed it:

    newbox-mixed     2,110 damaged + 1,308 package   (the decisive one)
    biradar-damage      69 damaged +     6 package   (marginal)

check_source_independence() below is the guard that stops the single-class
regime returning unnoticed: it refuses to write a dataset in which any class
exceeds 0.95 concentration in one source.

USAGE
  # Phase 1 - see what the sources actually contain before mapping anything:
  python scripts/prepare_dataset.py --inspect

  # Phase 2 - build the merged dataset:
  python scripts/prepare_dataset.py

SCOPE: OFFLINE BUILD TOOL, NOT PART OF THE SERVED APPLICATION
This script is the only thing in the repository that talks to Roboflow, and it
runs offline, before training. It needs ROBOFLOW_API_KEY in .env (free account,
universe.roboflow.com) to re-download the public source projects.

The served API does not import this module, does not read that variable, and
makes no outbound request to Roboflow. The Roboflow SDK is deliberately absent
from requirements.txt and from the Docker image; it lives only in
requirements-data.txt. Run this only to rebuild the dataset from scratch.
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

TARGET_CLASSES = {
    "package": 0,
    "damaged-package": 1,
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SOURCES = [
    {
        # THE SOURCE THAT BREAKS THE SHORTCUT.
        #
        # Every other source supplies exactly one class, so "which dataset is
        # this" predicted the label at 99.97% and the v1 model learned capture
        # provenance instead of damage. This project photographs BOTH intact
        # and damaged boxes in one capture setup:
        #     Intact 1,381 instances vs damaged 2,216 (crushed/punctured/torn/leaking)
        #     all images 640x640 -> resolution-only accuracy 0.6128, which is
        #     exactly the majority baseline, i.e. zero resolution leakage.
        # A detector trained on this cannot separate the classes by provenance;
        # it has to look at the box.
        #
        # The four damage types collapse into `damaged-package`: they trigger
        # the same operational path, and keeping them apart would re-create the
        # thin-class problem that forced the earlier severity-tier collapse.
        "name": "newbox-mixed",
        "workspace": "damages-intact-box-dataset",
        "project": "new-box-dataset",
        "version": 4,
        "remap": {
            "intact box": "package",
            "crushed box": "damaged-package",
            "punctured box": "damaged-package",
            "torn box": "damaged-package",
            "leaking box": "damaged-package",
        },
    },
    {
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


# --- Canonical image normalization ---------------------------------------
# Every image is resampled through an IDENTICAL pipeline before it is written:
# downscale to a common bottleneck with INTER_AREA, then up to the canonical
# size with INTER_LINEAR.
#
# WHY. The v1 dataset leaked the label through capture provenance. Measured on
# the built v1 data: a rule using only image resolution classified 99.97% of
# images, because haw-packages (the sole `package` source) was the sole 416x416
# source. Resizing everything up to 640 would have hidden that from the
# resolution diagnostic while leaving the real signal intact: after resize to
# 640, a single sharpness threshold still separated the classes at 88.78%
# against a 56.47% baseline, since upscaled images are blurry and native-640
# images are sharp.
#
# Forcing every image through the same bottleneck equalises that. Measured on a
# 900-image sample:
#     native -> 640            sharpness rule 0.8922   medians 30.7 / 198.6
#     -> 416 -> 640  (chosen)  sharpness rule 0.6289   medians 30.7 /  42.3
#     -> 320 -> 640            sharpness rule 0.7444
# 416 was chosen by measurement, not assumption; a harder bottleneck was worse.
#
# YOLO labels are relative to image dimensions, so a uniform resize leaves every
# box valid without recomputation.
CANONICAL_SIZE = 640
BOTTLENECK_SIZE = 416

RAW_DIR = Path("dataset/_raw")
OUT_DIR = Path("dataset")


def log(msg: str) -> None:
    print(msg, flush=True)


def normalize_image(src: Path, dst: Path) -> bool:
    """Write `src` to `dst` through the canonical resampling pipeline.

    Returns False if the image cannot be decoded, so the caller can drop it
    rather than write a corrupt file.
    """
    import cv2

    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        return False
    small = cv2.resize(image, (BOTTLENECK_SIZE, BOTTLENECK_SIZE),
                       interpolation=cv2.INTER_AREA)
    canonical = cv2.resize(small, (CANONICAL_SIZE, CANONICAL_SIZE),
                           interpolation=cv2.INTER_LINEAR)
    return bool(cv2.imwrite(str(dst), canonical, [cv2.IMWRITE_JPEG_QUALITY, 95]))


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


def check_source_independence(source_class_counts: Dict[str, Counter],
                              max_concentration: float,
                              allow_skew: bool) -> dict:
    """
    Fail the build when a class is supplied almost entirely by one source.

    WHY THIS GATE EXISTS
    The first model trained from this pipeline reached 0.601 mAP50 and a
    `package` recall of exactly 1.000, then turned out to be a source
    classifier rather than a damage detector: no source carried both classes,
    so "which dataset is this image from" predicted the label perfectly. That
    was discovered by auditing predictions AFTER two training runs. It is
    cheaper to catch here, before any GPU time is spent.

    Two numbers are reported per class:
      concentration - share of instances from that class's largest single source
      sources       - how many sources contribute the class at all
    Plus the count of sources carrying more than one class, which is the
    quantity that actually determines whether the shortcut is available.
    """
    classes = sorted(TARGET_CLASSES)
    totals = {c: sum(counts[c] for counts in source_class_counts.values()) for c in classes}
    concentration, dominant = {}, {}
    for c in classes:
        per_src = {s: counts[c] for s, counts in source_class_counts.items() if counts[c]}
        if not per_src or not totals[c]:
            concentration[c], dominant[c] = 1.0, "none"
            continue
        top_source = max(per_src, key=per_src.get)
        dominant[c] = top_source
        concentration[c] = per_src[top_source] / totals[c]

    multi_class_sources = [s for s, counts in source_class_counts.items()
                           if sum(1 for c in classes if counts[c]) > 1]

    log("\n" + "=" * 72)
    log("CLASS-SOURCE INDEPENDENCE CHECK")
    log("=" * 72)
    log("   %-22s %10s %8s  %s" % ("class", "top-source", "sources", "dominant source"))
    for c in classes:
        n_src = sum(1 for counts in source_class_counts.values() if counts[c])
        log("   %-22s %9.1f%% %8d  %s"
            % (c, 100 * concentration[c], n_src, dominant[c]))
    log("\n   sources carrying more than one class: %d of %d  %s"
        % (len(multi_class_sources), len(source_class_counts),
           multi_class_sources or "(none)"))

    worst = max(concentration, key=concentration.get) if concentration else None
    report = {
        "concentration_per_class": {c: round(concentration[c], 4) for c in classes},
        "dominant_source_per_class": dominant,
        "multi_class_sources": multi_class_sources,
        "threshold": max_concentration,
        "passed": bool(worst and concentration[worst] <= max_concentration),
    }

    if worst and concentration[worst] > max_concentration:
        message = (
            "Class-source independence FAILED: '%s' draws %.1f%% of its instances "
            "from a single source (%s), above the %.0f%% threshold, and only %d "
            "source(s) carry more than one class.\n"
            "A detector trained on this split can separate the classes by source "
            "appearance alone, which will not transfer to unseen data.\n"
            "Fix the data, or pass --allow-source-skew to proceed deliberately."
            % (worst, 100 * concentration[worst], dominant[worst],
               100 * max_concentration, len(multi_class_sources))
        )
        if not allow_skew:
            raise RuntimeError(message)
        log("\n   WARNING (proceeding under --allow-source-skew):\n   "
            + message.replace("\n", "\n   "))
        report["overridden"] = True

    return report


def build(downloaded: List[Tuple[dict, Path]], max_background_ratio: float,
          max_concentration: float = 0.95, allow_skew: bool = False,
          normalize: bool = False) -> dict:
    """Merge, remap, split, and write the unified dataset."""
    random.seed(SEED)

    staged: List[Tuple[Path, str, str]] = []
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

    cap = int(len(staged) * max_background_ratio)
    random.shuffle(backgrounds)
    for image_path, group in backgrounds[:cap]:
        staged.append((image_path, "", group))
    log("[build] kept %d background images (cap %d of %d available)"
        % (min(cap, len(backgrounds)), cap, len(backgrounds)))

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

    # Pre-flight: tally class instances per source from the staged items and gate
    # the build BEFORE any files are written. Failing here costs nothing; failing
    # after the write loop would leave a corrupt dataset on disk and waste the
    # copy of several thousand images.
    source_class_counts = defaultdict(Counter)
    for _, content, group in staged:
        source_name = group.split("::")[0]
        for line in content.splitlines():
            if line:
                idx = int(line.split()[0])
                name = [k for k, v in TARGET_CLASSES.items() if v == idx][0]
                source_class_counts[source_name][name] += 1

    independence = check_source_independence(
        source_class_counts, max_concentration, allow_skew)

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

        stem = "%s_%s" % (stem, hashlib.md5(str(image_path).encode()).hexdigest()[:8])

        if normalize:
            if not normalize_image(image_path, OUT_DIR / split / "images" / (stem + ".jpg")):
                drop_stats["undecodable-image"] += 1
                continue
        else:
            shutil.copy2(image_path,
                         OUT_DIR / split / "images" / (stem + image_path.suffix))
        (OUT_DIR / split / "labels" / (stem + ".txt")).write_text(content, encoding="utf-8")

        split_counts[split] += 1
        written_stems[split].add(stem)
        for line in content.splitlines():
            if line:
                idx = int(line.split()[0])
                name = [k for k, v in TARGET_CLASSES.items() if v == idx][0]
                split_class_counts[split][name] += 1

    overlap = ((written_stems["train"] & written_stems["test"])
               | (written_stems["train"] & written_stems["val"])
               | (written_stems["val"] & written_stems["test"]))
    if overlap:
        raise RuntimeError("Split leakage: %d filenames appear in multiple splits" % len(overlap))

    for split in SPLIT_RATIOS:
        on_disk = len(list((OUT_DIR / split / "images").iterdir()))
        if on_disk != split_counts[split]:
            raise RuntimeError(
                "Write mismatch in '%s': staged %d images but %d are on disk. "
                "Output filenames are colliding."
                % (split, split_counts[split], on_disk))

    # --- Resolution-vs-class gate ----------------------------------------
    # Canonical normalization should make every written image identical in
    # size. Verify that on disk rather than trusting the code above: if a
    # resolution still predicts the class, the shortcut survived.
    from PIL import Image as _Image
    res_groups = defaultdict(Counter)
    for split in SPLIT_RATIOS:
        for label_path in (OUT_DIR / split / "labels").glob("*.txt"):
            classes = {c for c in (
                [k for k, v in TARGET_CLASSES.items() if v == int(line.split()[0])][0]
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if len(line.split()) >= 5)}
            if len(classes) != 1:
                continue
            matches = list((OUT_DIR / split / "images").glob(label_path.stem + ".*"))
            if not matches:
                continue
            try:
                res_groups[_Image.open(matches[0]).size][classes.pop()] += 1
            except Exception:
                continue

    res_total = sum(sum(c.values()) for c in res_groups.values())
    res_correct = sum(max(c.values()) for c in res_groups.values())
    res_acc = res_correct / res_total if res_total else 0.0
    log("[build] resolution-only class predictability: %.4f across %d distinct size(s)"
        % (res_acc, len(res_groups)))
    if len(res_groups) > 1:
        log("[build] WARNING: images were written at %d different sizes; canonical "
            "normalization did not apply uniformly." % len(res_groups))

    for cache in OUT_DIR.rglob("*.cache"):
        cache.unlink()
        log("[build] removed stale label cache %s" % cache)

    report = {
        "seed": SEED,
        "class_source_independence": independence,
        "class_instances_per_source": {s: dict(c) for s, c in source_class_counts.items()},
        "total_images": total,
        "split_counts": dict(split_counts),
        "class_instances_total": dict(class_stats),
        "class_instances_per_split": {k: dict(v) for k, v in split_class_counts.items()},
        "images_per_source": dict(per_source),
        "dropped_annotations": dict(drop_stats),
        "groups": len(groups),
        "split_ratios_requested": SPLIT_RATIOS,
        "stratified_by": "source project, then capture-sequence group",
        "canonical_normalization": normalize,
        "canonical_size": CANONICAL_SIZE if normalize else None,
        "bottleneck_size": BOTTLENECK_SIZE if normalize else None,
        "resolution_only_accuracy": round(res_acc, 4),
        "distinct_resolutions": len(res_groups),
    }
    Path("dataset/split_report.json").write_text(json.dumps(report, indent=2))

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
    parser.add_argument("--normalize", action="store_true",
                        help="resample every image through a common bottleneck to "
                             "remove capture-provenance leakage. MEASURED RESULT: this "
                             "cut sharpness leakage from +0.32 to +0.06 above chance, "
                             "but damaged-package mAP50 fell 0.217 -> 0.189 and "
                             "per-source spread widened 0.707 -> 0.829, because damage "
                             "evidence is fine detail that the bottleneck also destroys. "
                             "Off by default; kept so the experiment is reproducible.")
    parser.add_argument("--max-concentration", type=float, default=0.95,
                        help="fail if any class draws more than this share of its "
                             "instances from one source (default 0.95)")
    parser.add_argument("--allow-source-skew", action="store_true",
                        help="downgrade the class-source independence failure to a "
                             "warning; use only when the skew is a deliberate, "
                             "documented choice")
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
    report = build(downloaded, args.background_ratio,
                   args.max_concentration, args.allow_source_skew, args.normalize)
    return 0 if report else 1


if __name__ == "__main__":
    sys.exit(main())
