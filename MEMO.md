# Technical Memo: Logistics Parcel Exception Detection & Reasoning API

**Track:** CV + Applied ML Engineering — RT-DETR-L detection with a framework-free reasoning layer

---

## 1. Domain, sourcing, labelling

Logistics Parcel triage at a sorting hub: concrete classes, a binary operational decision (normal
flow or exception bay), cost in both directions. COCO has backpack, handbag and suitcase
but no box, package, parcel or carton, so both final classes are non-COCO.

Seven public Roboflow Universe detection projects (CC BY 4.0), merged by
`scripts/prepare_dataset.py`; `dataset/split_report.json` is its committed output.

| Roboflow source (workspace/project, version) | package | damaged | images |
| :--- | ---: | ---: | ---: |
| damages-intact-box-dataset/new-box-dataset v4 | 1,308 | 2,110 | 3,389 |
| haw-odap3/packages-iw8aw v3 | 1,617 | 0 | 1,615 |
| project-hffml/parcel-box-damage-classification v1 | 0 | 753 | 745 |
| damaged-package/packages2-8qxzc v1 | 0 | 382 | 292 |
| box-prfzk/box-a2mjf v3 | 0 | 299 | 89 |
| university-of-moratuwa-ztkqd/parcel-damage-detection v2 | 0 | 81 | 74 |
| bhagyashri-biradar/damage-package-detection v7 | 6 | 69 | 54 |

Damage vocabularies collapse into `damaged-package` (crushed, punctured, torn, leaking,
plus inconsistently-applied severity tiers `damage_A/B/C`): they trigger one operational
path. `Invoice`, `paint` and `label` are dropped and counted, never silently absorbed.

## 2. Pivots — what we tried, the evidence, why we changed

**V1 (2,869 images) scored 0.601 mAP50 with a `package` recall of exactly 1.000.** That
perfect recall was treated as suspicious rather than good. Three shortcuts were measured,
each an accuracy achievable *without looking at the parcel* (50.6% baseline):

| shortcut | V1 | V3 |
| :--- | ---: | ---: |
| source identity alone | 99.85% | **79.05%** |
| image resolution alone | 99.83% | **79.05%** |
| box-area ratio, damaged:package | 8.78x | **5.40x** |

Root cause: no source contained both classes, so "which dataset is this" *was* the label.

**V2 rejected.** A common 416 bottleneck cut sharpness leakage from +0.32 to +0.06 above
chance yet made the model worse on an identical test set (damaged mAP50 0.217→0.189, FPs
389→467): damage evidence is fine detail, so the bottleneck destroyed signal and confound
together. They share a spatial frequency; preprocessing cannot separate them. Kept
reproducible behind `--normalize`.

**V3 succeeded by fixing the data, not the training.** `new-box-dataset` photographs
intact *and* damaged boxes in one capture setup (verified visually: same desk, lighting,
distance). Its 640×640 images carry both classes, so resolution stops predicting the label
and the class-source independence gate in `prepare_dataset.py` now passes **without** an
override for the first time.

## 3. Split strategy

70/15/15, seed 42: **6,258 images — including augmented variants — drawn from 2,133
unique capture groups**, split 4,379 / 940 / 939. That distinction matters and is not
cosmetic: the newbox export alone ships 3,567 files from only 627 original captures, 5.69
copies each. Effective visual diversity is the group count, not the image count, and this
memo does not claim 6,258 independent photographs.

Split assignment therefore happens at **capture-group level**, never per image, so
augmented siblings cannot straddle partitions. **Content-hash dedup** removed 178 exact
duplicates. **Stratified per source, deficit-greedy.** Verified by image content rather
than filename: **0 of 2,133 groups and 0 identical images cross splits.** The build raises
on filename collision — an earlier version silently overwrote 239 images before that
assertion existed.

## 4. Results — V1 vs V3 on the identical test set

Comparing a model's old test score against a new one is meaningless when the split
changed, so V1 was re-evaluated on the V3 test set (939 images).

| Metric | V1 | V3 | Δ |
| :--- | ---: | ---: | ---: |
| damaged-package mAP50 | 0.245 | **0.625** | +0.380 |
| damaged-package precision | 0.307 | **0.730** | +0.423 |
| damaged-package recall | 0.397 | **0.627** | +0.230 |
| package mAP50 | 0.626 | **0.834** | +0.207 |
| overall mAP50 | 0.436 | **0.730** | +0.294 |
| background false positives | 1,537 | **307** | −80% |

**Confusion behaviour** @0.25: package 321 correct / 63 called damaged / 1 missed;
damaged 419 correct / 19 called package / 165 missed.

**The honest caveat.** Per-source recall spread *widened*, 0.684 → 0.934: newbox damaged
0.934, but 0.000 on `packages2` and `biradar`, and down on `moratuwa` (0.455→0.273) and
`parcel-box-damage` (0.570→0.395). **V3 traded one source dependence for another rather
than becoming source-independent**; source-only predictability is still 0.79 against a
0.56 baseline. Failure case 3 gives the mechanism. Hidden-set performance may therefore
vary materially with capture style; the 0.625 figure is a measurement on our own held-out
split, not a forecast for an unseen distribution. One real gain: newbox
`package` recall 0.000 → 0.559, so V3 finds intact parcels outside `haw-packages`.

## 5. Five failure cases (shipped model, each image inspected)

Every case below is a specific image with the model's actual prediction, not a category.
Rendered side by side in `reports/samples/v3_failures_final.jpg`.

1. **`newbox_new_170_cropped` — intact called damaged, conf 0.950, IoU 0.99.** Localisation
   is near-perfect; only the class is wrong. The image is a used brown carton with fold
   creases and slightly bulging edges. *Cause:* annotation boundary — creasing on a used
   carton is visually indistinguishable from mild damage, and a human could label it either
   way. *Fix:* severity-graded labels, or an explicit "worn but serviceable" class. Not implemented.
2. **`newbox_82` — intact called damaged, conf 0.945.** A parcel in wrinkled green plastic
   film with a document pouch. *Cause:* crinkled wrap produces the same irregular contour
   signature as a deformed box. *Fix:* train with wrapped-but-intact negatives. Not implemented.
3. **`parcel-box-damage` — damage region missed while the whole parcel is predicted.**
   Ground truth marks one localised crease; the model returns a box covering the parcel, so
   IoU stays below 0.5 and the same prediction counts as both a miss and a false positive.
   *Cause:* **the sources annotate at incompatible granularity.** Median damaged-box area:
   newbox 0.607 and parcel-box-damage 0.374 (whole parcel) versus box-damage-open 0.011 and
   biradar 0.043 (defect region) — a 55x span. newbox supplies 61% of damaged instances, so
   the model learned whole-parcel boxes. This is the mechanism behind the per-source recall
   spread, and why `biradar` and `packages2` sit at 0.000. *Fix:* re-annotate to one
   convention, or train per-convention heads. Not implemented — the honest ceiling here.
4. **Smallest missed damage, 0.00087 of frame.** In
   `box-damage-open_IMG_20221028_174044-removebg-preview` the model does fire (0.73, 0.80)
   but never matches the tiny ground truth. *Cause:* the file is a background-removed line-art outline, not a photograph, and
   the targets are a few dozen pixels wide. *Fix:* reject non-photographic inputs at ingest;
   tiled inference for genuinely small defects. Not implemented.
5. **Duplicate detections on one parcel.** `package 0.959 [210.2,200.7,271.6,301.5]`
   alongside `package 0.575 [210.2,200.5,271.7,301.5]` — one object, two boxes. *Cause:* RT-DETR is
   NMS-free, relying on one-to-one Hungarian matching in training, so duplicate decoder
   queries survive inference; **12.4% of test images emitted more boxes than objects.**
   *Fix:* **implemented.** Class-aware IoU>0.7 suppression in `app/detector.py` cut false
   positives 420 to 330 at zero recall cost (740 TP, 248 FN unchanged), precision
   0.638 to 0.692. Four regression tests pin it.

## 6. Part B: the reasoning layer

**No framework.** Control flow is plain `if` statements in `reason()`; the one LLM call
uses the official OpenAI SDK. `audit_submission.py` verifies this by AST over real imports.

**Routing** is deterministic keyword matching — explainable line by line, and it must not
spend a network call deciding whether one is needed. Four outcomes: `VISION_REQUIRED`,
`LEDGER_ONLY`, `UNSUPPORTED_CAPABILITY`, `OUT_OF_SCOPE`. Refusal requires **positive**
evidence, never an absent keyword: an earlier version refused whenever no keyword matched,
silently rejecting 8 of 16 ordinary questions including the brief's own "What's the most
common object here?". `UNSUPPORTED_CAPABILITY` is checked first, or "Is the seal on this
box intact?" would be answered from carton evidence.

**Guardrail:** 0.65 on peak `damaged-package` confidence. A weak defect halts *before* the
prompt is assembled; a parcel above 0.65 with no defect returns `CLEAR`; no defect *and* no
parcel located also halts, because "detected nothing" is not "nothing is wrong".

**Unstaged insufficient-information example**, real model on a real test image:

```
image:      sample_images/ambiguous_parcel.jpg
detections: damaged-package 0.6227          <- genuinely borderline
status:     INSUFFICIENT_INFORMATION
summary:    peak confidence 0.62 is below the operational threshold 0.65.
            Routing to manual inspection rather than guessing.
```

All three branches reproduce on committed samples: `ambiguous_parcel.jpg` 0.6227 halts,
`damaged_parcel.jpg` 0.8583 flags an exception, `intact_parcel.jpg` 0.9526 with no defect
returns `CLEAR`.

No LLM call on that path. Given section 4 this is essential: a detector with measured
source dependence **must** refuse rather than narrate, or it invents liability claims from
noise. `IMMUTABLE_TRANSIT_LEDGER` exercises both branches — `PKG-8821` left origin
`INTACT`, so damage here is the carrier's; `PKG-9940` left `ALREADY_DAMAGED`, so identical
detections are *not* a new claim. 32 tests pin routing, guardrail boundaries and dedup.
