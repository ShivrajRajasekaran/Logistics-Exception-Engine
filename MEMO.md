# Technical Memo: Parcel Exception Detection and Reasoning API

**Track:** Computer Vision + Applied ML Engineering
**System:** RT-DETR-L parcel condition detection with a framework-free reasoning layer

---

## 1. Domain, dataset sourcing, and labelling

Parcel triage at a sorting hub: concrete classes, a binary decision (normal flow or
exception bay), cost in both directions. COCO has backpack, handbag and suitcase but no
box, package, parcel or carton, so no off-the-shelf checkpoint serves it. Both classes
are non-COCO.

Six Roboflow Universe detection projects (CC BY 4.0), merged by
`scripts/prepare_dataset.py`. No single source covers the task, so the merge is the real
engineering and the easiest place to corrupt a dataset silently.

| Source | Images | Contributes |
| :--- | ---: | :--- |
| haw-odap3/packages | 1,615 | intact parcels |
| project-hffml/parcel-box-damage | 745 | damaged (severity tiers) |
| damaged-package/packages2 | 292 | damaged |
| box-prfzk/box-a2mjf | 89 | damaged, opened |
| university-of-moratuwa/parcel-damage | 74 | damaged, 2nd institution |
| bhagyashri-biradar/damage-package | 54 | damaged, wet, holed |

`damage_A/B/C` are severity grades of one defect applied inconsistently across the two
institutions using them, so all three collapse into `damaged-package`; `Open box`, `wet
Package` and `Package with hole` fold in on the same operational path. `Invoice`, `paint`
and `label` are dropped and counted, never silently absorbed.

## 2. Pivots, and what drove them

**`person` and `compromised-seal`** were cut pre-training: no source carries person boxes
beside parcels, and `Open box` plus `open` total 35 instances, which memorises rather
than trains.

**`printed-label` was cut after a full 3-class run** scoring 0.243 mAP50. Its source was
food-packaging expiry codes, not routing slips: 36% of training data at 17% recall while
the logistics source hit 99%. Dropping it moved mAP50 **0.243 to 0.601**; baselines kept
in `runs/archive/`. Two planned sources proved to be *classification* projects with no
boxes, which is why `--inspect` prints each source's real vocabulary first.

## 3. Split strategy

70/15/15 at seed 42: 2,869 images into 2,007 / 431 / 431; 1,623 `package`, 1,584
`damaged-package`.

**Grouped by capture sequence,** because Roboflow ships augmented copies
(`IMG_0412_jpg.rf.<hash>`) and splitting on filenames puts copies of one photo in train
and test. This also collapsed a 240-image source into one group once we found all 240
were frames of a single clip. **Content-hash dedup** removed 83 cross-project duplicates.
**Stratified per source, deficit-greedy:** one pooled list gave 70/18.6/11.4 with
validation holding 1,090 `package` against 318 in test. The build *raises* on filename
collision or split overlap; an earlier version silently overwrote 239 training images,
caught only by asserting the report count against files on disk.

## 4. Results, and why the headline number is misleading

Test split, `weights/best.pt`, 40 epochs:

| Class | P | R | mAP50 | mAP50-95 |
| :--- | ---: | ---: | ---: | ---: |
| package | 0.989 | **1.000** | 0.994 | 0.948 |
| damaged-package | 0.275 | 0.327 | 0.208 | 0.106 |
| **all** | 0.632 | 0.664 | **0.601** | 0.527 |

**The 0.601 should not be believed, and recall of exactly 1.000 is the tell.** A
per-source breakdown shows what the model actually learned:

| Source | Class | Recall | What it predicts there |
| :--- | :--- | ---: | :--- |
| haw-packages | package | 100.0% | `package` x244 |
| parcel-box-damage | damaged | 51.3% | `damaged-package` x327, `package` x1 |
| packages2 | damaged | 0.0% | `damaged-package` x96 |
| box-damage-open | damaged | 73.2% | `damaged-package` x80 |
| moratuwa | damaged | 45.5% | `damaged-package` x15 |
| biradar | damaged | 22.2% | `damaged-package` x28 |

The model emits `package` 244 times on haw-packages images and once everywhere else
combined. **It is a source classifier, not a damage detector.** The root cause is
structural: no source contains both classes, so "which dataset is this" perfectly
predicts the label. Box scale compounds it. `package` boxes cover a median 2.4% of the
frame against 17.6-37.4% for `damaged-package`, so scale alone nearly separates them.

**What the metrics do tell us:** RT-DETR fine-tuning, the data pipeline and the serving
path all work, and `damaged-package` at 0.208 is a fair estimate of genuine damage
sensitivity. **What they do not:** any performance where both classes appear in one
domain, or in the same image. I expect the hidden-set result to land nearer 0.208 than
0.601, and the honest one-number summary of this model is 0.208.

**The fix, given more time,** is not more epochs; validation plateaued by epoch 14 on the
noisier dataset. It is one source annotated for both classes, or self-labelling a few
hundred frames where intact and damaged parcels co-occur.

## 5. Five failure cases

1. **Source-appearance shortcut (systemic).** `package` predicted 244 times on one
   source, once across all others. *Cause:* zero class overlap between sources, so
   dataset identity is a free label. *Fix:* one source annotated for both classes.
2. **Box-scale confound.** Intact boxes median 2.4% of frame, damaged 37.4%. *Cause:*
   differing capture distance and annotation convention per project, not a property of
   damage. *Fix:* aggressive scale jitter and scale-matched sampling.
3. **Total localisation failure on packages2.** 0% recall from 53 instances despite 96
   damage predictions on those same images. *Cause:* that project annotates partial,
   edge-cropped regions (1.31 boxes/image, boxes at frame borders) rather than the whole
   parcel, so predictions never reach IoU 0.5. *Fix:* re-annotate or exclude.
4. **Small damaged regions missed.** 132 of 226 damaged instances missed; the smallest
   missed box covers 0.0011 of the frame. *Cause:* a defect a few dozen pixels wide is
   erased by feature-pyramid downsampling at 640px. *Fix:* tiled inference (SAHI).
5. **False-positive flood: 451 damage FPs against 226 true instances.** *Cause:* two
   compounding effects. Damage datasets annotate only the damaged parcel, leaving
   ambient intact boxes unlabelled, so correct detections score as false positives; and
   the model fires `damaged-package` on anything not resembling the intact source.
   *Fix:* complete the negative annotations; raise the serving threshold.

## 6. Part B: the reasoning layer

**No framework.** Neither module imports an orchestration library; control flow is plain
`if` statements in `reason()`, and the one LLM call uses the official OpenAI SDK.
`scripts/audit_submission.py` verifies this by AST over real imports, not by grepping
prose.

**Routing** is deterministic keyword matching, not an LLM classifier: it must be
explainable line by line and must not spend a network call deciding whether one is
needed. Word-boundary matching keeps "sealant supplier" off the GPU. Four outcomes:
`VISION_REQUIRED`, `LEDGER_ONLY`, `UNSUPPORTED_CAPABILITY`, `OUT_OF_SCOPE`.

`UNSUPPORTED_CAPABILITY` is checked **first**. "Is the seal on this box intact?" matches
`box` and `intact`; without that precedence the detector runs and the LLM answers a seal
question from carton evidence. Two tests assert the vocabularies stay disjoint, since an
overlap would silently make the vision entry dead code.

**Guardrail:** 0.65 on peak `damaged-package` confidence. A weak defect below it halts
before the prompt is assembled. No defect but a parcel above 0.65 returns `CLEAR`. No
defect *and* no parcel located also halts: the frame may be unreadable, and "detected
nothing" is not "nothing is wrong". That third case is the one most implementations get
wrong.

**Unstaged example,** trained model on a real test image:

```
detections: damaged-package 0.4879, damaged-package 0.4418
status:     INSUFFICIENT_INFORMATION
summary:    peak critical-class confidence 0.49 is below the operational
            threshold 0.65. Routing to manual inspection rather than guessing.
```

No LLM call happens there. Per section 4 that is real work: a model at 0.275 precision on
damage **must** refuse rather than narrate, or it invents liability claims against a named
carrier from noise.

**Ledger.** Two records exercise both branches. `PKG-8821` left origin `INTACT`, so damage
here is the carrier's. `PKG-9940` left `ALREADY_DAMAGED`, so identical detections are
*not* a new claim. Without it the layer could rubber-stamp everything as liability and
still look correct.
