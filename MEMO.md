# Technical Memo: Parcel Exception Detection and Reasoning API

**Track:** Computer Vision + Applied ML Engineering
**System:** RT-DETR-L parcel condition detection with a framework-free reasoning layer

---

## 1. Domain, dataset sourcing, and labelling

Inbound parcel triage at a sorting hub is a good constrained detection problem: the
classes are visually concrete, the decision is binary (normal flow or exception bay),
and errors cost in both directions. It also forces a non-COCO label set. COCO has
backpack, handbag and suitcase, but no box, package, parcel or carton, so no
off-the-shelf checkpoint can serve this task. Both final classes are non-COCO.

The dataset merges six public Roboflow Universe object-detection projects (CC BY 4.0).
`scripts/prepare_dataset.py` is the reproducible definition; `dataset/split_report.json`
is its committed output. No single source covers the task, so the merge is the real
engineering, and it is where a dataset most easily gets silently corrupted.

| Source | Images | Contributes |
| :--- | ---: | :--- |
| haw-odap3/packages | 1,615 | intact parcels |
| project-hffml/parcel-box-damage | 745 | damaged (severity tiers) |
| damaged-package/packages2 | 292 | damaged |
| box-prfzk/box-a2mjf | 89 | damaged, opened |
| university-of-moratuwa/parcel-damage | 74 | damaged, second institution |
| bhagyashri-biradar/damage-package | 54 | damaged, wet, holed |

**Labelling decisions.** `damage_A/B/C` are severity grades of one defect, applied
inconsistently between the two institutions that use them, so all three collapse into
`damaged-package`. Training three classes to reproduce an inconsistent human judgement
would yield three weak classes instead of one usable one. `Open box`, `wet Package` and
`Package with hole` fold in for the same reason: they trigger the identical operational
path. `Invoice`, `paint` and `label` are dropped and counted, never silently absorbed.

## 2. Pivots, and what drove them

Three classes were cut, each on measured evidence rather than preference.

**`person` and `compromised-seal`** were cut before training. No downloadable source
carries person boxes alongside parcels. `Open box` plus `open` total 35 instances across
every source combined, which memorises rather than trains.

**`printed-label` was cut after a full 3-class training run** that scored 0.243 mAP50.
The audit found its source was food-packaging expiry codes, not logistics routing slips.
That source was 36% of training data and reached 17% recall while the logistics source
reached 99%. Dropping it and its class moved overall mAP50 from **0.243 to 0.601**.
Baseline artefacts are preserved under `runs/archive/` rather than discarded.

Two of the three sources in the original plan turned out to be *classification* projects
with no boxes at all. That is why `prepare_dataset.py --inspect` prints each source's
real class vocabulary before anything is built.

## 3. Split strategy

70/15/15, seeded at 42: 2,869 images into 2,007 / 431 / 431, with 1,623 `package` and
1,584 `damaged-package` instances. Three leakage controls:

**Grouped by capture sequence.** Roboflow exports contain augmented copies under names
like `IMG_0412_jpg.rf.<hash>`. Splitting on raw filenames would put copies of one photo
in both train and test. Groups move as units. This also correctly collapsed a 240-image
source into a single group once we found all 240 were frames of one video clip.

**Content-hash dedup.** 83 duplicate images appear across projects, dropped before splitting.

**Stratified per source, deficit-greedy.** Splitting one pooled list gave 70/18.6/11.4
with validation holding 1,090 `package` against 318 in test, making the two
non-comparable. The build *raises* on filename collision or split overlap: an earlier
version silently overwrote 239 training images, caught only by asserting the report
count against files on disk.

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

**No framework.** `app/reasoning.py` and `app/main.py` import no orchestration library.
Control flow is a sequence of `if` statements in `reason()`; the single LLM call goes
through the official OpenAI SDK. `grep -rE "langchain|llama_index|crewai|autogen"`
returns only the two docstrings declaring their absence.

**Routing** is deterministic keyword matching, chosen over an LLM classifier because it
must be explainable line by line and must not spend a network call deciding whether a
network call is needed. Single words match on word boundaries, so "sealant supplier"
never reaches the GPU. Four outcomes: `VISION_REQUIRED`, `LEDGER_ONLY`,
`UNSUPPORTED_CAPABILITY`, `OUT_OF_SCOPE`.

`UNSUPPORTED_CAPABILITY` is checked **first**, and exists because of the class cuts
above. "Is the seal on this box intact?" matches `box` and `intact` as visual tokens;
without that precedence the detector would run, return parcel boxes, and the LLM would
answer a question about seals using evidence about cartons. "The model has no seal class"
is a different and more useful answer than a guess. Two unit tests assert the vision and
unsupported vocabularies stay disjoint, since an overlap would silently make the vision
entry dead code.

**Guardrail:** threshold 0.65 on peak `damaged-package` confidence, with three outcomes.
A weak defect below 0.65 halts before the prompt is assembled. No defect but a parcel
located above 0.65 returns `CLEAR`. No defect *and* no parcel located also halts, because
the frame may be empty or unreadable, and "detected nothing" is not "nothing is wrong".
That third case is the one most implementations get wrong.

**Unstaged worked example**, produced by the trained model on a real test image:

```
detections: damaged-package 0.4879, damaged-package 0.4418
status:     INSUFFICIENT_INFORMATION
summary:    peak critical-class confidence 0.49 is below the operational
            threshold 0.65. Routing to manual inspection rather than guessing.
```

No LLM call happens on that path. Given section 4 this guardrail is doing real work: a
model with 0.275 precision on damage **must** refuse rather than narrate, or it would
produce confident liability claims against a named carrier from noise.

**Ledger reconciliation.** `IMMUTABLE_TRANSIT_LEDGER` holds two records that exercise
both branches. `PKG-8821` left origin `INTACT`, so damage found here is the carrier's.
`PKG-9940` left `ALREADY_DAMAGED`, so identical detections are *not* a new claim.
Without that second record the layer could rubber-stamp every detection as liability and
still appear to work.
