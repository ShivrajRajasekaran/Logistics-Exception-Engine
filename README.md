# Parcel Exception Detection & Reasoning API

RT-DETR-L parcel condition detection served through FastAPI, with a hand-written
reasoning layer that decides when to call the detector, reconciles findings against a
transit ledger, and refuses to answer when the evidence is too weak.

**Read [MEMO.md](MEMO.md) section 4 before quoting any metric.** The headline 0.601
mAP50 is inflated by a dataset artefact; 0.208 is the honest measure of this model's
damage sensitivity, and the memo explains exactly why.

---

## Constraints checklist

- [x] **No agentic frameworks.** No LangChain, LlamaIndex, CrewAI, AutoGen, Haystack, or
      Semantic Kernel anywhere. Routing, guardrails and context assembly are vanilla
      Python; the one LLM call uses the official `openai` SDK.
      Verify: `grep -rE "langchain|llama_index|crewai|autogen" app/ scripts/ tests/`
      returns only the two docstrings declaring their absence.
- [x] **RT-DETR.** `ultralytics.RTDETR` fine-tuned from `rtdetr-l.pt`. Own training code,
      no AutoML, no no-code platform.
- [x] **Non-COCO classes.** Both `package` and `damaged-package` are outside the COCO
      label set, which has backpack, handbag and suitcase but no box, package, parcel or
      carton category.
- [x] **Reproducible.** Pinned dependencies, seed 42, documented hardware, measured
      training time, and a `run_manifest.json` written beside every checkpoint. See
      [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
- [x] **Standard upload.** `/api/v1/detect` takes a plain `UploadFile`.

## Architecture

```text
POST /api/v1/detect ──> RT-DETR-L ──> boxes + classes + confidences

POST /api/v1/reason
      │
      ├─ 1. route_intent()      vanilla keyword routing, 4 outcomes
      │      ├─ UNSUPPORTED_CAPABILITY ─> refuse: outside the trained label set
      │      ├─ OUT_OF_SCOPE           ─> refuse: not about this parcel
      │      └─ LEDGER_ONLY            ─> answer from the transit record, no inference
      │
      ├─ 2. run_inference()     same model object as Part A
      ├─ 3. evaluate_guardrail() ─> below 0.65? HALT, no LLM call is made
      ├─ 4. ledger lookup       IMMUTABLE_TRANSIT_LEDGER
      └─ 5. synthesize()        one direct OpenAI SDK call
```

Steps 3 and 5 sit adjacent in [app/main.py](app/main.py) so it is obvious by reading
that no model is consulted when the guardrail fails.

## Quickstart

```bash
py -3.11 -m venv venv
venv\Scripts\activate              # Windows;  source venv/bin/activate elsewhere
python -m pip install --upgrade pip

pip install -r requirements-cuda.txt   # GPU (Blackwell needs CUDA 12.8). Skip for CPU.
pip install -r requirements.txt

cp .env.example .env                   # OPENAI_API_KEY optional, see below
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Model weights (`weights/best.pt`) are included directly in this repository. No
separate download is required: clone, install, and the API serves the trained
model immediately. They load from `WEIGHTS_PATH`, which defaults to that path.

To retrain from scratch, follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md): build the dataset, run
`python scripts/train.py`, then copy `runs/train/rtdetr_logistics_v1/weights/best.pt`
to `weights/best.pt`.

**Without an API key** the service still runs end to end. `/reason` returns a clearly
labelled deterministic summary instead of LLM prose, so routing and guardrail behaviour
are reviewable without credentials.

### Docker

```bash
docker compose up --build
```

Then open **<http://localhost:8000/>** for the demo console, or hit the API directly.
Weights and sample images are bind-mounted, so a checkpoint can be swapped without a
rebuild.

### Demo console

The container serves a single-page UI at `/` alongside the API, so one command brings up
the whole system. It has no build step and no CDN dependencies, so it works offline.

* **Detection panel** draws bounding boxes on a canvas with class colours and confidence
  values, and exports the annotated result as a PNG.
* **Reasoning panel** shows the status as a colour-coded badge, with preset questions that
  exercise each routing branch: a custody question that spends no inference, a question
  about people that returns `UNSUPPORTED_CAPABILITY`, and an unrelated one that returns
  `OUT_OF_SCOPE`.
* **Raw JSON** toggles on both panels, so the exact API response sits beside the visual.

The UI is convenience only. Both endpoints are fully usable without it, and it is excluded
from the OpenAPI schema at `/docs`.

## Endpoints

All responses below are real output from the trained model, not illustrations.

### `GET /health`

```json
{"status":"ok","model_loaded":true,"weights_path":"weights/best.pt",
 "classes":["package","damaged-package"],"critical_classes":["damaged-package"],
 "detection_threshold":0.25,"guardrail_threshold":0.65}
```

### `POST /api/v1/detect`

```bash
curl -X POST http://localhost:8000/api/v1/detect \
     -F "file=@sample_images/intact_parcel.jpg;type=image/jpeg"
```

```json
{
  "status": "success",
  "filename": "intact_parcel.jpg",
  "image_size": [416, 416],
  "inference_time_ms": 35.95,
  "count": 1,
  "detections": [
    {"label": "package", "confidence": 0.967, "bbox": [350.95, 205.96, 412.55, 302.25]}
  ]
}
```

`bbox` is `[x1, y1, x2, y2]` in absolute pixels. Errors are explicit: 415 for a
non-image content type, 400 for undecodable bytes, 413 over the size limit, 503 when no
checkpoint is loaded.

The first request after startup costs ~2.5 s for CUDA warm-up. Steady-state inference is
24 ms per image.

### `POST /api/v1/reason`

**Guardrail refusal.** A genuinely damaged parcel detected at 0.4879, below the 0.65
threshold. The LLM is never called on this path.

```bash
curl -X POST http://localhost:8000/api/v1/reason \
  -H "Content-Type: application/json" \
  -d '{"package_id":"PKG-8821",
       "image_path":"sample_images/damaged_parcel.jpg",
       "query":"Is this parcel damaged enough to raise a carrier claim?"}'
```

```json
{
  "package_id": "PKG-8821",
  "status": "INSUFFICIENT_INFORMATION",
  "requires_vision_model": true,
  "guardrail_passed": false,
  "max_critical_confidence": 0.4879,
  "decision_summary": "INSUFFICIENT_INFORMATION. Defect signal present but weak: peak critical-class confidence 0.49 is below the operational threshold 0.65. Refusing to assign liability from an ambiguous detection. Routing this parcel to manual inspection rather than guessing.",
  "detections": [
    {"label": "damaged-package", "confidence": 0.4879, "bbox": [25.12, 21.79, 637.09, 622.15]},
    {"label": "damaged-package", "confidence": 0.4418, "bbox": [379.02, 89.86, 539.31, 289.1]}
  ],
  "ledger_record": {"carrier": "Apex Logistics", "origin_label_status": "INTACT", "...": "..."}
}
```

**Clear finding.** A parcel located confidently with no defect above threshold. Reported
as `CLEAR` rather than "insufficient", because we know the camera actually saw a parcel.

```json
{
  "package_id": "PKG-9940",
  "status": "CLEAR",
  "requires_vision_model": true,
  "guardrail_passed": true,
  "max_critical_confidence": 0.0,
  "decision_summary": "[deterministic fallback - LLM unavailable] No defect class detected. Observed objects: {'package': 1}.",
  "detections": [
    {"label": "package", "confidence": 0.967, "bbox": [350.95, 205.96, 412.55, 302.25]}
  ]
}
```

**Capability refusal.** A visual question naming something outside the trained label set.
Distinct from `INSUFFICIENT_INFORMATION`: there we looked and the evidence was weak, here
we never had the capability.

```json
{
  "package_id": "PKG-8821",
  "status": "UNSUPPORTED_CAPABILITY",
  "requires_vision_model": false,
  "guardrail_passed": false,
  "decision_summary": "Cannot answer from this model. The question asks about ['people']; the model has no person class. The detector is trained on package and damaged-package only. Answering from those classes would be a confident wrong answer, so this is routed to manual inspection instead."
}
```

**Ledger-only.** A custody question needs no pixels, so no inference is spent.
`status` is `ANSWERED_WITHOUT_VISION` and `requires_vision_model` is `false`.

### Status values

| Status | Meaning |
| :--- | :--- |
| `EXCEPTION_FLAGGED` | damage above threshold, reconciled against the ledger |
| `CLEAR` | parcel located confidently, no defect above threshold |
| `INSUFFICIENT_INFORMATION` | evidence too weak, or frame unreadable. No LLM call |
| `UNSUPPORTED_CAPABILITY` | question is outside the trained label set |
| `ANSWERED_WITHOUT_VISION` | answered from the transit record alone |

## Tests

```bash
python tests/test_reasoning.py      # 24 tests, no GPU/checkpoint/API key needed
```

The decision layer is our own code, so it is verifiable on its own terms rather than
only observable through the model. Coverage includes every routing branch, both sides of
the 0.65 boundary, the empty-frame case, and two invariants: that the vision and
unsupported vocabularies stay disjoint, and that every unsupported token has a reason
string.

## Layout

```text
app/       main.py (endpoints + ledger), detector.py, reasoning.py, schemas.py
scripts/   prepare_dataset.py, train.py, evaluate.py
tests/     test_reasoning.py
dataset/   data.yaml, split_report.json  (images rebuilt by prepare_dataset.py)
runs/      archive/ holds the superseded 3-class baseline for comparison
```
