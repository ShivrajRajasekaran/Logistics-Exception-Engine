# Reproducibility Guide

Every number below was measured on the machine in section 1 during the actual
run, not estimated in advance. Where an early estimate turned out wrong, the
correction is recorded rather than quietly replaced.

---

## 1. Hardware and environment

Verified on this machine, not copied from a template:

| Item | Value |
| :--- | :--- |
| OS | Windows 11 Home Single Language, build 10.0.26200 |
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU |
| VRAM | 7.96 GB usable |
| Compute capability | `sm_120` (Blackwell) |
| NVIDIA driver | 595.97 |
| Python | 3.11.9 |
| PyTorch | 2.7.0+cu128 |
| Ultralytics | 8.3.145 |

### The CUDA constraint that dictates the torch pin

The RTX 50-series is Blackwell, compute capability 12.0. No torch wheel built
against CUDA 12.1 or 12.4 contains kernels for it. Installing the default PyPI
torch on this hardware produces a failure only when a tensor actually reaches
the GPU:

```
CUDA error: no kernel image is available for execution on the device
```

Both `import torch` and `torch.cuda.is_available()` return successfully in that
broken state, so neither is a valid check. Verify with a real allocation:

```bash
python -c "import torch; x=torch.randn(512,512,device='cuda'); (x@x).sum(); torch.cuda.synchronize(); print(torch.cuda.get_arch_list())"
```

`sm_120` must appear in the printed arch list. On this machine it does:

```
['sm_50','sm_60','sm_61','sm_70','sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']
```

---

## 2. Environment setup

```bash
# Python 3.11 is required. 3.13 has no wheels for parts of this stack, and 3.11
# matches the Docker base image so the checkpoint loads under identical versions.
py -3.11 -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

python -m pip install --upgrade pip

# GPU build FIRST. This carries the CUDA 12.8 index URL.
pip install -r requirements-cuda.txt

# Then everything else. torch is already satisfied, so this leaves it alone.
pip install -r requirements.txt
```

For CPU-only inference, skip `requirements-cuda.txt` entirely and install
`requirements.txt` alone. Training will not be practical, but the API will serve.

---

## 3. Dataset

The dataset is assembled from public Roboflow Universe projects by
`scripts/prepare_dataset.py`. It is not committed to the repository; the script
reproduces it. `dataset/split_report.json` IS committed so the exact class
distribution can be audited without downloading anything.

```bash
# Free key from https://app.roboflow.com/settings/api
cp .env.example .env
# set ROBOFLOW_API_KEY in .env

# Phase 1: print each source project's real class vocabulary and show which
# classes map to our scheme and which are dropped.
python scripts/prepare_dataset.py --inspect

# Phase 2: download, remap, dedupe, split, write.
python scripts/prepare_dataset.py
```

Determinism: the split is seeded at 42 and grouped by capture sequence, so
re-running reproduces the same partition. The script raises rather than
continues if any filename lands in more than one split.

Resulting layout:

```text
dataset/
├── data.yaml
├── split_report.json
├── train/{images,labels}/
├── val/{images,labels}/
└── test/{images,labels}/
```

### `data.yaml`

```yaml
path: ./dataset
train: train/images
val: val/images
test: test/images

nc: 2
names:
  0: package
  1: damaged-package
```

Both classes are outside COCO, which has backpack, handbag and suitcase but no
box, package, parcel or carton category. The scheme was reduced from an earlier
five- and then three-class design; MEMO.md section 2 records what was cut and
the measured evidence for each cut.

---

## 4. Training

### Hyperparameters

All set explicitly in `scripts/train.py`. Nothing is auto-selected.

| Parameter | Value | Why |
| :--- | :--- | :--- |
| Base checkpoint | `rtdetr-l.pt` | RT-DETR Large, as required |
| Epochs | 60 requested, stopped at 58 (best epoch 43) | early stopping, patience 15 |
| Batch size | 4 | batch 8 at 640px raises `torch.OutOfMemoryError` on 7.96 GB |
| Image size | 640 | RT-DETR's pretrained resolution; 512 loses small-defect detail |
| Optimizer | AdamW | fixed, not `auto` |
| lr0 / lrf | 1e-4 / 0.01 | cosine schedule to 1% of base |
| Weight decay | 1e-4 | |
| Warmup epochs | 3.0 | |
| Seed | 42 | all RNGs pinned; see determinism note below |
| Workers | 2 | Windows spawns workers as processes, so 8 costs more than it returns |
| `flipud` | 0.0 | parcels are gravity-oriented; vertical flip is unphysical |
| `close_mosaic` | 10 | mosaic off for the last 10 epochs to settle box regression |

### Determinism: seeds pinned, cuDNN autotuning left on

The default is `strict_deterministic=False`. Seeds, data order, augmentation
draws, and weight init are all fixed at 42, so the run is reproducible to
within cuDNN kernel non-determinism, the normal standard for a detection
benchmark.

Full bitwise determinism is available with `--strict-deterministic` but is not
the default, because RT-DETR's deformable attention calls
`grid_sampler_2d_backward`, which has no deterministic CUDA implementation, so
PyTorch falls back to a slower path and also loses cuDNN autotuning across the
whole backbone.

A note on timing, because we got this wrong once. Early cold-start readings
suggested ~2.2 s/iteration and a 27-hour run. The steady state is 3.6-3.75 it/s,
about 4m55s per epoch on the v3 dataset. Do not size a training budget from the first thirty
iterations; the first epoch includes dataset scanning, AMP checks and cuDNN
autotuning. GPU utilisation still peaks around 88% but dips between the many
small kernels that Ultralytics' Python-loop deformable attention launches.

### Command

Run from the repository root with the venv active. All defaults are already the
recommended values, so no flags are needed:

```bash
python scripts/train.py --name dataset_v3 --epochs 60
```

Equivalent explicit form:

```bash
python scripts/train.py \
  --model rtdetr-l.pt \
  --data dataset/data.yaml \
  --epochs 60 \
  --batch 4 \
  --imgsz 640 \
  --seed 42 \
  --workers 2 \
  --device 0 \
  --name dataset_v3
```

Expected: ~5m per epoch on 4,379 training images (1,095 iterations), so roughly
5.4 hours before early stopping intervenes. Best weights land at
`runs/train/dataset_v3/weights/best.pt`; copy that to `weights/best.pt` to serve.

The script writes `run_manifest.json` beside the weights, capturing the torch
version, CUDA version, resolved device name, and every hyperparameter actually
used. That file is the ground truth for what ran, not this document.

| Measurement | Value |
| :--- | :--- |
| Batch size actually used | 4 |
| Epochs completed | 58 of 60; best checkpoint epoch 43 |
| Wall-clock training time | 5.420 hours |
| Steady-state throughput | 3.58-3.75 it/s, ~4m55s per epoch |
| Peak VRAM | 4.03 GB of 7.96 GB |
| Iterations per epoch | 1,095 |

---

### Serving-time de-duplication

`DEDUP_IOU=0.7` suppresses the weaker of two heavily-overlapping same-class boxes.
RT-DETR is NMS-free, so duplicate decoder queries otherwise survive. Measured with
`scripts/tune_dedup.py`: false positives 420 -> 330, precision 0.638 -> 0.692, recall
unchanged at 0.749.

## 5. Evaluation

```bash
python scripts/evaluate.py --weights weights/best.pt --split test
```

Writes `runs/eval/metrics.json` with per-class precision, recall, mAP@50, and
mAP@50-95. The metrics table in `MEMO.md` is transcribed from that file.

Evaluation runs at `conf=0.001`, which is correct for mAP: a higher confidence
floor truncates the precision-recall curve and inflates the result. The serving
default of 0.25 is an operational choice and is not what the metrics use.

| Measurement | Value (test split) |
| :--- | :--- |
| mAP@50 (all) | 0.730 |
| mAP@50-95 (all) | 0.601 |
| mAP@50 `package` | 0.834 |
| mAP@50 `damaged-package` | 0.625 |
| Inference latency | 24.2 ms/image warm; first request ~2.5 s (CUDA warm-up) |

Read MEMO.md section 4 before quoting the 0.601. It is inflated by a
dataset artefact, and 0.208 is the honest summary of this model's damage
sensitivity.
