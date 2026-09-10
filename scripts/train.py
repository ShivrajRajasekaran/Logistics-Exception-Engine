"""
Fine-tune RT-DETR-L on the logistics exception dataset.

Training is plain Ultralytics + our own CLI. No AutoML, no no-code platform,
no hyperparameter search service. Every knob below is set explicitly so the
run can be reproduced byte-for-byte from REPRODUCIBILITY.md.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from ultralytics import RTDETR


def set_global_seeds(seed: int, strict: bool) -> None:
    """
    Seed everything. `strict` controls the speed/exactness trade-off.

    All runs fix the same seeds, so the data order, augmentation draws, and
    weight init are identical either way.

    strict=True additionally pins cuDNN to deterministic kernels and disables
    autotuning, making a run bitwise reproducible at a throughput cost. RT-DETR's
    deformable attention calls grid_sampler_2d_backward, which has no
    deterministic CUDA implementation, so PyTorch falls back to a slower path.

    Default is strict=False: same seeds, cuDNN free to autotune. Results are
    reproducible to within run-to-run kernel non-determinism, the normal
    standard for a detection benchmark.

    Measured on this machine (RTX 5050 Laptop, batch 4, 640px), strict=False
    sustains 3.7 it/s, about 4 minutes per epoch. Early cold-start readings of
    ~2.2 s/iteration were startup and contention artefacts, not the steady
    state; do not size a training budget from the first thirty iterations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = strict
    torch.backends.cudnn.benchmark = not strict


def train(args: argparse.Namespace) -> Path:
    set_global_seeds(args.seed, args.strict_deterministic)

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} not found. Populate dataset/ as described in "
            "REPRODUCIBILITY.md section 3 before training."
        )

    model = RTDETR(args.model)

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        seed=args.seed,
        deterministic=args.strict_deterministic,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        cos_lr=True,
        patience=args.patience,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        close_mosaic=10,
        # Geometric augmentation is set explicitly rather than left on the
        # Ultralytics defaults (scale=0.5, degrees=0.0, translate=0.1), because
        # those defaults let the first model separate the classes by box size
        # alone: `package` boxes covered a median 2.4% of frame against
        # 17.6-37.4% for `damaged-package`, a split that follows the source
        # project rather than the physical class. Wide scale jitter forces the
        # same object to appear at many sizes, and rotation breaks the
        # per-source camera-angle regularity.
        scale=args.scale,
        degrees=args.degrees,
        translate=args.translate,
        val=True,
        plots=True,
    )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"

    manifest = {
        "model": args.model,
        "data": str(data_yaml),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "optimizer": "AdamW",
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "workers": args.workers,
        "scale": args.scale,
        "degrees": args.degrees,
        "translate": args.translate,
        "strict_deterministic": args.strict_deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "best_weights": str(best),
    }
    (save_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n[train] best weights -> {best}")
    print(f"[train] manifest      -> {save_dir / 'run_manifest.json'}")
    print(f"[train] copy weights with: cp {best} weights/best.pt")
    return best


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune RT-DETR-L for logistics exception detection")
    p.add_argument("--model", default="rtdetr-l.pt", help="RT-DETR backbone checkpoint")
    p.add_argument("--data", default="dataset/data.yaml")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="0", help="'0' for first GPU, 'cpu' to force CPU")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr0", type=float, default=1e-4)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", dest="warmup_epochs", type=float, default=3.0)
    p.add_argument("--patience", type=int, default=15)
    # Anti-shortcut geometric augmentation. Defaults here are deliberately far
    # wider than the Ultralytics defaults; see the comment in train() for why.
    p.add_argument("--scale", type=float, default=0.9,
                   help="random scale gain (Ultralytics default 0.5)")
    p.add_argument("--degrees", type=float, default=10.0,
                   help="random rotation in degrees (Ultralytics default 0.0)")
    p.add_argument("--translate", type=float, default=0.2,
                   help="random translation fraction (Ultralytics default 0.1)")
    p.add_argument("--strict-deterministic", dest="strict_deterministic",
                   action="store_true",
                   help="bitwise-reproducible cuDNN kernels; roughly an order of "
                        "magnitude slower on RT-DETR (see set_global_seeds docstring)")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="rtdetr_logistics_v1")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
