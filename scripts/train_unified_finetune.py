"""Safe 15-epoch fine-tuning of an existing 10-class YOLO26n checkpoint.

Put this file in <project>/scripts/train_unified_finetune.py and run from
<project> with: python scripts/train_unified_finetune.py

Keeps the previous weights, original data, and ONNX model unchanged.
"""

from pathlib import Path
import sys

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "datasets" / "unified_finetune" / "data.yaml"
START_WEIGHTS = ROOT / "runs" / "unified" / "unified_50_epochs" / "weights" / "best.pt"
RESULTS_DIR = ROOT / "runs" / "finetune"
RUN_NAME = "unified_finetune_15"
OUTPUT_DIR = RESULTS_DIR / RUN_NAME
EXPECTED_CLASSES = [
    "person", "car", "truck", "bus", "motorcycle",
    "helmet", "vest", "fire", "smoke", "spill",
]


def main():
    if not START_WEIGHTS.is_file():
        raise SystemExit(f"Cannot find starting weights: {START_WEIGHTS}")
    if not DATA_YAML.is_file():
        raise SystemExit(f"Cannot find dataset configuration: {DATA_YAML}")
    if OUTPUT_DIR.exists():
        raise SystemExit(
            f"Output already exists: {OUTPUT_DIR}\n"
            "Do not overwrite a previous run. Change RUN_NAME to a new value."
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is not available. Check PyTorch/CUDA in this .venv.")

    data = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8-sig"))
    classes = data.get("names")
    if isinstance(classes, dict):
        classes = [classes[k] for k in sorted(classes, key=int)]
    if classes != EXPECTED_CLASSES or int(data.get("nc", -1)) != len(EXPECTED_CLASSES):
        raise SystemExit(f"Unexpected dataset classes: {classes}")

    model = YOLO(str(START_WEIGHTS))
    model_classes = model.names
    if isinstance(model_classes, dict):
        model_classes = [model_classes[i] for i in sorted(model_classes)]
    if model_classes != EXPECTED_CLASSES:
        raise SystemExit(f"Unexpected weights classes: {model_classes}")

    print("Starting weights:", START_WEIGHTS)
    print("Dataset:", DATA_YAML)
    print("CUDA GPU:", torch.cuda.get_device_name(0))
    print("Output:", OUTPUT_DIR)
    print("Fine-tuning for up to 15 epochs. Original model is not overwritten.\n")
    sys.stdout.flush()

    model.train(
        data=str(DATA_YAML),
        epochs=15,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        cache=False,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.1,
        warmup_epochs=1.0,
        cos_lr=True,
        patience=8,
        amp=True,
        seed=42,
        resume=False,
        save=True,
        save_period=5,
        project=str(RESULTS_DIR),
        name=RUN_NAME,
        exist_ok=False,
        verbose=True,
    )

    print("\nTraining finished. Inspect:")
    print("  Best weights:", OUTPUT_DIR / "weights" / "best.pt")
    print("  Final weights:", OUTPUT_DIR / "weights" / "last.pt")
    print("  Training metrics:", OUTPUT_DIR / "results.csv")
    print("Do not replace models/unified/unified640.onnx before independent testing.")


if __name__ == "__main__":
    main()
