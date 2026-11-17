from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

model_path = (
    ROOT / "runs" / "unified"
    / "unified_50_epochs" / "weights" / "best.pt"
)

model = YOLO(str(model_path))

model.export(
    format="onnx",
    imgsz=320,
    simplify=True,
    dynamic=False,
    opset=17,
)