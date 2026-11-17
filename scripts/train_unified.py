from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("Видеокарта CUDA недоступна")

    data_path = ROOT / "datasets" / "unified" / "data.yaml"

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    model = YOLO("yolo26n.pt")

    model.train(
        data=str(data_path),
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        amp=True,
        project=str(ROOT / "runs" / "unified"),
        name="unified_50_epochs",
        exist_ok=False,
        plots=True,
    )


if __name__ == "__main__":
    main()