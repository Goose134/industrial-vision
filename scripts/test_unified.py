from pathlib import Path
from multiprocessing import freeze_support
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    model_path = (
        ROOT / "runs" / "unified"
        / "unified_50_epochs" / "weights" / "best.pt"
    )

    data_path = ROOT / "datasets" / "unified" / "data.yaml"

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(data_path),
        split="test",
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        project=str(ROOT / "runs" / "unified"),
        name="test_50_epochs",
        exist_ok=True,
        plots=True,
    )

    print("\n=== ТЕСТИРОВАНИЕ ЗАВЕРШЕНО ===")
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    freeze_support()
    main()