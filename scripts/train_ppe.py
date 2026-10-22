
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    # Корневая папка нашего проекта
    project_root = Path(__file__).resolve().parent.parent

    # Проверяем доступность видеокарты
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA недоступна. Проверь PyTorch и видеокарту."
        )

    # Выводим название используемой видеокарты
    print("GPU:", torch.cuda.get_device_name(0))

    # Загружаем предварительно обученную YOLO
    model = YOLO("yolo26n.pt")

    # Запускаем дообучение для распознавания СИЗ
    model.train(
        data="construction-ppe.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        project=str(project_root / "runs" / "ppe"),
        name="experiment_01",
        exist_ok=False,
        patience=10,
        seed=42,
        plots=True
    )

    print("Обучение завершено!")


if __name__ == "__main__":
    main()
