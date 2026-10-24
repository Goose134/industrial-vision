
from pathlib import Path
import random

import cv2
import torch
from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parent.parent

    model_path = (
        project_root
        / "runs"
        / "spills"
        / "experiment_01"
        / "weights"
        / "best.pt"
    )

    images_dir = (
        project_root
        / "datasets"
        / "spills"
        / "test"
        / "images"
    )

    output_dir = project_root / "output" / "spill_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Собираем тестовые изображения
    images = [
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if not images:
        raise FileNotFoundError(
            "В тестовой папке нет изображений"
        )

    # Для воспроизводимости выбираем 10 изображений
    random.seed(42)
    selected = random.sample(images, min(10, len(images)))

    model = YOLO(str(model_path))
    device = 0 if torch.cuda.is_available() else "cpu"

    print("Классы модели:", model.names)

    for image_path in selected:
        frame = cv2.imread(str(image_path))

        if frame is None:
            print("Не удалось открыть:", image_path)
            continue

        result = model.predict(
            source=frame,
            conf=0.35,
            imgsz=640,
            device=device,
            verbose=False
        )[0]

        # Рисуем результаты YOLO
        annotated = result.plot()

        output_path = output_dir / image_path.name

        cv2.imwrite(str(output_path), annotated)

        print(
            f"{image_path.name}: "
            f"обнаружений {len(result.boxes)}"
        )

    print("\nИзображения сохранены в:")
    print(output_dir)


if __name__ == "__main__":
    main()
