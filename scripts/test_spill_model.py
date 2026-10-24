
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    # Корневая папка проекта
    project_root = Path(__file__).resolve().parent.parent

    # Наша обученная модель
    model_path = (
        project_root
        / "runs"
        / "spills"
        / "experiment_01"
        / "weights"
        / "best.pt"
    )

    # Конфигурация датасета
    data_path = (
        project_root
        / "datasets"
        / "spills"
        / "data_local.yaml"
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Модель не найдена: {model_path}"
        )

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Конфигурация не найдена: {data_path}"
        )

    device = 0 if torch.cuda.is_available() else "cpu"

    print("Начинаем тестирование модели разливов")

    model = YOLO(str(model_path))

    print("Классы:", model.names)

    # Проверяем качество именно на test, а не val
    metrics = model.val(
        data=str(data_path),
        split="test",
        imgsz=640,
        batch=8,
        device=device,
        workers=0,
        project=str(project_root / "runs" / "spills"),
        name="test_evaluation",
        exist_ok=True,
        plots=True
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)

    # F1 из усреднённых Precision и Recall
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    print("\n===== РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ =====")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1:          {f1:.4f}")
    print(f"mAP@50:      {map50:.4f}")
    print(f"mAP@50-95:   {map5095:.4f}")
    print("===================================")


if __name__ == "__main__":
    main()
