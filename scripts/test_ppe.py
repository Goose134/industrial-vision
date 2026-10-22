
from pathlib import Path
from ultralytics import YOLO


def main():
    # Определяем корень проекта
    project_root = Path(__file__).resolve().parent.parent

    # Путь к лучшим весам обученной модели
    model_path = (
        project_root
        / "runs"
        / "ppe"
        / "experiment_01"
        / "weights"
        / "best.pt"
    )

    # Загружаем обученную модель
    model = YOLO(str(model_path))

    print("Модель загружена:", model_path)

    # Проверяем на тестовой выборке
    metrics = model.val(
        data="construction-ppe.yaml",
        split="test",
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        project=str(project_root / "runs" / "ppe"),
        name="test_01",
        plots=True
    )

    # Выводим основные показатели
    print("\n=== Результаты тестирования ===")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
