
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def main():
    # Корневая папка проекта
    project_root = Path(__file__).resolve().parent.parent

    # Папка скачанного датасета
    dataset_dir = project_root / "datasets" / "spills"

    # Исходный YAML с настройками датасета
    source_yaml = dataset_dir / "data.yaml"

    if not source_yaml.exists():
        raise FileNotFoundError(
            f"Не найден файл датасета: {source_yaml}"
        )

    with open(source_yaml, "r", encoding="utf-8") as file:
        dataset_config = yaml.safe_load(file)

    # Проверяем реальные названия папок выборок
    valid_dir = (
        "valid"
        if (dataset_dir / "valid" / "images").exists()
        else "val"
    )

    # Создаём конфигурацию с абсолютными путями
    # во избежание ошибок при поиске изображений
    local_config = {
        "path": str(dataset_dir.resolve()),
        "train": "train/images",
        "val": f"{valid_dir}/images",
        "test": "test/images",
        "names": dataset_config["names"]
    }

    # Проверяем наличие изображений
    for split in (
        local_config["train"],
        local_config["val"],
        local_config["test"]
    ):
        folder = dataset_dir / split

        if not folder.is_dir():
            raise FileNotFoundError(
                f"Не найдена папка: {folder}"
            )

    config_path = dataset_dir / "data_local.yaml"

    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            local_config,
            file,
            allow_unicode=True,
            sort_keys=False
        )

    print("Классы датасета:", local_config["names"])
    print("Конфигурация:", config_path)

    # Выбираем GPU или CPU
    device = 0 if torch.cuda.is_available() else "cpu"

    print("Устройство обучения:", device)

    # Берём лёгкую YOLO26n
    model = YOLO("yolo26n.pt")

    # Обучаем на размеченных разливах
    model.train(
        data=str(config_path),
        epochs=20,
        imgsz=640,
        batch=8,
        device=device,
        workers=0,
        patience=5,
        project=str(project_root / "runs" / "spills"),
        name="experiment_01",
        exist_ok=True
    )

    print("Обучение завершено!")


if __name__ == "__main__":
    main()
