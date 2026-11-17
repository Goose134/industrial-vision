
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict, Counter

# Папка проекта
ROOT = Path(__file__).resolve().parents[1]

# Исходный COCO
COCO = ROOT / "datasets" / "coco_source"
IMAGES = COCO / "val2017"
ANNOTATIONS = COCO / "annotations" / "instances_val2017.json"

# Куда сохраняем результат
OUTPUT = ROOT / "datasets" / "coco_selected"

# ID COCO -> ID нашей будущей модели
CLASS_MAP = {
    1: 0,  # person
    3: 1,  # car
    8: 2,  # truck
    6: 3,  # bus
    4: 4,  # motorcycle
}

CLASS_NAMES = ["person", "car", "truck", "bus", "motorcycle"]


def convert_annotation(ann, width, height):
    """Переводит COCO bbox в YOLO bbox."""

    x, y, w, h = ann["bbox"]

    x1 = max(0, min(width, x))
    y1 = max(0, min(height, y))
    x2 = max(0, min(width, x + w))
    y2 = max(0, min(height, y + h))

    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return None

    xc = (x1 + x2) / (2 * width)
    yc = (y1 + y2) / (2 * height)
    nw = w / width
    nh = h / height

    class_id = CLASS_MAP[ann["category_id"]]

    return f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"


def main():
    if not ANNOTATIONS.is_file():
        raise FileNotFoundError(
            f"Не найден файл аннотаций: {ANNOTATIONS}"
        )

    if not IMAGES.is_dir():
        raise FileNotFoundError(
            f"Не найдена папка с фотографиями: {IMAGES}"
        )

    if OUTPUT.exists():
        raise RuntimeError(
            "Папка coco_selected уже существует. "
            "Не перезаписываем подготовленные данные."
        )

    print("Загружаем аннотации COCO...")

    with open(ANNOTATIONS, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Группируем разметку по изображениям
    annotations = defaultdict(list)

    for ann in coco["annotations"]:
        if ann["category_id"] in CLASS_MAP:
            if ann.get("iscrowd", 0) == 0:
                annotations[ann["image_id"]].append(ann)

    # Берём изображения с нужными классами
    selected = []

    for image in coco["images"]:
        if image["id"] not in annotations:
            continue

        if not (IMAGES / image["file_name"]).is_file():
            continue

        selected.append(image)

    print("Подходящих изображений:", len(selected))

    # Перемешиваем воспроизводимым образом
    random.Random(42).shuffle(selected)

    total = len(selected)
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    splits = {
        "train": selected[:train_end],
        "val": selected[train_end:val_end],
        "test": selected[val_end:],
    }

    totals = Counter()

    for split, images in splits.items():
        image_dir = OUTPUT / "images" / split
        label_dir = OUTPUT / "labels" / split

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        saved = 0

        for image in images:
            source = IMAGES / image["file_name"]

            lines = []

            for ann in annotations[image["id"]]:
                line = convert_annotation(
                    ann, image["width"], image["height"]
                )

                if line is not None:
                    lines.append(line)
                    totals[int(line.split()[0])] += 1

            if not lines:
                continue

            shutil.copy2(source, image_dir / source.name)

            label_path = label_dir / (
                Path(image["file_name"]).stem + ".txt"
            )

            label_path.write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8"
            )

            saved += 1

        print(f"{split}: {saved} изображений")

    print("\nКоличество размеченных объектов:")

    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {totals[class_id]}")

    print("\nГотово! Данные находятся в:", OUTPUT)


if __name__ == "__main__":
    main()
