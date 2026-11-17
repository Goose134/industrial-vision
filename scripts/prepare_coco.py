
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "coco_source" / "annotations"
OUTPUT = ROOT / "datasets" / "coco_selected"

# Оригинальные ID COCO -> наши ID
CLASS_MAP = {
    1: 0,   # person
    3: 1,   # car
    8: 2,   # truck
    6: 3,   # bus
    4: 4,   # motorcycle
}


def load_annotations(split):
    path = SOURCE / f"instances_{split}2017.json"

    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    images = {image["id"]: image for image in data["images"]}
    annotations = defaultdict(list)

    for ann in data["annotations"]:
        if ann["category_id"] in CLASS_MAP and not ann.get("iscrowd", 0):
            annotations[ann["image_id"]].append(ann)

    return images, annotations


def select_images(annotations, limit):
    rng = random.Random(42)
    by_class = defaultdict(list)

    for image_id, anns in annotations.items():
        for class_id in {ann["category_id"] for ann in anns}:
            by_class[class_id].append(image_id)

    for candidates in by_class.values():
        rng.shuffle(candidates)

    selected = []
    used = set()

    # Сначала выбираем изображения с каждым нужным классом
    target = limit // len(CLASS_MAP)

    for class_id in CLASS_MAP:
        added = 0

        for image_id in by_class[class_id]:
            if image_id in used:
                continue

            selected.append(image_id)
            used.add(image_id)
            added += 1

            if added >= target:
                break

    # Заполняем оставшиеся места
    remaining = list(set(annotations) - used)
    rng.shuffle(remaining)

    selected.extend(remaining[:max(0, limit - len(selected))])

    return selected[:limit]


def download_image(split, image):
    folder = OUTPUT / "images" / split
    folder.mkdir(parents=True, exist_ok=True)

    destination = folder / image["file_name"]

    if destination.exists() and destination.stat().st_size > 0:
        return destination

    url = (
        f"https://images.cocodataset.org/"
        f"{'train2017' if split == 'train' else 'val2017'}/"
        f"{image['file_name']}"
    )

    for attempt in range(3):
        try:
            request = Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            with urlopen(request, timeout=30) as response:
                content = response.read()

            if not content.startswith(b"\xff\xd8"):
                raise ValueError("Получен не JPEG")

            destination.write_bytes(content)
            return destination

        except Exception as exc:
            print(f"Ошибка загрузки {image['file_name']}: {exc}")

            if attempt < 2:
                time.sleep(2)

    return None


def convert_labels(image, annotations):
    width = image["width"]
    height = image["height"]
    lines = []

    for ann in annotations:
        x, y, w, h = ann["bbox"]

        x1 = max(0, min(width, x))
        y1 = max(0, min(height, y))
        x2 = max(0, min(width, x + w))
        y2 = max(0, min(height, y + h))

        bw = x2 - x1
        bh = y2 - y1

        if bw <= 0 or bh <= 0:
            continue

        xc = ((x1 + x2) / 2) / width
        yc = ((y1 + y2) / 2) / height

        lines.append(
            f"{CLASS_MAP[ann['category_id']]} "
            f"{xc:.6f} {yc:.6f} "
            f"{bw / width:.6f} {bh / height:.6f}"
        )

    return lines


def prepare(split, limit):
    source_split = "train" if split == "train" else "val"

    print(f"\nПодготовка {split}: {limit} изображений")

    images, annotations = load_annotations(source_split)
    selected = select_images(annotations, limit)

    labels_dir = OUTPUT / "labels" / split
    labels_dir.mkdir(parents=True, exist_ok=True)

    successful = 0

    for index, image_id in enumerate(selected, 1):
        image = images[image_id]
        downloaded = download_image(split, image)

        if downloaded is None:
            continue

        lines = convert_labels(image, annotations[image_id])

        if not lines:
            downloaded.unlink()
            continue

        label_path = labels_dir / (
            Path(image["file_name"]).stem + ".txt"
        )

        label_path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8"
        )

        successful += 1

        if index % 50 == 0 or index == len(selected):
            print(
                f"{split}: обработано {index}/{len(selected)}, "
                f"скачано {successful}"
            )

    print(f"Готово: {split} — {successful} изображений")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=100)
    parser.add_argument("--val", type=int, default=20)
    args = parser.parse_args()

    prepare("train", args.train)
    prepare("val", args.val)
