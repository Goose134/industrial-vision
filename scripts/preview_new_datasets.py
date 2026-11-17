"""Визуальная проверка новых YOLO bbox датасетов, без изменений исходных файлов.
Запуск из корня industrial-vision: python scripts/preview_new_datasets.py
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    ("fire_smoke_v3", {0: "fire", 1: "smoke"}, 0, 1),
    ("spills_filtered_bbox", {0: "spillage"}, 0, None),
]
SPLITS = ("train", "valid", "test")
OUT = ROOT / "runs" / "diagnostics" / "new_dataset_previews"
RNG = random.Random(42)
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}


def read_labels(path: Path):
    labels = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}, строка {line_no}: ожидалось 5 значений")
        class_id = int(fields[0])
        x, y, w, h = map(float, fields[1:])
        labels.append((class_id, x, y, w, h))
    return labels


def choose_images(split_path: Path, second_class):
    images = sorted(
        p for p in (split_path / "images").iterdir()
        if p.suffix.lower() in IMAGE_TYPES
    )
    found = []
    for img in images:
        label_path = split_path / "labels" / f"{img.stem}.txt"
        if label_path.is_file():
            ids = {item[0] for item in read_labels(label_path)}
            found.append((img, ids))
    if second_class is None:
        return RNG.sample(found, min(6, len(found)))
    selected = []
    for cl in (0, second_class):
        pool = [(p, ids) for p, ids in found if cl in ids and p not in selected]
        chosen = RNG.sample(pool, min(3, len(pool)))
        selected.extend(p for p, _ in chosen)
    # ensure one image isn't counted twice
    return [(p, ids) for p, ids in found if p in selected]


def draw(image, labels, names):
    height, width = image.shape[:2]
    for cl, xc, yc, bw, bh in labels:
        # YOLO: значения относительные, от 0 до 1
        x1 = max(0, min(width - 1, round((xc - bw / 2) * width)))
        y1 = max(0, min(height - 1, round((yc - bh / 2) * height)))
        x2 = max(0, min(width - 1, round((xc + bw / 2) * width)))
        y2 = max(0, min(height - 1, round((yc + bh / 2) * height)))
        color = (0, 0, 255) if cl == 0 else (0, 255, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        name = names.get(cl, f"UNKNOWN_{cl}")
        pos = (x1, max(18, y1 - 6))
        cv2.putText(image, name, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image


def main():
    total = 0
    for folder, names, _, second in DATASETS:
        for split in SPLITS:
            source = ROOT / "datasets" / "new_sources" / folder / split
            if not source.is_dir():
                raise SystemExit(f"Не найдена папка: {source}")
            output = OUT / folder / split
            output.mkdir(parents=True, exist_ok=True)
            selected = choose_images(source, second)
            count = 0
            for image_path, ids in selected:
                img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"Не удалось прочитать: {image_path}")
                    continue
                labels = read_labels(source / "labels" / f"{image_path.stem}.txt")
                if any(cl not in names for cl, *_ in labels):
                    raise SystemExit(f"Неожиданный класс: {image_path.name}")
                img = draw(img, labels, names)
                target = output / f"{count+1:02d}_{image_path.stem}.jpg"
                success, encoded = cv2.imencode(".jpg", img)
                if success:
                    encoded.tofile(str(target))  # путь Windows с русскими символами
                    count += 1
            total += count
            print(f"{folder} / {split}: сохранено {count} примеров")
    print(f"\nГотово. Всего {total} изображений с рамками.")
    print(f"Открой: {OUT}")
    print("Исходные датасеты не изменены.")


if __name__ == "__main__":
    main()
