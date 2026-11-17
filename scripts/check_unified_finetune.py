"""Read-only validation of datasets/unified_finetune before YOLO fine-tuning.

Place under industrial-vision/scripts and run:
    python scripts/check_unified_finetune.py
"""

from collections import Counter
from pathlib import Path
import math
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "unified_finetune"
EXPECTED = ["person", "car", "truck", "bus", "motorcycle",
            "helmet", "vest", "fire", "smoke", "spill"]
SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def check():
    config_file = DATASET / "data.yaml"
    if not config_file.is_file():
        raise ValueError(f"Нет файла {config_file}")

    data = yaml.safe_load(config_file.read_text(encoding="utf-8-sig"))
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    if names != EXPECTED or int(data.get("nc", -1)) != len(EXPECTED):
        raise ValueError(f"Неверный список классов: {names}")

    print("Классы:", ", ".join(f"{i}={name}" for i, name in enumerate(names)))
    errors = []
    total_images = 0

    for split in SPLITS:
        images_dir = DATASET / "images" / split
        labels_dir = DATASET / "labels" / split
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise ValueError(f"Нет папки: {images_dir} или {labels_dir}")

        images = {p.stem: p for p in images_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
        labels = {p.stem: p for p in labels_dir.glob("*.txt")}
        if len(images) != len([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]):
            errors.append(f"{split}: разные изображения с одинаковым именем без расширения")

        for stem in sorted(images.keys() - labels.keys())[:10]:
            errors.append(f"{split}: отсутствует TXT для {stem}")
        for stem in sorted(labels.keys() - images.keys())[:10]:
            errors.append(f"{split}: TXT без изображения: {stem}")

        class_counts = Counter()
        empty_labels = 0
        for stem, label_path in labels.items():
            try:
                lines = label_path.read_text(encoding="utf-8-sig").splitlines()
            except (OSError, UnicodeError) as ex:
                errors.append(f"{label_path}: {ex}")
                continue
            if not any(row.strip() for row in lines):
                empty_labels += 1
            for lineno, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                parts = line.split()
                try:
                    if len(parts) != 5:
                        raise ValueError("ожидалось пять значений")
                    class_id = int(parts[0])
                    x, y, w, h = (float(v) for v in parts[1:])
                    if class_id not in range(len(EXPECTED)):
                        raise ValueError("неизвестный класс")
                    if not all(math.isfinite(v) for v in (x, y, w, h)):
                        raise ValueError("нечисловые координаты")
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        raise ValueError("координаты вне диапазона 0..1 или пустая рамка")
                    if x - w / 2 < -0.001 or x + w / 2 > 1.001 or y - h / 2 < -0.001 or y + h / 2 > 1.001:
                        raise ValueError("рамка выходит за изображение")
                    class_counts[class_id] += 1
                except ValueError as ex:
                    if len(errors) < 30:
                        errors.append(f"{label_path.name}:{lineno} — {ex}")
        total_images += len(images)
        print(f"\n{split}: {len(images)} изображений, {len(labels)} TXT, пустых TXT: {empty_labels}")
        print("  " + "; ".join(f"{names[i]}={class_counts[i]}" for i in range(len(names))))
        if split == "train" and any(class_counts[i] == 0 for i in range(len(names))):
            errors.append("В train отсутствуют объекты некоторых классов")

    print(f"\nВСЕГО изображений: {total_images}")
    if total_images != 46819:
        print("ВНИМАНИЕ: число отличается от предыдущего отчёта сборки (46819)")
    if errors:
        print("\nОШИБКИ (первые 30):")
        for issue in errors[:30]:
            print(" -", issue)
        raise ValueError(f"Найдены проблемы: {len(errors)}")
    print("\nПРОВЕРКА ПРОЙДЕНА: классы, пути и bbox-разметка выглядят корректно.")
    print("Это не заменяет визуальную проверку и тестирование на независимых видео.")


if __name__ == "__main__":
    try:
        check()
    except (ValueError, OSError, yaml.YAMLError) as exc:
        print("\nОШИБКА:", exc)
        sys.exit(1)
