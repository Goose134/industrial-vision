"""Safely combine original unified YOLO dataset and new Fire/Smoke/Spills datasets.

Run from industrial-vision project root:
    python scripts/build_unified_finetune.py

Creates datasets/unified_finetune without touching original datasets or weights.
Exact image duplicates are kept once; duplicate images with conflicting labels
are entirely excluded. Fails if an exact image occurs across data splits.
"""

from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from shutil import copy2
import csv
import math

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
OUTPUT = DATASETS / "unified_finetune"
STAGING = DATASETS / "unified_finetune_building"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CLASSES = ["person", "car", "truck", "bus", "motorcycle", "helmet", "vest", "fire", "smoke", "spill"]

# Source folder layout: "separate" for images/train + labels/train,
# "together" for train/images + train/labels.
SOURCES = [
    ("old_unified", DATASETS / "unified", "separate",
     {"train": "train", "val": "val", "test": "test"}, None),
    ("new_fire_smoke", DATASETS / "new_sources" / "fire_smoke_v3", "together",
     {"train": "train", "valid": "val", "test": "test"}, {0: 7, 1: 8}),
    ("new_spills", DATASETS / "new_sources" / "spills_filtered_bbox", "together",
     {"train": "train", "valid": "val", "test": "test"}, {0: 9}),
]


def hash_file(path):
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_label(path, class_map):
    if not path.is_file():
        raise ValueError(f"Нет файла разметки: {path}")
    rewritten = []
    comparison = []
    for number, row in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not row.strip():
            continue
        parts = row.split()
        if len(parts) != 5:
            raise ValueError(f"{path}, строка {number}: нужно 5 значений, получено {len(parts)}")
        try:
            old_cls = int(parts[0])
            new_cls = old_cls if class_map is None else class_map[old_cls]
            if new_cls not in range(len(CLASSES)):
                raise ValueError("Недопустимый класс")
            coords = [float(x) for x in parts[1:]]
            if not all(math.isfinite(x) and 0 <= x <= 1 for x in coords):
                raise ValueError("Координаты должны быть от 0 до 1")
            if coords[2] <= 0 or coords[3] <= 0:
                raise ValueError("Ширина или высота рамки равна нулю")
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{path}, строка {number}: {exc}") from exc
        rewritten.append(f"{new_cls} " + " ".join(f"{x:.6f}" for x in coords))
        comparison.append((new_cls, *(round(x, 4) for x in coords)))
    return tuple(sorted(comparison)), rewritten


def build():
    if OUTPUT.exists():
        raise SystemExit(f"Папка назначения уже существует: {OUTPUT}. Ничего не перезаписано.")
    if STAGING.exists():
        raise SystemExit(f"Найдена незавершённая сборка: {STAGING}. Проверь её перед повтором.")

    groups = defaultdict(list)
    source_stats = Counter()
    print("ПРОВЕРЯЕМ ДАННЫЕ (ничего не изменяем)...", flush=True)
    for source, folder, layout, splits, class_map in SOURCES:
        for split_folder, split in splits.items():
            if layout == "separate":
                images_dir = folder / "images" / split_folder
                labels_dir = folder / "labels" / split_folder
            else:
                images_dir = folder / split_folder / "images"
                labels_dir = folder / split_folder / "labels"
            if not images_dir.is_dir() or not labels_dir.is_dir():
                raise SystemExit(f"Не найдены папки: {images_dir} или {labels_dir}")
            image_paths = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            for img in image_paths:
                relative = img.relative_to(images_dir).with_suffix(".txt")
                label = labels_dir / relative
                try:
                    signature, lines = parse_label(label, class_map)
                    digest = hash_file(img)
                except (OSError, ValueError) as exc:
                    raise SystemExit(f"Ошибка входных данных: {exc}") from exc
                groups[digest].append((source, split, img, signature, lines))
                source_stats[(source, split)] += 1
            print(f"{source:17s} {split:5s}: {len(image_paths)} изображений", flush=True)

    conflicted = {digest for digest, group in groups.items() if len({rec[3] for rec in group}) > 1}
    cross_split = {digest for digest, group in groups.items() if len({rec[1] for rec in group}) > 1}
    if cross_split:
        raise SystemExit(f"Найдено {len(cross_split)} точных дубликатов между train/val/test. Сборка остановлена.")
    rejected = sum(len(groups[digest]) for digest in conflicted)
    duplicate_copies = sum(len(group) - 1 for digest, group in groups.items() if digest not in conflicted)
    selected = [(digest, group[0]) for digest, group in groups.items() if digest not in conflicted]
    print(f"\nВсего входных изображений: {sum(source_stats.values())}")
    print(f"Групп с конфликтующей разметкой: {len(conflicted)} (исключаем {rejected} изображений)")
    print(f"Обычных лишних точных копий: {duplicate_copies} (оставляем по одной)")
    print(f"В итоговый датасет войдёт: {len(selected)} изображений")
    print("\nКОПИРУЕМ В НОВУЮ ПАПКУ (оригиналы не трогаем)...", flush=True)

    try:
        STAGING.mkdir(parents=True, exist_ok=False)
        for split in ("train", "val", "test"):
            (STAGING / "images" / split).mkdir(parents=True)
            (STAGING / "labels" / split).mkdir(parents=True)
        saved = Counter()
        manifest = STAGING / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8-sig") as report:
            writer = csv.writer(report)
            writer.writerow(["source", "split", "original_image", "new_image", "sha256", "object_count"])
            for digest, (source, split, image, signature, lines) in selected:
                base = f"{source}_{digest}"
                dest_image = STAGING / "images" / split / (base + image.suffix.lower())
                dest_label = STAGING / "labels" / split / (base + ".txt")
                copy2(image, dest_image)
                dest_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                writer.writerow([source, split, str(image.relative_to(ROOT)), dest_image.name, digest, len(lines)])
                saved[split] += 1
        yaml_content = (
            f'path: "{OUTPUT.resolve().as_posix()}"\n'
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "nc: 10\n"
            "names: [" + ", ".join(CLASSES) + "]\n"
        )
        (STAGING / "data.yaml").write_text(yaml_content, encoding="utf-8")
        (STAGING / "build_summary.txt").write_text(
            f"Input: {sum(source_stats.values())}\n"
            f"Conflicting groups excluded: {len(conflicted)}\n"
            f"Conflicting images excluded: {rejected}\n"
            f"Exact duplicate copies skipped: {duplicate_copies}\n"
            f"Output images: {len(selected)}\n"
            f"Train: {saved['train']}\nVal: {saved['val']}\nTest: {saved['test']}\n",
            encoding="utf-8",
        )
        STAGING.rename(OUTPUT)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Не удалось завершить копирование: {exc}\nПроверь папку {STAGING}. Исходные данные не изменены.") from exc

    print("\nГОТОВО!")
    for split in ("train", "val", "test"):
        print(f"{split}: {saved[split]}")
    print("Всего:", sum(saved.values()))
    print("Папка:", OUTPUT)
    print("Конфигурация:", OUTPUT / "data.yaml")
    print("Модель и исходные датасеты не изменены.")


if __name__ == "__main__":
    build()
