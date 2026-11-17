
from pathlib import Path
from collections import Counter
import shutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
OUTPUT = DATASETS / "unified"

CLASSES = [
    "person", "car", "truck", "bus", "motorcycle",
    "helmet", "vest", "fire", "smoke", "spill"
]

# Сопоставление исходных классов с общими
SOURCES = {
    "coco_selected": {
        "mapping": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        "layout": "standard",
    },
    "ppe": {
        "mapping": {6: 0, 0: 5, 2: 6},
        "layout": "standard",
    },
    "fire_smoke": {
        "mapping": {0: 7, 1: 8},
        "layout": "roboflow",
    },
    "spills": {
        "mapping": {0: 9, 1: 9},
        "layout": "roboflow",
    },
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_paths(source_name, config, split):
    root = DATASETS / source_name

    if config["layout"] == "standard":
        return (
            root / "images" / split,
            root / "labels" / split,
        )

    source_split = "valid" if split == "val" else split

    return (
        root / source_split / "images",
        root / source_split / "labels",
    )


def main():
    if OUTPUT.exists():
        raise RuntimeError(
            f"Папка {OUTPUT} уже существует. "
            "Не перезаписываем существующие данные."
        )

    stats = Counter()
    objects = Counter()
    skipped = Counter()

    for source_name, config in SOURCES.items():
        print(f"\nОбрабатываем: {source_name}")

        for split in ("train", "val", "test"):
            images_dir, labels_dir = get_paths(
                source_name, config, split
            )

            if not images_dir.is_dir():
                raise FileNotFoundError(images_dir)

            if not labels_dir.is_dir():
                raise FileNotFoundError(labels_dir)

            images = sorted(
                p for p in images_dir.rglob("*")
                if p.is_file()
                and p.suffix.lower() in EXTENSIONS
            )

            for image in images:
                relative = image.relative_to(images_dir)
                label = labels_dir / relative.with_suffix(".txt")

                if not label.exists():
                    raise FileNotFoundError(
                        f"Нет разметки: {label}"
                    )

                new_lines = []

                for line in label.read_text(
                    encoding="utf-8-sig"
                ).splitlines():
                    parts = line.strip().split()

                    if not parts:
                        continue

                    old_id = int(parts[0])

                    # Если разметка содержит полигон сегментации,
                    # преобразуем его в Bounding Box
                    if len(parts) > 5:

                        if (len(parts) - 1) % 2 != 0 or len(parts) < 7:
                            raise ValueError(
                                f"Неверная сегментационная разметка: {label}"
                            )

                        values = [float(v) for v in parts[1:]]

                        xs = values[0::2]
                        ys = values[1::2]

                        x_min = min(xs)
                        x_max = max(xs)
                        y_min = min(ys)
                        y_max = max(ys)

                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        width = x_max - x_min
                        height = y_max - y_min

                        parts = [
                            parts[0],
                            str(x_center),
                            str(y_center),
                            str(width),
                            str(height),
                        ]

                    elif len(parts) != 5:
                        raise ValueError(
                            f"Неверный формат разметки: {label}"
                        )

                    if old_id not in config["mapping"]:
                        continue

                    coords = [float(v) for v in parts[1:]]

                    if (
                        not all(0 <= v <= 1 for v in coords)
                        or coords[2] <= 0
                        or coords[3] <= 0
                        or coords[0] - coords[2] / 2 < -1e-4
                        or coords[1] - coords[3] / 2 < -1e-4
                        or coords[0] + coords[2] / 2 > 1.0001
                        or coords[1] + coords[3] / 2 > 1.0001
                    ):
                        raise ValueError(
                            f"Неверные координаты: {label}"
                        )

                    new_id = config["mapping"][old_id]

                    new_lines.append(
                        f"{new_id} " + " ".join(parts[1:])
                    )
                    objects[CLASSES[new_id]] += 1

                # Пропускаем изображения без выбранных классов
                if not new_lines:
                    skipped[source_name] += 1
                    continue

                out_images = OUTPUT / "images" / split
                out_labels = OUTPUT / "labels" / split

                out_images.mkdir(parents=True, exist_ok=True)
                out_labels.mkdir(parents=True, exist_ok=True)

                # Уникальные имена для разных датасетов
                stem = source_name + "_" + "_".join(
                    relative.with_suffix("").parts
                )

                new_image = out_images / (
                    stem + image.suffix.lower()
                )
                new_label = out_labels / (stem + ".txt")

                if new_image.exists() or new_label.exists():
                    raise RuntimeError(
                        f"Совпадение имён: {stem}"
                    )

                shutil.copy2(image, new_image)

                new_label.write_text(
                    "\n".join(new_lines) + "\n",
                    encoding="utf-8"
                )

                stats[(source_name, split)] += 1

            print(
                f"  {split}: "
                f"{stats[(source_name, split)]} изображений"
            )

    config = {
        "path": str(OUTPUT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASSES),
        "names": CLASSES,
    }

    with open(OUTPUT / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config, f,
            allow_unicode=True,
            sort_keys=False
        )

    print("\n=== ИТОГ ===")
    print("Изображений:", sum(stats.values()))

    for class_name in CLASSES:
        print(f"{class_name}: {objects[class_name]} объектов")

    print("\nПропущенные изображения:", dict(skipped))
    print("Готово:", OUTPUT)


if __name__ == "__main__":
    main()
