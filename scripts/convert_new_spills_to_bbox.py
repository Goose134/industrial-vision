from pathlib import Path
from shutil import copy2
import math

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "datasets/new_sources/spills_filtered_v1"
OUTPUT = ROOT / "datasets/new_sources/spills_filtered_bbox"

SPLITS = ["train", "valid", "test"]


def convert_annotation(line):
    values = line.split()

    # В исходном датасете только класс 0 — spillage
    if values[0] != "0":
        raise ValueError(f"Неизвестный класс: {values[0]}")

    coords = list(map(float, values[1:]))

    if not all(math.isfinite(v) and 0 <= v <= 1 for v in coords):
        raise ValueError("Некорректные координаты")

    if len(coords) == 4:
        # Если уже Bounding Box
        xc, yc, w, h = coords

    elif len(coords) >= 6 and len(coords) % 2 == 0:
        # Если полигон: x1 y1 x2 y2 x3 y3...
        xs = coords[0::2]
        ys = coords[1::2]

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        xc = (xmin + xmax) / 2
        yc = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin

    else:
        raise ValueError("Неправильный формат полигона")

    if w <= 0 or h <= 0:
        raise ValueError("Пустая рамка")

    return f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def main():
    if not SOURCE.is_dir():
        raise SystemExit(f"Не найдена папка: {SOURCE}")

    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(
            f"Выходная папка уже содержит файлы: {OUTPUT}"
        )

    prepared = {}
    total = 0

    # Сначала проверяем всю разметку, ничего не изменяя
    for split in SPLITS:
        images_dir = SOURCE / split / "images"
        labels_dir = SOURCE / split / "labels"

        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise SystemExit(f"Не найдены папки для {split}")

        images = [
            p for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
        labels = list(labels_dir.glob("*.txt"))

        if {p.stem for p in images} != {p.stem for p in labels}:
            raise SystemExit(
                f"{split}: изображения и TXT не совпадают"
            )

        prepared[split] = {}

        for label in labels:
            converted = []

            for number, line in enumerate(
                label.read_text(encoding="utf-8-sig").splitlines(), 1
            ):
                if not line.strip():
                    continue

                try:
                    converted.append(convert_annotation(line))
                except ValueError as exc:
                    raise SystemExit(
                        f"Ошибка {label.name}, строка {number}: {exc}"
                    )

            prepared[split][label.name] = converted
            total += len(converted)

    # После успешной проверки создаём новую копию
    for split in SPLITS:
        out_images = OUTPUT / split / "images"
        out_labels = OUTPUT / split / "labels"

        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        src_images = SOURCE / split / "images"

        image_count = 0

        for image in src_images.iterdir():
            if image.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".webp"
            }:
                copy2(image, out_images / image.name)
                image_count += 1

        for name, lines in prepared[split].items():
            (out_labels / name).write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8"
            )

        print(
            f"{split}: {image_count} изображений, "
            f"{len(prepared[split])} TXT"
        )

    (OUTPUT / "data.yaml").write_text(
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names: ['spillage']\n",
        encoding="utf-8"
    )

    print(f"\nГотово! Преобразовано рамок: {total}")
    print(f"Результат: {OUTPUT}")
    print("Оригинальный датасет не изменён.")


if __name__ == "__main__":
    main()