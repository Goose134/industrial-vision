
from pathlib import Path
from collections import defaultdict
import hashlib
import shutil

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "unified"
BACKUP = ROOT / "datasets" / "unified_duplicates_backup"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def file_hash(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def normalized_labels(path):
    """Сравниваем разметку без учёта порядка строк."""
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    return sorted(lines)


def main():
    removed = 0
    conflicts = 0

    for split in ("train", "val", "test"):
        image_dir = DATASET / "images" / split
        label_dir = DATASET / "labels" / split

        groups = defaultdict(list)

        for image in image_dir.iterdir():
            if image.suffix.lower() in EXTENSIONS:
                groups[file_hash(image)].append(image)

        for group in groups.values():
            if len(group) < 2:
                continue

            # Разбиваем одинаковые изображения по разметке
            by_labels = defaultdict(list)

            for image in group:
                label = label_dir / (image.stem + ".txt")

                if not label.is_file():
                    print("Нет разметки:", image)
                    conflicts += 1
                    continue

                key = tuple(normalized_labels(label))
                by_labels[key].append((image, label))

            if len(by_labels) > 1:
                conflicts += 1
                print("Разная разметка у одинаковых изображений:")
                for items in by_labels.values():
                    print(" ", [img.name for img, _ in items])
                continue

            # Первую копию оставляем
            for items in by_labels.values():
                for image, label in items[1:]:
                    backup_images = BACKUP / "images" / split
                    backup_labels = BACKUP / "labels" / split

                    backup_images.mkdir(parents=True, exist_ok=True)
                    backup_labels.mkdir(parents=True, exist_ok=True)

                    shutil.move(str(image), str(backup_images / image.name))
                    shutil.move(str(label), str(backup_labels / label.name))

                    removed += 1

    print("\n=== РЕЗУЛЬТАТ ===")
    print("Перемещено дубликатов:", removed)
    print("Конфликтных групп:", conflicts)
    print("Резервная копия:", BACKUP)


if __name__ == "__main__":
    main()
