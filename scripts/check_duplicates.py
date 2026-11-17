from pathlib import Path
from collections import defaultdict
import hashlib

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "unified"

hashes = defaultdict(list)

for split in ("train", "val", "test"):
    folder = DATASET / "images" / split

    for image in folder.iterdir():
        if image.suffix.lower() not in (
            ".jpg", ".jpeg", ".png", ".webp"
        ):
            continue

        # Вычисляем хеш содержимого изображения
        h = hashlib.sha256()

        with open(image, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)

        hashes[h.hexdigest()].append((split, image.name))

duplicates = []

for files in hashes.values():
    splits = {split for split, _ in files}

    if len(splits) > 1:
        duplicates.append(files)

print("Всего уникальных файлов:", len(hashes))
print("Групп дубликатов между выборками:", len(duplicates))

for group in duplicates[:10]:
    print(group)