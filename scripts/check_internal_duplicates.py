from pathlib import Path
from collections import defaultdict
import hashlib

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "unified"

for split in ("train", "val", "test"):
    folder = DATASET / "images" / split
    hashes = defaultdict(list)

    for image in folder.iterdir():
        if image.suffix.lower() not in (
            ".jpg", ".jpeg", ".png", ".webp"
        ):
            continue

        digest = hashlib.sha256(
            image.read_bytes()
        ).hexdigest()

        hashes[digest].append(image.name)

    duplicates = [
        files for files in hashes.values()
        if len(files) > 1
    ]

    extra = sum(len(files) - 1 for files in duplicates)

    print(f"\n=== {split.upper()} ===")
    print("Всего изображений:", sum(map(len, hashes.values())))
    print("Уникальных:", len(hashes))
    print("Лишних копий:", extra)
    print("Групп повторов:", len(duplicates))

    for group in duplicates[:3]:
        print("Пример:", group)