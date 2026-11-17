"""Read-only exact-duplicate and annotation-conflict audit before YOLO fine-tuning.

Place in industrial-vision/scripts/check_finetune_duplicates.py and run:
    python scripts/check_finetune_duplicates.py

Original dataset files and model weights are NEVER modified.
"""

from collections import defaultdict, Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
REPORT = ROOT / "runs" / "diagnostics" / "finetune_duplicates"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# (source name, folder, split folder -> canonical split, local class ID -> unified ID)
SOURCES = (
    ("old_unified", DATASETS / "unified", {"train": "train", "val": "val", "test": "test"}, None),
    ("new_fire_smoke", DATASETS / "new_sources" / "fire_smoke_v3",
     {"train": "train", "valid": "val", "test": "test"}, {0: 7, 1: 8}),
    ("new_spills", DATASETS / "new_sources" / "spills_filtered_bbox",
     {"train": "train", "valid": "val", "test": "test"}, {0: 9}),
)


@dataclass(frozen=True)
class ImageRecord:
    source: str
    split: str
    path: Path
    labels: tuple | None


def image_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_labels(label_path: Path, mapping: dict | None) -> tuple | None:
    """Canonicalize classes + bbox data to compare identical images' annotations.

    None means a missing/invalid label; () means a valid empty label.
    A different ordering of the same boxes is treated as equivalent.
    """
    if not label_path.is_file():
        return None
    entries = []
    try:
        for row in label_path.read_text(encoding="utf-8-sig").splitlines():
            if not row.strip():
                continue
            values = row.split()
            if len(values) != 5:
                return None
            class_id = int(values[0])
            if mapping is not None:
                class_id = mapping[class_id]
            values = tuple(round(float(v), 4) for v in values[1:])
            entries.append((class_id, *values))
        return tuple(sorted(entries))
    except (OSError, UnicodeError, ValueError, KeyError):
        return None


def audit():
    hashes = defaultdict(list)
    counts = Counter()
    missing_folders = []
    print("ПРОВЕРКА ТОЧНЫХ ДУБЛИКАТОВ ДЛЯ ДОOБУЧЕНИЯ")
    print("Старые и новые данные только читаются; ничего не удаляется.\n")

    for source, folder, splits, mapping in SOURCES:
        if not folder.is_dir():
            missing_folders.append(str(folder))
            continue
        for split_folder, canonical_split in splits.items():
            if source == "old_unified":
                images_dir = folder / "images" / split_folder
                labels_dir = folder / "labels" / split_folder
            else:
                images_dir = folder / split_folder / "images"
                labels_dir = folder / split_folder / "labels"
            if not images_dir.is_dir() or not labels_dir.is_dir():
                missing_folders.append(f"{images_dir} или {labels_dir}")
                continue
            images = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            print(f"{source:16s} {canonical_split:5s} {len(images):>7} изображений")
            for image in images:
                label_path = labels_dir / f"{image.stem}.txt"
                labels = normalized_labels(label_path, mapping)
                try:
                    digest = image_hash(image)
                except OSError as exc:
                    raise SystemExit(f"Ошибка чтения {image}: {exc}")
                hashes[digest].append(ImageRecord(source, canonical_split, image, labels))
                counts["images"] += 1
                if labels is None:
                    counts["missing_or_invalid_labels"] += 1

    if missing_folders:
        print("\nОШИБКА: отсутствуют папки:")
        for folder in missing_folders:
            print(" -", folder)
        print("Проверь расположение датасетов. Объединение пока не запускай.")
        return

    REPORT.mkdir(parents=True, exist_ok=True)
    duplicate_groups = [(digest, records) for digest, records in hashes.items() if len(records) > 1]
    duplicate_groups.sort(key=lambda item: (-len(item[1]), item[0]))

    cross_split = 0
    old_new = 0
    label_conflicts = 0
    duplicate_copies = 0

    report_csv = REPORT / "exact_duplicate_members.csv"
    with report_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["group", "sha256", "group_size", "cross_split", "old_and_new", "labels_differ", "dataset", "split", "image_path", "label_state"])
        for group_id, (digest, records) in enumerate(duplicate_groups, 1):
            split_risk = len({record.split for record in records}) > 1
            sources = {record.source for record in records}
            shared_old_new = "old_unified" in sources and len(sources) > 1
            labels_differ = len({record.labels for record in records}) > 1
            cross_split += int(split_risk)
            old_new += int(shared_old_new)
            label_conflicts += int(labels_differ)
            duplicate_copies += len(records) - 1
            for record in records:
                label_state = "MISSING/INVALID" if record.labels is None else f"{len(record.labels)} objects"
                writer.writerow([group_id, digest, len(records), int(split_risk), int(shared_old_new), int(labels_differ), record.source, record.split, str(record.path.relative_to(ROOT)), label_state])

    summary = [
        f"Всего изображений: {counts['images']}",
        f"Уникальных файлов по SHA256: {len(hashes)}",
        f"Групп точных дубликатов: {len(duplicate_groups)}",
        f"Лишних копий (в группах): {duplicate_copies}",
        f"Групп дубликатов между train/val/test: {cross_split}",
        f"Групп совпадений старого и нового датасетов: {old_new}",
        f"Групп дубликатов с различающейся разметкой: {label_conflicts}",
        f"Изображений без валидной bbox-разметки: {counts['missing_or_invalid_labels']}",
        f"Полный список: {report_csv}",
    ]
    (REPORT / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n" + "=" * 65)
    print("\n".join(summary))
    if cross_split or label_conflicts:
        print("\nВНИМАНИЕ: до объединения нужно убрать конфликты и пересечения между выборками.")
    print("\nПроверены только идентичные байты файлов. Изменённые/аугментированные копии могут быть не найдены.")


if __name__ == "__main__":
    audit()
