"""Read-only quality check for two new YOLO detection datasets.

Run from the industrial-vision project root:
    python scripts/check_new_datasets.py

Does not modify any images, labels, YAML files or model weights.
"""

from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    ("fire_smoke_v3", ("fire", "smoke")),
    ("spills_filtered_v1", ("spillage",)),
]
SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_SAMPLES = 6


def sample_text(paths):
    names = [str(p) for p in paths[:MAX_SAMPLES]]
    return ", ".join(names) + (" ..." if len(paths) > MAX_SAMPLES else "")


def check_dataset(folder_name, expected_classes):
    root = PROJECT_ROOT / "datasets" / "new_sources" / folder_name
    print("\n" + "=" * 68)
    print(f"DATASET: {folder_name}")
    print(f"PATH: {root}")

    if not root.is_dir():
        print("ERROR: folder not found. Check where you extracted the archive.")
        return False

    yamls = sorted(root.glob("data*.yaml"))
    if not yamls:
        print("WARNING: data.yaml not found in dataset root")
    elif len(yamls) > 1:
        print(f"INFO: {len(yamls)} YAML files in root. Only one is needed: data.yaml")
    else:
        print(f"YAML: {yamls[0].name}")

    print("EXPECTED SOURCE CLASS IDS:")
    for class_id, class_name in enumerate(expected_classes):
        print(f"  {class_id} -> {class_name}")

    total_images = 0
    total_labels = 0
    total_missing = 0
    total_invalid = 0
    total_orphans = 0
    all_classes = Counter()
    looks_ok = True

    for split in SPLITS:
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            print(f"\n{split}: ERROR: expected {split}/images and {split}/labels")
            looks_ok = False
            continue

        images = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        labels = sorted(labels_dir.rglob("*.txt"))
        image_stems = Counter(p.stem for p in images)
        label_stems = Counter(p.stem for p in labels)
        missing = [p.name for p in images if p.stem not in label_stems]
        orphans = [p.name for p in labels if p.stem not in image_stems]
        repeated = [stem for stem, count in image_stems.items() if count > 1]
        counts = Counter()
        invalid = []
        empty = 0

        for label in labels:
            try:
                lines = label.read_text(encoding="utf-8-sig").splitlines()
            except (OSError, UnicodeError) as exc:
                invalid.append(f"{label.name}: cannot read ({exc})")
                continue
            if not any(line.strip() for line in lines):
                empty += 1
            for line_no, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    invalid.append(f"{label.name}:{line_no}: {len(parts)} fields, expected 5 (bbox)")
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, w, h = (float(v) for v in parts[1:])
                    valid = (
                        0 <= cls < len(expected_classes)
                        and 0 <= cx <= 1 and 0 <= cy <= 1
                        and 0 < w <= 1 and 0 < h <= 1
                        and cx - w / 2 >= -0.002 and cx + w / 2 <= 1.002
                        and cy - h / 2 >= -0.002 and cy + h / 2 <= 1.002
                    )
                except (ValueError, OverflowError):
                    valid = False
                if valid:
                    counts[cls] += 1
                else:
                    invalid.append(f"{label.name}:{line_no}: {' '.join(parts[:5])}")

        total_images += len(images)
        total_labels += len(labels)
        total_missing += len(missing)
        total_orphans += len(orphans)
        total_invalid += len(invalid)
        all_classes.update(counts)
        print(f"\n{split}: {len(images)} images | {len(labels)} labels | {empty} empty labels")
        print("  Boxes: " + ", ".join(f"{expected_classes[c]}={counts[c]}" for c in range(len(expected_classes))))
        if missing:
            print(f"  NOTE: {len(missing)} images have no TXT label (may be negative samples)")
            print(f"    Examples: {sample_text(missing)}")
        if orphans:
            print(f"  WARNING: {len(orphans)} label files have no matching image")
            print(f"    Examples: {sample_text(orphans)}")
        if repeated:
            print(f"  WARNING: {len(repeated)} repeated image stems")
            print(f"    Examples: {sample_text(repeated)}")
        if invalid:
            print(f"  ERROR: {len(invalid)} invalid label lines")
            for example in invalid[:MAX_SAMPLES]:
                print(f"    {example}")
            looks_ok = False

    print("\nTOTAL:")
    print(f"  Images: {total_images}, TXT: {total_labels}")
    print("  Bounding boxes: " + ", ".join(f"{expected_classes[c]}={all_classes[c]}" for c in range(len(expected_classes))))
    print(f"  No matching TXT: {total_missing}, orphan TXT: {total_orphans}, invalid lines: {total_invalid}")
    if total_images == 0 or not all_classes:
        print("  ERROR: no usable labeled samples found")
        looks_ok = False
    if looks_ok:
        print("  RESULT: basic structure and label format look OK (visual review still needed)")
    return looks_ok


def main():
    print("New source datasets audit — read only")
    outcomes = [check_dataset(name, classes) for name, classes in DATASETS]
    print("\n" + "=" * 68)
    if all(outcomes):
        print("DONE: basic checks passed. Next: visually review boxes, de-duplicate and remap classes.")
    else:
        print("CHECK REQUIRED: one or more dataset structures or labels need attention.")


if __name__ == "__main__":
    main()
