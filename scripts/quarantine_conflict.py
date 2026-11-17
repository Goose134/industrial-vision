from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
dataset = root / "datasets" / "unified"
backup = root / "datasets" / "unified_conflicts_backup"

prefix = "spills_11_mp4-5_jpg.rf."

for split in ("train", "val", "test"):
    images = dataset / "images" / split
    labels = dataset / "labels" / split

    for image in images.glob(prefix + "*.jpg"):
        label = labels / (image.stem + ".txt")

        if not label.exists():
            raise FileNotFoundError(label)

        backup_images = backup / "images" / split
        backup_labels = backup / "labels" / split

        backup_images.mkdir(parents=True, exist_ok=True)
        backup_labels.mkdir(parents=True, exist_ok=True)

        shutil.move(str(image), str(backup_images / image.name))
        shutil.move(str(label), str(backup_labels / label.name))

        print("Перемещено:", image.name)

print("Готово")