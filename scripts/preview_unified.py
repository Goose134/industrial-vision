
from pathlib import Path
import random
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "unified"
OUTPUT = ROOT / "output" / "dataset_preview"

CLASSES = [
    "person", "car", "truck", "bus", "motorcycle",
    "helmet", "vest", "fire", "smoke", "spill"
]

random.seed(42)
OUTPUT.mkdir(parents=True, exist_ok=True)

images_dir = DATASET / "images" / "train"
labels_dir = DATASET / "labels" / "train"

# Собираем изображения по классам
by_class = {i: [] for i in range(len(CLASSES))}

for label in labels_dir.glob("*.txt"):
    class_ids = set()

    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            class_ids.add(int(parts[0]))

    for class_id in class_ids:
        if class_id in by_class:
            by_class[class_id].append(label)


def draw_image(label):
    image_path = None

    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = images_dir / (label.stem + ext)
        if candidate.exists():
            image_path = candidate
            break

    if image_path is None:
        return None

    # Поддержка путей с русскими символами Windows
    raw = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    if image is None:
        return None

    height, width = image.shape[:2]

    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:])

        x1 = int((xc - w / 2) * width)
        y1 = int((yc - h / 2) * height)
        x2 = int((xc + w / 2) * width)
        y2 = int((yc + h / 2) * height)

        cv2.rectangle(
            image, (x1, y1), (x2, y2),
            (0, 255, 0), 2
        )

        cv2.putText(
            image, CLASSES[class_id],
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )

    return image


for class_id, name in enumerate(CLASSES):
    candidates = by_class[class_id]
    selected = random.sample(
        candidates, min(3, len(candidates))
    )

    for index, label in enumerate(selected, 1):
        image = draw_image(label)

        if image is not None:
            destination = OUTPUT / f"{name}_{index}.jpg"

            success, encoded = cv2.imencode(".jpg", image)

            if success:
                encoded.tofile(str(destination))

    print(f"{name}: проверочные изображения сохранены")

print("\nГотово:", OUTPUT)
