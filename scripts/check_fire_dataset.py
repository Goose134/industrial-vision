"""Проверить соответствие разметки fire/smoke и предсказаний YOLO на train/val.

Из корня industrial-vision:
    python scripts/check_fire_dataset.py
    python scripts/check_fire_dataset.py --per-class 5

Создаёт runs/diagnostics/fire_dataset_check/*.jpg (слева разметка, справа YOLO).
Не меняет веса, разметку или configuration.yaml.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import random

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {7: "fire", 8: "smoke"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
COLORS = {7: (20, 20, 245), 8: (40, 235, 235)}  # OpenCV BGR


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def find_dirs(dataset: Path, split: str) -> tuple[Path, Path]:
    candidates = (
        (dataset / split / "labels", dataset / split / "images"),
        (dataset / "labels" / split, dataset / "images" / split),
    )
    for label_dir, image_dir in candidates:
        if label_dir.is_dir() and image_dir.is_dir():
            return label_dir, image_dir
    raise FileNotFoundError(f"Не найдены папки {split}/labels и {split}/images в {dataset}")


def find_image(label_file: Path, labels: Path, images: Path) -> Path | None:
    relative = label_file.relative_to(labels)
    for extension in IMAGE_EXTS:
        path = images / relative.parent / (relative.stem + extension)
        if path.is_file():
            return path
    return None


def load_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    items = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, start=1):
            parts = line.split()
            if not parts:
                continue
            try:
                cls = int(parts[0])
                coords = list(map(float, parts[1:5]))
            except (ValueError, IndexError) as error:
                print(f"[WARN] Неверная строка {path.name}:{number}: {error}")
                continue
            if len(parts) != 5 or not all(0 <= v <= 1 for v in coords):
                print(f"[WARN] Проверь формат bbox: {path.name}:{number}")
                continue
            items.append((cls, *coords))
    return items


def open_unicode(path: Path):
    raw = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def save_unicode(path: Path, image) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError(f"Не удалось записать {path}")
    encoded.tofile(str(path))


def draw_label(image, cls: int, bbox: tuple[int, int, int, int], text: str):
    color = COLORS.get(cls, (255, 255, 255))
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, text, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def main():
    parser = argparse.ArgumentParser(description="Проверка fire/smoke на обучающих изображениях")
    parser.add_argument("--dataset", default="datasets/unified")
    parser.add_argument("--model", default="runs/unified/unified_50_epochs/weights/best.pt")
    parser.add_argument("--per-class", type=int, default=3, help="Изображений на класс для каждого split")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    args = parser.parse_args()
    if not 0 < args.conf < 1 or args.per_class < 1:
        parser.error("--conf должен быть в (0, 1), --per-class >= 1")

    dataset, weights = resolve(args.dataset), resolve(args.model)
    if not weights.is_file():
        raise FileNotFoundError(f"Весов модели нет: {weights}")
    output = ROOT / "runs" / "diagnostics" / "fire_dataset_check"
    output.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(42)
    model = YOLO(str(weights), task="detect")
    names = model.names
    print("Модель:", weights)
    print("Классы модели:", names)
    for id_, name in CLASSES.items():
        if str(names.get(id_, "")).lower() != name:
            raise RuntimeError(f"Ожидался класс {id_} = {name}, а модель содержит {names.get(id_)}")

    total = Counter()
    examples: list[tuple[str, int, Path, list]] = []
    for split in ("train", "val"):
        labels_dir, images_dir = find_dirs(dataset, split)
        candidates = {cls: [] for cls in CLASSES}
        for label_file in labels_dir.rglob("*.txt"):
            labels = load_labels(label_file)
            for cls, *_ in labels:
                if cls in CLASSES:
                    total[(split, cls, "boxes")] += 1
            found = set(cls for cls, *_ in labels) & set(CLASSES)
            for cls in found:
                total[(split, cls, "images")] += 1
            if not found:
                continue
            image_path = find_image(label_file, labels_dir, images_dir)
            if image_path is None:
                total[(split, "missing_image")] += 1
                continue
            for cls in found:
                candidates[cls].append((image_path, labels))
        for cls, name in CLASSES.items():
            print(f"{split:5s} {name:5s}: {total[(split,cls,'images')]} размеченных изображений, "
                  f"{total[(split,cls,'boxes')]} bbox")
            for img, labels in randomizer.sample(candidates[cls], min(args.per_class, len(candidates[cls]))):
                examples.append((split, cls, img, labels))
        if total[(split, "missing_image")]:
            print(f"[WARN] {split}: нет изображения для {total[(split,'missing_image')]} размеченных файлов")

    if not examples:
        print("\nНет примеров классов 7/8 в датасете! Проверь prepare_unified.py и unified/data.yaml.")
        return

    print("\nСравниваем разметку (слева) и предсказания best.pt (справа):")
    for number, (split, expected_cls, path, labels) in enumerate(examples, start=1):
        frame = open_unicode(path)
        if frame is None:
            print("[WARN] Не прочитано изображение:", path)
            continue
        h, w = frame.shape[:2]
        gt = frame.copy()
        for cls, cx, cy, bw, bh in labels:
            if cls not in CLASSES:
                continue
            x1 = round((cx - bw / 2) * w)
            y1 = round((cy - bh / 2) * h)
            x2 = round((cx + bw / 2) * w)
            y2 = round((cy + bh / 2) * h)
            draw_label(gt, cls, (x1, y1, x2, y2), f"GT {CLASSES[cls]}")

        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                               iou=0.7, verbose=False, max_det=200)[0]
        predicted = frame.copy()
        targets = []
        others = []
        for box in result.boxes:
            cls = int(box.cls.item())
            confidence = float(box.conf.item())
            if cls in CLASSES:
                targets.append((CLASSES[cls], confidence))
                draw_label(predicted, cls, tuple(map(int, box.xyxy[0].tolist())),
                           f"PRED {CLASSES[cls]} {confidence:.2f}")
            else:
                others.append((str(names[cls]), confidence))
        targets.sort(key=lambda item: item[1], reverse=True)
        others.sort(key=lambda item: item[1], reverse=True)
        print(f"{number:02d} {split:5s} GT={CLASSES[expected_cls]:5s} "
              f"pred={targets[:3] or 'нет'} другие={others[:2]} | {path.name}")

        # Не создаём гигантских картинок при высоком разрешении исходника.
        scale = min(1.0, 740.0 / max(1, w))
        width, height = max(1, round(w * scale)), max(1, round(h * scale))
        left = cv2.resize(gt, (width, height))
        right = cv2.resize(predicted, (width, height))
        for panel, title in ((left, "GROUND TRUTH (training labels)"),
                             (right, "MODEL PREDICTIONS")):
            cv2.rectangle(panel, (0, 0), (min(width - 1, 490), 28), (0, 0, 0), -1)
            cv2.putText(panel, title, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        joined = np.concatenate([left, right], axis=1)
        save_unicode(output / f"{number:02d}_{split}_{CLASSES[expected_cls]}.jpg", joined)

    print("\nГотово! Открой изображения:", output)
    print("Если слева неправильная разметка — ошибка подготовки данных.")
    print("Если слева правильно, а справа даже train не распознаётся — проблема в обучении/весах.")
    print("Если train распознаётся, а новые видео нет — проблема обобщения на новые сцены.")


if __name__ == "__main__":
    main()
