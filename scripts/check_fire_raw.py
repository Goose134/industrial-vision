"""Проверка СЫРЫХ предсказаний YOLO по ВСЕМ классам.

Из корня проекта:
    python scripts/check_fire_raw.py --video "data/Firehouse.mp4" --time 4

При необходимости можно передать --image путь/к/картинке.jpg
или --model models/unified/unified640.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "runs/unified/unified_50_epochs/weights/best.pt"


def path_in_project(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверяем все классы YOLO без фильтров проекта")
    parser.add_argument("--video", default="data/Firehouse.mp4")
    parser.add_argument("--time", type=float, default=4.0, help="Время кадра в секундах")
    parser.add_argument("--image", help="Вместо видео проверить отдельное изображение")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    args = parser.parse_args()

    model_path = path_in_project(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    if args.image:
        source = path_in_project(args.image)
        frame = cv2.imread(str(source))
        if frame is None:
            raise RuntimeError(f"Не удалось прочитать изображение: {source}")
    else:
        source = path_in_project(args.video)
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {source}")
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Не удалось получить кадр на {args.time} секунде")
        finally:
            cap.release()

    model = YOLO(str(model_path), task="detect")
    print("Модель:", model_path)
    print("Все имена классов:", model.names)
    print("Кадр:", source)
    print("Порог:", args.conf, "размер:", args.imgsz)

    # Здесь НЕТ фильтра по fire/smoke, ByteTrack и дополнительных NMS проекта.
    result = model.predict(
        frame,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=0.7,
        agnostic_nms=False,
        max_det=100,
        verbose=False,
    )[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        print("YOLO не вернула НИ ОДНОГО обнаружения ни по одному классу.")
    else:
        found = []
        for b in boxes:
            cls_id = int(b.cls.item())
            class_name = str(result.names[cls_id])
            conf = float(b.conf.item())
            found.append((conf, cls_id, class_name))
        found.sort(reverse=True)
        print(f"Всего обнаружений всех классов: {len(found)}")
        print("Топ-30 (уверенность, ID класса, имя):")
        for conf, cls_id, name in found[:30]:
            print(f"  {conf:.4f}  {cls_id:2d}  {name}")
        fire_smoke = [row for row in found if row[2].strip().lower() in ("fire", "smoke")]
        print(f"Из них fire/smoke: {len(fire_smoke)}")

    output = ROOT / "runs" / "diagnostics" / "raw_firecheck.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), result.plot())
    print("Размеченный кадр сохранён:", output)


if __name__ == "__main__":
    main()
