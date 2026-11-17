"""Диагностика огня и дыма: сравнение best.pt и ONNX на одних кадрах.

Запуск из корня industrial-vision:
    python scripts/diagnose_fire_smoke.py --video "data/fire_test.mp4"

Скрипт не меняет main.py и configuration.yaml. Нужны уже установленные
ultralytics, opencv-python и PyYAML из окружения проекта.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PT = PROJECT_ROOT / "runs" / "unified" / "unified_50_epochs" / "weights" / "best.pt"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "configuration.yaml"
CLASSES = {"fire", "smoke"}
COLORS = {"fire": (0, 0, 255), "smoke": (0, 255, 255)}


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_default_onnx(config_file: Path) -> Path:
    if not config_file.is_file():
        raise FileNotFoundError(f"Нет конфигурации: {config_file}")
    with config_file.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return resolve_path(str(config["models"]["detector_path"]))


def make_preview(frame, detections, title: str, max_width: int = 740, show_missing: bool = True):
    h, w = frame.shape[:2]
    scale = min(1.0, max_width / max(1, w))
    target = (max(1, int(w * scale)), max(1, int(h * scale)))
    canvas = cv2.resize(frame, target, interpolation=cv2.INTER_AREA) if scale < 1 else frame.copy()
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1] - 1, 700), 32), (10, 10, 10), -1)
    cv2.putText(canvas, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    for item in detections:
        x1, y1, x2, y2 = item["bbox"]
        x1, y1, x2, y2 = (round(v * scale) for v in (x1, y1, x2, y2))
        color = COLORS[item["class"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f'{item["class"]} {item["confidence"]:.2f} area:{item["area_pct"]:.0f}%'
        cv2.putText(canvas, label, (x1, max(48, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 2)
    if show_missing and not detections:
        cv2.putText(canvas, "NO FIRE / SMOKE", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return canvas


def detect(model: YOLO, frame, size: int, conf: float, iou: float):
    result = model.predict(
        source=frame,
        imgsz=size,
        conf=conf,
        iou=iou,
        agnostic_nms=False,
        verbose=False,
    )[0]
    h, w = frame.shape[:2]
    matches = []
    if result.boxes is None:
        return matches
    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = str(result.names[class_id])
        if class_name not in CLASSES:
            continue
        bbox = tuple(float(v) for v in box.xyxy[0].tolist())
        x1, y1, x2, y2 = bbox
        area_pct = 100.0 * max(0.0, x2-x1) * max(0.0, y2-y1) / max(1, w*h)
        matches.append({
            "class": class_name,
            "confidence": float(box.conf.item()),
            "bbox": bbox,
            "area_pct": area_pct,
        })
    return sorted(matches, key=lambda item: item["confidence"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Сравнить обнаружение огня/дыма напрямую через PT и ONNX")
    parser.add_argument("--video", required=True, help="Путь к проверяемому видео")
    parser.add_argument("--pt", default=str(DEFAULT_PT), help="Путь к весам best.pt")
    parser.add_argument("--onnx", default=None, help="Путь к ONNX; по умолчанию из configuration.yaml")
    parser.add_argument("--size", type=int, default=640, help="Размер входа модели (для ONNX должен совпадать с экспортом)")
    parser.add_argument("--conf", type=float, default=0.01, help="Диагностический порог, по умолчанию 0.01")
    parser.add_argument("--iou", type=float, default=0.70, help="Порог встроенного YOLO NMS, по умолчанию 0.70")
    parser.add_argument("--interval", type=float, default=2.0, help="Интервал между проверяемыми кадрами, секунды")
    parser.add_argument("--samples", type=int, default=8, help="Максимальное число кадров")
    args = parser.parse_args()

    if args.samples < 1 or args.interval <= 0 or args.size < 32 or not 0 < args.conf < 1:
        parser.error("Проверьте аргументы samples, interval, size и conf")

    video_path = resolve_path(args.video)
    pt_path = resolve_path(args.pt)
    onnx_path = resolve_path(args.onnx) if args.onnx else get_default_onnx(DEFAULT_CONFIG)
    for path in (video_path, pt_path, onnx_path):
        if not path.is_file():
            raise FileNotFoundError(f"Файл не найден: {path}")

    output = PROJECT_ROOT / "runs" / "diagnostics" / video_path.stem
    output.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not 0.1 < fps < 1000:
        fps = 30.0
    sample_step = max(1, round(fps * args.interval))

    print(f"Видео: {video_path}\nPT: {pt_path}\nONNX: {onnx_path}")
    print(f"Порог диагностики: {args.conf}, imgsz={args.size}, iou={args.iou}")
    print("Сравниваем одни и те же кадры без ByteTrack, NMS проекта и его class_confidence.")
    models = {"PT": YOLO(str(pt_path), task="detect"), "ONNX": YOLO(str(onnx_path), task="detect")}
    rows = []
    frame_number = 0
    sample_number = 0

    try:
        while sample_number < args.samples:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_number % sample_step != 0:
                frame_number += 1
                continue
            current_sec = frame_number / fps
            sample_number += 1
            print(f"\nКадр {sample_number}, время {current_sec:.1f} сек")
            original_preview = make_preview(frame, [], "ORIGINAL", show_missing=False)
            for model_name, model in models.items():
                found = detect(model, frame, args.size, args.conf, args.iou)
                summary = ", ".join(
                    f'{p["class"]} {p["confidence"]:.2f} area:{p["area_pct"]:.0f}%'
                    for p in found[:7]
                ) or "нет обнаружений"
                print(f"  {model_name}: {summary}")
                preview = make_preview(frame, found, model_name)
                cv2.imwrite(str(output / f"{sample_number:02d}_{model_name.lower()}.jpg"), preview)
                for p in found:
                    rows.append({
                        "time_sec": round(current_sec, 3), "model": model_name,
                        "class": p["class"], "confidence": round(p["confidence"], 5),
                        "frame_area_percent": round(p["area_pct"], 2),
                        "x1": round(p["bbox"][0], 1), "y1": round(p["bbox"][1], 1),
                        "x2": round(p["bbox"][2], 1), "y2": round(p["bbox"][3], 1),
                    })
            cv2.imwrite(str(output / f"{sample_number:02d}_original.jpg"), original_preview)
            frame_number += 1
    finally:
        cap.release()

    csv_file = output / "detections.csv"
    fieldnames = ["time_sec", "model", "class", "confidence", "frame_area_percent", "x1", "y1", "x2", "y2"]
    with csv_file.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nГотово. Кадров проверено: {sample_number}. Результаты: {output}")
    print("Смотрите файлы *_pt.jpg, *_onnx.jpg и detections.csv")


if __name__ == "__main__":
    main()
