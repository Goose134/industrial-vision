
from pathlib import Path
from time import perf_counter
import csv

import cv2
import numpy as np
from ultralytics import YOLO


# Корневая папка проекта
ROOT = Path(__file__).resolve().parents[1]

# Модели для сравнения
PYTORCH_MODEL = ROOT / "yolo26n.pt"
ONNX_MODEL = ROOT / "yolo26n.onnx"

# Тестовое видео
VIDEO_PATH = ROOT / "data" / "ppe_test.mp4"

# Параметры одинаковы для обеих моделей
IMAGE_SIZE = 320
WARMUP_FRAMES = 10
MEASURE_FRAMES = 300


def benchmark(model_path, name):
    print(f"\n===== {name} =====")

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"Не удалось открыть видео: {VIDEO_PATH}"
        )

    times = []
    total_detections = 0

    try:
        total_frames = WARMUP_FRAMES + MEASURE_FRAMES

        for i in range(total_frames):
            success, frame = cap.read()

            # Если видео закончилось, начинаем сначала
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = cap.read()

                if not success:
                    raise RuntimeError(
                        "Не удалось прочитать кадр видео"
                    )

            start = perf_counter()

            # Одинаковый режим YOLO + Tracking
            results = model.track(
                frame,
                imgsz=IMAGE_SIZE,
                conf=0.25,
                device="cpu",
                persist=True,
                verbose=False
            )

            elapsed = (perf_counter() - start) * 1000

            # Первые кадры не учитываем
            if i >= WARMUP_FRAMES:
                times.append(elapsed)
                total_detections += len(results[0].boxes)

    finally:
        cap.release()

    times = np.array(times)

    mean_ms = float(np.mean(times))
    p95_ms = float(np.percentile(times, 95))
    max_ms = float(np.max(times))

    fps = 1000.0 / mean_ms

    result = {
        "model": name,
        "image_size": IMAGE_SIZE,
        "frames": len(times),
        "fps": round(fps, 2),
        "mean_ms": round(mean_ms, 2),
        "p95_ms": round(p95_ms, 2),
        "max_ms": round(max_ms, 2),
        "model_size_mb": round(
            model_path.stat().st_size / (1024 * 1024), 2
        ),
        "detections": total_detections
    }

    print(f"Кадров: {result['frames']}")
    print(f"FPS: {result['fps']}")
    print(f"Средняя задержка: {result['mean_ms']} мс")
    print(f"P95: {result['p95_ms']} мс")
    print(f"Максимум: {result['max_ms']} мс")
    print(f"Размер модели: {result['model_size_mb']} МБ")
    print(f"Обнаружений: {result['detections']}")

    return result


def main():
    print("===== PYTORCH VS ONNX BENCHMARK =====")
    print(f"Разрешение: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print("Устройство: CPU")

    pytorch_result = benchmark(
        PYTORCH_MODEL,
        "PyTorch"
    )

    onnx_result = benchmark(
        ONNX_MODEL,
        "ONNX Runtime"
    )

    results = [pytorch_result, onnx_result]

    output_path = ROOT / "output" / "benchmark_onnx.csv"
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    speedup = (
        onnx_result["fps"] / pytorch_result["fps"]
    )

    print("\n===== ИТОГОВОЕ СРАВНЕНИЕ =====")

    print(f"PyTorch: {pytorch_result['fps']} FPS")
    print(f"ONNX: {onnx_result['fps']} FPS")
    print(f"Отношение скоростей ONNX / PyTorch: {speedup:.2f}x")

    print(f"\nРезультаты сохранены: {output_path}")


if __name__ == "__main__":
    main()
