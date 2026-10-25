
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from src.inference.detector import Detector
from src.ppe.ppe_detector import PPEDetector
from src.fire_smoke.fire_detector import FireSmokeDetector
from src.spills.spill_detector import SpillDetector


# Сколько кадров измеряем
MEASURE_FRAMES = 300

# Сколько кадров используем для прогрева
WARMUP_FRAMES = 10


def synchronize_gpu():
    """
    Дожидаемся окончания операций CUDA,
    чтобы правильно измерять время.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_benchmark(name, config, project_root):
    """
    Измеряет время обработки видео
    для одной конфигурации нейросетей.
    """

    print(f"\n===== Конфигурация {name} =====")

    # Определяем нужные модели
    use_ppe = name in ("B", "C", "D")
    use_fire = name in ("C", "D")
    use_spill = name == "D"

    # Загружаем основной детектор
    model_config = config["models"]

    detector = Detector(
        model_path=model_config["detector_path"],
        confidence=model_config["person_confidence"],
        image_size=model_config["image_size"]
    )

    # PPE-модель
    ppe_detector = None

    if use_ppe:
        ppe_config = config["ppe"]

        ppe_detector = PPEDetector(
            model_path=project_root / ppe_config["model_path"],
            confidence=ppe_config["confidence"],
            min_person_height=ppe_config["min_person_height"]
        )

    # Детектор огня и дыма
    fire_detector = None

    if use_fire:
        fire_config = config["fire_smoke"]

        fire_detector = FireSmokeDetector(
            model_path=project_root / fire_config["model_path"],
            confidence=fire_config["confidence"],
            image_size=fire_config["image_size"]
        )

    # Детектор разливов
    spill_detector = None

    if use_spill:
        spill_config = config["spills"]

        spill_detector = SpillDetector(
            model_path=project_root / spill_config["model_path"],
            confidence=spill_config["confidence"],
            image_size=spill_config["image_size"]
        )

    # Берём видео из configuration.yaml
    source = config["camera"]["source"]

    if not isinstance(source, str) or "://" in source:
        raise ValueError(
            "Для Benchmark укажи локальный MP4-файл"
        )

    video_path = project_root / source

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Не удалось открыть видео: {video_path}"
        )

    frame_times = []
    processed = 0

    try:
        for index in range(
            WARMUP_FRAMES + MEASURE_FRAMES
        ):
            success, frame = cap.read()

            if not success:
                break

            # Прогреваем модели на начальных кадрах
            # Время чтения видео не учитываем
            synchronize_gpu()
            start = time.perf_counter()

            # Базовая YOLO и Tracking
            detections = detector.track(frame)

            # PPE проверяется для каждого человека
            if ppe_detector is not None:
                for det in detections:
                    if (
                        det.class_name == "person"
                        and det.track_id is not None
                    ):
                        ppe_detector.check_person(
                            frame, det
                        )

            # Проверка огня и дыма
            if fire_detector is not None:
                fire_detector.detect(frame)

            # Проверка разливов
            if spill_detector is not None:
                spill_detector.detect(frame)

            synchronize_gpu()
            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            # Первые кадры исключаем из измерений
            if index >= WARMUP_FRAMES:
                frame_times.append(elapsed_ms)
                processed += 1

    finally:
        cap.release()

    if not frame_times:
        raise RuntimeError(
            "Не удалось обработать измеряемые кадры"
        )

    # Статистика задержек
    total_seconds = sum(frame_times) / 1000.0

    fps = processed / total_seconds
    mean_ms = float(np.mean(frame_times))
    p95_ms = float(np.percentile(frame_times, 95))

    print(f"Кадров:       {processed}")
    print(f"FPS:          {fps:.2f}")
    print(f"Среднее:      {mean_ms:.2f} мс")
    print(f"P95:          {p95_ms:.2f} мс")

    return {
        "configuration": name,
        "frames": processed,
        "fps": round(fps, 2),
        "mean_ms": round(mean_ms, 2),
        "p95_ms": round(p95_ms, 2)
    }


def main():
    project_root = Path(__file__).resolve().parent.parent

    config_path = (
        project_root / "configs" / "configuration.yaml"
    )

    with open(
        config_path, "r", encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)

    print("===== INDUSTRIAL VISION BENCHMARK =====")

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Устройство: CPU")

    results = []

    # Последовательно проверяем 4 конфигурации
    for name in ("A",):
        result = run_benchmark(
            name,
            config,
            project_root
        )
        results.append(result)

    # Сохраняем результаты
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "benchmark.csv"

    with open(
        csv_path, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "configuration",
                "frames",
                "fps",
                "mean_ms",
                "p95_ms"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n===== ИТОГОВАЯ ТАБЛИЦА =====")

    for item in results:
        print(
            f"{item['configuration']}: "
            f"FPS={item['fps']}, "
            f"Mean={item['mean_ms']} ms, "
            f"P95={item['p95_ms']} ms"
        )

    print("\nРезультаты сохранены:", csv_path)


if __name__ == "__main__":
    main()
