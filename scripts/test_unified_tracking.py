
from pathlib import Path

import cv2
import yaml

from src.inference.unified_detector import UnifiedDetector

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "unified" / "unified.onnx"
CONFIG_PATH = ROOT / "configs" / "configuration.yaml"

TRACKED_CLASSES = {
    "person", "car", "truck", "bus", "motorcycle"
}


def main():
    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Берём видео из конфигурации проекта
    source = config["camera"]["source"]

    if isinstance(source, str):
        if source.isdigit():
            source = int(source)
        elif "://" not in source:
            source = str(ROOT / source)

    video = cv2.VideoCapture(source)

    if not video.isOpened():
        raise RuntimeError(
            f"Не удалось открыть видео: {source}"
        )

    # Создаём единую модель
    detector = UnifiedDetector(
        model_path=MODEL_PATH,
        confidence=0.20,
        image_size=320
    )

    print("\nНачинаем проверку ByteTrack...")

    unique_ids = set()

    try:
        for frame_number in range(1, 61):
            success, frame = video.read()

            if not success:
                break

            # Одна нейросеть + ByteTrack
            detections = detector.track(frame)

            print(f"\nКадр {frame_number}")

            for det in detections:

                if det.class_name in TRACKED_CLASSES:
                    print(
                        f"{det.class_name} | "
                        f"ID: {det.track_id} | "
                        f"Confidence: {det.confidence:.2f}"
                    )

                    if det.track_id is not None:
                        unique_ids.add(det.track_id)

                else:
                    print(
                        f"{det.class_name} | "
                        f"Confidence: {det.confidence:.2f}"
                    )

    finally:
        video.release()

    print("\n=== РЕЗУЛЬТАТ ===")
    print("Уникальных Track ID:", len(unique_ids))
    print("Проверка завершена!")


if __name__ == "__main__":
    main()
