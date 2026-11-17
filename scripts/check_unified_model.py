
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "unified" / "unified.onnx"
VIDEO_PATH = ROOT / "data" / "spi_test.mp4"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    print("Загружаем единую модель...")

    model = YOLO(str(MODEL_PATH), task="detect")

    print("\nКлассы модели:")
    for class_id, name in model.names.items():
        print(f"{class_id}: {name}")

    if not VIDEO_PATH.exists():
        print("\nВидео не найдено. Классы проверены.")
        return

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    success, frame = cap.read()
    cap.release()

    if not success:
        print("Не удалось прочитать кадр видео.")
        return

    result = model.predict(
        source=frame,
        imgsz=320,
        conf=0.25,
        verbose=False
    )[0]

    print("\nОбнаружено объектов:", len(result.boxes))

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        print(
            f"{result.names[class_id]}: "
            f"{confidence:.2f}"
        )

    print("\nПроверка завершена!")


if __name__ == "__main__":
    main()
