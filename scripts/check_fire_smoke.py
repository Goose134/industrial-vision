
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def main():
    # Определяем пути относительно корня проекта
    project_root = Path(__file__).resolve().parent.parent

    model_path = (
        project_root / "models" / "fire_smoke" / "best.pt"
    )

    video_path = project_root / "data" / "fire_test.mp4"

    # Проверяем наличие файлов
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Модель не найдена: {model_path}"
        )

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Тестовое видео не найдено: {video_path}"
        )

    # Используем GPU, если CUDA доступна
    device = 0 if torch.cuda.is_available() else "cpu"
    print("Устройство:", device)

    # Загружаем модель
    model = YOLO(str(model_path))

    print("Модель Fire/Smoke загружена")
    print("Классы модели:", model.names)

    # Открываем видео
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видео")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Видео завершено")
                break

            # Запускаем детекцию дыма и огня
            result = model.predict(
                source=frame,
                conf=0.35,
                imgsz=640,
                device=device,
                verbose=False
            )[0]

            fire_count = 0
            smoke_count = 0

            # Обрабатываем обнаруженные объекты
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = str(result.names[class_id]).lower()
                confidence = float(box.conf.item())

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                # Цвета OpenCV задаются в формате BGR
                if class_name == "fire":
                    color = (0, 0, 255)
                    fire_count += 1

                elif class_name == "smoke":
                    color = (0, 255, 255)
                    smoke_count += 1

                else:
                    continue

                # Рисуем границы объекта
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # Подпись обнаруженного объекта
                label = f"{class_name}: {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

            # Выводим количество обнаружений
            cv2.putText(
                frame,
                f"Fire: {fire_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"Smoke: {smoke_count}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            cv2.imshow(
                "Industrial Vision - Fire and Smoke",
                frame
            )

            # Нажимаем Q для завершения
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Проверка завершена")


if __name__ == "__main__":
    main()
