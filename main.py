
from pathlib import Path

import cv2
import yaml

from src.capture.video_capture import VideoSource
from src.capture.fps_meter import FPSMeter
from src.inference.detector import Detector


def main():

    # Определяем корневую папку проекта
    project_root = Path(__file__).resolve().parent

    # Загружаем настройки
    config_path = project_root / "configs" / "configuration.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Настройки камеры
    camera_config = config["camera"]
    source = camera_config["source"]

    # Преобразуем относительный путь к абсолютному
    if isinstance(source, str) and "://" not in source:
        source = str(project_root / source)

    # Создаём объект видеопотока
    video = VideoSource(
        source=source,
        reconnect_sec=camera_config["reconnect_sec"]
    )

    # Настройки нейросети
    model_config = config["models"]

    # Создаём детектор
    detector = Detector(
        model_path=model_config["detector_path"],
        confidence=model_config["person_confidence"],
        image_size=model_config["image_size"]
    )

    # Счётчик FPS
    fps_meter = FPSMeter(window=30)

    try:

        # Открываем видео
        video.open()

        print("Источник видео успешно открыт")
        print("Нейросеть YOLO загружена")

        while True:

            # Получаем кадр
            frame = video.read()

            # Проверяем конец видео
            if frame is None:
                print("Видеофайл завершён")
                break

            # Запускаем нейросеть
            detections = detector.predict(frame)

            # Отображаем каждый найденный объект
            for det in detections:

                # Координаты
                x1, y1 = det.x1, det.y1
                x2, y2 = det.x2, det.y2

                # Рисуем прямоугольник
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Подпись класса и уверенности
                label = (
                    f"{det.class_name}: "
                    f"{det.confidence:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # Считаем FPS с учётом нейросети
            fps = fps_meter.tick()

            # Показываем FPS
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Количество обнаруженных объектов
            cv2.putText(
                frame,
                f"Objects: {len(detections)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Показываем изображение
            cv2.imshow(
                "Industrial Vision - YOLO Detection",
                frame
            )

            # Выход по Q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Остановка пользователем")
                break

    except Exception as error:
        print(f"Ошибка: {error}")

    finally:

        # Закрываем видеопоток
        video.close()

        # Закрываем окна
        cv2.destroyAllWindows()

        print("Программа завершена")


if __name__ == "__main__":
    main()
