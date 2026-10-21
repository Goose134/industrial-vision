
from pathlib import Path
import cv2
import yaml

from src.capture.video_capture import VideoSource
from src.capture.fps_meter import FPSMeter


def main():
    # Определяем корень проекта
    project_root = Path(__file__).resolve().parent

    # Загружаем конфигурацию
    config_path = project_root / "configs" / "configuration.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Получаем параметры камеры
    camera_config = config["camera"]
    source = camera_config["source"]

    # Преобразуем относительный путь к видео в абсолютный
    if isinstance(source, str) and "://" not in source:
        source = str(project_root / source)

    # Создаём объект видеопотока
    video = VideoSource(
        source=source,
        reconnect_sec=camera_config["reconnect_sec"]
    )

    # Создаём счётчик FPS
    fps_meter = FPSMeter(window=30)

    try:
        # Открываем видео
        video.open()
        print("Источник видео успешно открыт")

        while True:
            # Получаем очередной кадр
            frame = video.read()

            # Конец видеофайла
            if frame is None:
                print("Видеофайл завершён")
                break

            # Рассчитываем FPS обработки
            fps = fps_meter.tick()

            # Добавляем значение FPS на изображение
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Показываем изображение
            cv2.imshow("Industrial Vision - Video Capture", frame)

            # Нажатие Q завершает программу
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Остановка по команде пользователя")
                break

    except RuntimeError as error:
        print(f"Ошибка видеопотока: {error}")

    finally:
        # Освобождаем ресурсы
        video.close()

        # Закрываем окна OpenCV
        cv2.destroyAllWindows()

        print("Программа завершена")


if __name__ == "__main__":
    main()
