
from pathlib import Path

import cv2
import yaml

from src.capture.video_capture import VideoSource
from src.capture.fps_meter import FPSMeter
from src.inference.detector import Detector

from src.zones.geometry import (
    normalized_to_pixels,
    bottom_center,
    point_in_polygon,
    draw_roi
)

from src.events.zone_events import ZoneEventLogic

from src.ppe.ppe_detector import PPEDetector
from src.events.ppe_events import PPEEventLogic

def main():

    # Корневая папка проекта
    project_root = Path(__file__).resolve().parent

    # Загружаем конфигурацию
    config_path = project_root / "configs" / "configuration.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Получаем путь к видео
    camera_config = config["camera"]
    source = camera_config["source"]

    if isinstance(source, str) and "://" not in source:
        source = str(project_root / source)

    # Создаём источник видео
    video = VideoSource(
        source=source,
        reconnect_sec=camera_config["reconnect_sec"]
    )

    # Создаём детектор с поддержкой Tracking
    model_config = config["models"]

    detector = Detector(
        model_path=model_config["detector_path"],
        confidence=model_config["person_confidence"],
        image_size=model_config["image_size"]
    )

    # Настройки контролируемой зоны
    roi_normalized = config["zones"]["restricted_01"]

    # Создаём обработчик событий
    event_config = config["events"]

    event_logic = ZoneEventLogic(
        n=event_config["confirmation_n"],
        m=event_config["confirmation_m"],
        cooldown_sec=event_config["cooldown_sec"]
    )

    # Получаем настройки PPE из configuration.yaml
    ppe_config = config["ppe"]

    # Путь к нашей обученной модели
    ppe_model_path = project_root / ppe_config["model_path"]

    # Создаём детектор касок и жилетов
    ppe_detector = PPEDetector(
        model_path=ppe_model_path,
        confidence=ppe_config["confidence"],
        min_person_height=ppe_config["min_person_height"]
    )

    # Создаём обработчик PPE-событий
    ppe_event_logic = PPEEventLogic(
        confirmation_sec=ppe_config["confirmation_sec"]
    )


    # Счётчик FPS
    fps_meter = FPSMeter(window=30)

    try:
        video.open()
        print("Видеопоток успешно открыт")

        while True:

            # Получаем кадр
            frame = video.read()

            if frame is None:
                print("Видеофайл завершён")
                break

            # Размер кадра
            height, width = frame.shape[:2]

            # Переводим координаты ROI в пиксели
            polygon = normalized_to_pixels(
                roi_normalized,
                width,
                height
            )

            # Запускаем детекцию и tracking
            detections = detector.track(frame)

            # Идентификаторы людей внутри зоны
            objects_in_zone = set()

            original_frame = frame.copy()
            for det in detections:

                # Получаем координаты объекта
                x1, y1 = det.x1, det.y1
                x2, y2 = det.x2, det.y2

                # Проверяем нахождение в зоне
                point = bottom_center(det)

                inside = point_in_polygon(
                    point,
                    polygon
                )

                # Если человек в зоне и имеет Track ID
                if (
                    det.class_name == "person"
                    and inside
                    and det.track_id is not None
                ):
                    objects_in_zone.add(det.track_id)

                # Цвет прямоугольника:
                # красный — человек внутри ROI,
                # зелёный — остальные обнаружения
                if det.class_name == "person" and inside:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                # Рисуем Bounding Box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # Подпись
                track_label = (
                    det.track_id
                    if det.track_id is not None
                    else "-"
                )

                label = (
                    f"{det.class_name} "
                    f"ID:{track_label} "
                    f"{det.confidence:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                # Показываем нижнюю центральную точку
                cv2.circle(
                    frame,
                    point,
                    4,
                    color,
                    -1
                )

                # Проверяем PPE только у людей с Track ID
                if det.class_name == "person" and det.track_id is not None:

                    # Передаём исходный кадр и человека PPE-модели
                    ppe_result = ppe_detector.check_person(
                        original_frame,
                        det
                    )

                    # Если человек достаточно крупный
                    if ppe_result is not None:

                        # Передаём результат в Event Logic
                        ppe_events = ppe_event_logic.update(
                            track_id=det.track_id,
                            helmet_missing=ppe_result["helmet_missing"],
                            vest_missing=ppe_result["vest_missing"]
                        )

                        # Выводим подтверждённые события
                        for event in ppe_events:
                            print(
                                f"[EVENT] {event['event_type']} "
                                f"Track ID: {event['track_id']}"
                            )

                        # Рисуем каски и жилеты
                        for helmet in ppe_result["helmets"]:
                            hx1, hy1, hx2, hy2 = helmet["bbox"]

                            cv2.rectangle(
                                frame,
                                (hx1, hy1),
                                (hx2, hy2),
                                (0, 255, 0),
                                2
                            )

                        for vest in ppe_result["vests"]:
                            vx1, vy1, vx2, vy2 = vest["bbox"]

                            cv2.rectangle(
                                frame,
                                (vx1, vy1),
                                (vx2, vy2),
                                (255, 255, 0),
                                2
                            )

            # Передаём данные в Event Logic
            new_events = event_logic.update(objects_in_zone)

            # Выводим новые события в терминал
            for event in new_events:
                print(
                    f"[EVENT] {event['event_type']} "
                    f"Track ID: {event['track_id']}"
                )

            # Рисуем ROI поверх кадра
            draw_roi(frame, polygon)

            # Вычисляем FPS
            fps = fps_meter.tick()

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Objects: {len(detections)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"In ROI: {len(objects_in_zone)}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            # Отображаем кадр
            cv2.imshow(
                "Industrial Vision - Tracking and ROI",
                frame
            )

            # Выход по Q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Остановка пользователем")
                break

    except Exception as error:
        print(f"Ошибка: {error}")

    finally:
        video.close()
        cv2.destroyAllWindows()
        print("Программа завершена")


if __name__ == "__main__":
    main()
