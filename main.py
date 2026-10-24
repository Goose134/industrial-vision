
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

from src.fire_smoke.fire_detector import FireSmokeDetector
from src.events.fire_events import FireSmokeEventLogic
from src.events.event_engine import EventEngine

from src.spills.spill_detector import SpillDetector
from src.events.spill_events import SpillEventLogic

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


    # Загружаем настройки обнаружения огня и дыма
    fire_config = config["fire_smoke"]

    # Путь к специализированной модели
    fire_model_path = (
        project_root / fire_config["model_path"]
    )

    # Создаём детектор огня и дыма
    fire_detector = FireSmokeDetector(
        model_path=fire_model_path,
        confidence=fire_config["confidence"],
        image_size=fire_config["image_size"]
    )

    # Создаём обработчик событий пожара
    fire_event_logic = FireSmokeEventLogic(
        confirmation_sec=fire_config["confirmation_sec"]
    )


    # Создаём единую систему регистрации событий
    event_engine = EventEngine(
        project_root=project_root,
        camera_id=camera_config.get("id", "cam_01")
    )


    # Настройки обнаружения разливов
    spill_config = config["spills"]

    # Путь к обученной модели
    spill_model_path = (
        project_root / spill_config["model_path"]
    )

    # Создаём детектор разливов
    spill_detector = SpillDetector(
        model_path=spill_model_path,
        confidence=spill_config["confidence"],
        image_size=spill_config["image_size"]
    )

    # Логика подтверждения события разлива
    spill_event_logic = SpillEventLogic(
        confirmation_sec=spill_config["confirmation_sec"]
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

            # Анализируем кадр на наличие огня и дыма
            fire_detections = fire_detector.detect(frame)

            # Обнаруживаем разливы жидкостей
            spill_detections = spill_detector.detect(frame)

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


                        # Регистрируем нарушения PPE
                        for event in ppe_events:
                            event_engine.register(
                                event_type=event["event_type"],
                                frame=original_frame,
                                track_id=event["track_id"]
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


            # Регистрируем события нарушения зоны
            for event in new_events:
                event_engine.register(
                    event_type=event["event_type"],
                    frame=original_frame,
                    track_id=event["track_id"]
                )


            # Проверяем, обнаружены ли огонь и дым
            has_fire = any(
                item["class_name"] == "fire"
                for item in fire_detections
            )

            has_smoke = any(
                item["class_name"] == "smoke"
                for item in fire_detections
            )

            # Обновляем события огня и дыма
            fire_events = fire_event_logic.update(
                fire_detected=has_fire,
                smoke_detected=has_smoke
            )


            # Регистрируем обнаружение дыма и огня
            for event in fire_events:
                event_engine.register(
                    event_type=event["event_type"],
                    frame=original_frame
                )


            # Отображаем обнаруженные опасности
            for item in fire_detections:

                class_name = item["class_name"]
                confidence = item["confidence"]

                x1, y1, x2, y2 = item["bbox"]

                # Красный цвет для огня
                if class_name == "fire":
                    color = (0, 0, 255)

                # Жёлтый цвет для дыма
                else:
                    color = (0, 255, 255)

                # Рисуем Bounding Box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # Подписываем обнаружение
                label = (
                    f"{class_name}: {confidence:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )


            # Есть ли хотя бы один обнаруженный разлив?
            has_spill = len(spill_detections) > 0

            # Проверяем длительность обнаружения
            spill_events = spill_event_logic.update(
                spill_detected=has_spill
            )

            # Регистрируем подтверждённые события
            for event in spill_events:
                event_engine.register(
                    event_type=event["event_type"],
                    frame=original_frame
                )

            # Отображаем обнаруженные разливы
            for item in spill_detections:
                x1, y1, x2, y2 = item["bbox"]

                confidence = item["confidence"]

                # Фиолетовый цвет для разливов
                color = (255, 0, 255)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                label = f"spill: {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
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
