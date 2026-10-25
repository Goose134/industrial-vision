
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

    # Какие нейросети включены
    modules = config.get("modules", {})
    roi_enabled = modules.get("roi_enabled", True)
    ppe_enabled = modules.get("ppe_enabled", True)
    fire_enabled = modules.get("fire_smoke_enabled", True)
    spills_enabled = modules.get("spills_enabled", True)

    print("ROI:", roi_enabled)
    print("PPE:", ppe_enabled)
    print("Fire/Smoke:", fire_enabled)
    print("Spills:", spills_enabled)

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

    # Отдельная логика событий для транспорта
    vehicle_event_logic = ZoneEventLogic(
        n=event_config["confirmation_n"],
        m=event_config["confirmation_m"],
        cooldown_sec=event_config["cooldown_sec"]
    )

    # Классы транспорта, которые отслеживаем в ROI
    vehicle_classes = {
        "car",
        "truck",
        "bus",
        "motorcycle"
    }

    # PPE: каски и защитные жилеты
    ppe_detector = None
    ppe_event_logic = None

    if ppe_enabled:
        ppe_config = config["ppe"]

        ppe_detector = PPEDetector(
            model_path=project_root / ppe_config["model_path"],
            confidence=ppe_config["confidence"],
            min_person_height=ppe_config["min_person_height"]
        )

        ppe_event_logic = PPEEventLogic(
            confirmation_sec=ppe_config["confirmation_sec"]
        )




    # Fire/Smoke: обнаружение огня и дыма
    fire_detector = None
    fire_event_logic = None

    if fire_enabled:
        fire_config = config["fire_smoke"]

        fire_detector = FireSmokeDetector(
            model_path=project_root / fire_config["model_path"],
            confidence=fire_config["confidence"],
            image_size=fire_config["image_size"]
        )

        fire_event_logic = FireSmokeEventLogic(
            confirmation_sec=fire_config["confirmation_sec"]
        )



    # Создаём единую систему регистрации событий
    event_engine = EventEngine(
        project_root=project_root,
        camera_id=camera_config.get("id", "cam_01")
    )



    # Spill: обнаружение разливов жидкостей
    spill_detector = None
    spill_event_logic = None

    if spills_enabled:
        spill_config = config["spills"]

        spill_detector = SpillDetector(
            model_path=project_root / spill_config["model_path"],
            confidence=spill_config["confidence"],
            image_size=spill_config["image_size"]
        )

        spill_event_logic = SpillEventLogic(
            confirmation_sec=spill_config["confirmation_sec"]
        )


    # Счётчик FPS
    fps_meter = FPSMeter(window=30)

    # Настройки оптимизации из configuration.yaml
    optimization = config.get("optimization", {})

    EXTRA_DETECTION_INTERVAL = int(
        optimization.get("extra_detection_interval", 3)
    )

    if EXTRA_DETECTION_INTERVAL < 1:
        raise ValueError(
            "extra_detection_interval должен быть >= 1"
        )

    frame_counter = 0

    # Последние результаты обнаружения
    fire_detections = []
    spill_detections = []

    # Сохраняем последние обнаружения касок и жилетов
    ppe_cache = {}

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

            # Увеличиваем счётчик кадров
            frame_counter += 1

            # Дополнительные модели запускаем каждый третий кадр
            run_extra = (
                frame_counter % EXTRA_DETECTION_INTERVAL == 0
            )

            # Переводим координаты ROI в пиксели
            polygon = normalized_to_pixels(
                roi_normalized,
                width,
                height
            )

            # Запускаем детекцию и tracking
            detections = detector.track(frame)

            # Fire/Smoke выполняем только на каждом третьем кадре
            if fire_enabled and run_extra:
                fire_detections = fire_detector.detect(frame)

            # Spill выполняем только на каждом третьем кадре
            if spills_enabled and run_extra:
                spill_detections = spill_detector.detect(frame)

            # Люди внутри запрещённой зоны
            objects_in_zone = set()

            # Транспорт внутри запрещённой зоны
            vehicles_in_zone = set()

            original_frame = frame.copy()
            for det in detections:

                # Получаем координаты объекта
                x1, y1 = det.x1, det.y1
                x2, y2 = det.x2, det.y2

                # Нижняя центральная точка объекта
                point = bottom_center(det)

                # Проверяем ROI только тогда, когда она включена
                inside = (
                    point_in_polygon(point, polygon)
                    if roi_enabled
                    else False
                )


                # Проверяем попадание людей и транспорта в ROI
                if inside and det.track_id is not None:

                    # Если обнаружен человек
                    if det.class_name == "person":
                        objects_in_zone.add(det.track_id)

                    # Если обнаружен транспорт
                    elif det.class_name in vehicle_classes:
                        vehicles_in_zone.add(det.track_id)

                # Цвет рамки:
                # красный — человек или транспорт внутри ROI
                # зелёный — остальные обнаружения
                if (
                    inside
                    and (
                        det.class_name == "person"
                        or det.class_name in vehicle_classes
                    )
                ):
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

                # Обработка PPE для отслеживаемых людей
                if (
                    ppe_enabled
                    and det.class_name == "person"
                    and det.track_id is not None
                ):
                    track_id = det.track_id

                    # Запускаем PPE только каждый третий кадр
                    if run_extra:
                        ppe_result = ppe_detector.check_person(
                            original_frame,
                            det
                        )

                        if ppe_result is not None:

                            # Сохраняем координаты рамок
                            ppe_cache[track_id] = {
                                "helmets": ppe_result["helmets"],
                                "vests": ppe_result["vests"]
                            }

                            # Проверяем нарушения СИЗ
                            ppe_events = ppe_event_logic.update(
                                track_id=track_id,
                                helmet_missing=ppe_result["helmet_missing"],
                                vest_missing=ppe_result["vest_missing"]
                            )

                            for event in ppe_events:
                                event_engine.register(
                                    event_type=event["event_type"],
                                    frame=original_frame,
                                    track_id=event["track_id"]
                                )

                    # Рисуем последние сохранённые обнаружения
                    cached = ppe_cache.get(track_id)

                    if cached is not None:

                        # Оранжевые рамки касок
                        for helmet in cached["helmets"]:
                            hx1, hy1, hx2, hy2 = helmet["bbox"]

                            cv2.rectangle(
                                frame,
                                (hx1, hy1),
                                (hx2, hy2),
                                (0, 165, 255),
                                2
                            )

                        # Голубые рамки жилетов
                        for vest in cached["vests"]:
                            vx1, vy1, vx2, vy2 = vest["bbox"]

                            cv2.rectangle(
                                frame,
                                (vx1, vy1),
                                (vx2, vy2),
                                (255, 255, 0),
                                2
                            )

                        # Удаляем PPE-рамки людей, которых больше нет в кадре
                        active_track_ids = {
                            det.track_id
                            for det in detections
                            if det.class_name == "person"
                               and det.track_id is not None
                        }

                        for track_id in list(ppe_cache.keys()):
                            if track_id not in active_track_ids:
                                del ppe_cache[track_id]

                        # Сбрасываем PPE-события исчезнувших людей
                        if ppe_enabled:
                            ppe_event_logic.remove_missing_tracks(
                                active_track_ids
                            )

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
                                (0, 165, 255),
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


            # Проверяем события ROI, только если зона включена
            if roi_enabled:

                # 1. События нахождения людей в зоне
                person_events = event_logic.update(
                    objects_in_zone
                )

                for event in person_events:
                    event_engine.register(
                        event_type="person_in_restricted_zone",
                        frame=original_frame,
                        track_id=event["track_id"]
                    )

                # 2. События нахождения транспорта в зоне
                vehicle_events = vehicle_event_logic.update(
                    vehicles_in_zone
                )

                for event in vehicle_events:
                    event_engine.register(
                        event_type="vehicle_in_restricted_zone",
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
            fire_events = []

            if fire_enabled and run_extra:
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
            spill_events = []

            if spills_enabled and run_extra:
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

            # Рисуем ROI только при включённой зоне
            if roi_enabled:
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

            if roi_enabled:
                cv2.putText(
                    frame,
                    f"In ROI: {len(objects_in_zone) + len(vehicles_in_zone)}",
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
