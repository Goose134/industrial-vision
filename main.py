"""Industrial Vision: один проход YOLO26n для всех десяти классов.

Замени содержимое основного main.py этим файлом после сохранения резервной копии.
Использует уже созданный src/inference/unified_detector.py.
"""
from pathlib import Path
from collections import deque
import time

import cv2
import yaml

from src.capture.video_capture import VideoSource
from src.capture.fps_meter import FPSMeter
from src.capture.playback import playback_wait_ms
from src.inference.unified_detector import UnifiedDetector
from src.inference.vehicle_class_stabilizer import VehicleClassStabilizer
from src.inference.ppe_visual_stabilizer import PersonBoxSmoother, PPEVisualSmoother
from src.zones.geometry import (
    normalized_to_pixels,
    bbox_intersects_polygon,
    draw_roi,
)
from src.events.zone_events import ZoneEventLogic
from src.events.ppe_events import PPEEventLogic
from src.events.fire_events import FireSmokeEventLogic
from src.events.spill_events import SpillEventLogic
from src.events.event_engine import EventEngine

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
ALL_CLASSES = (
    "person", "car", "truck", "bus", "motorcycle",
    "helmet", "vest", "fire", "smoke", "spill",
)


def get_class_thresholds(model_config):
    """Проверяет персональные пороги; они не могут быть ниже порога YOLO."""
    lower_bound = float(model_config.get("detector_confidence", 0.20))
    raw = model_config.get("class_confidence")
    if not isinstance(raw, dict):
        raise ValueError(
            "В models.class_confidence должны быть указаны пороги 10 классов."
        )

    missing = sorted(set(ALL_CLASSES) - set(raw))
    extra = sorted(set(raw) - set(ALL_CLASSES))
    if missing or extra:
        raise ValueError(
            f"Ошибка class_confidence: отсутствуют={missing}, лишние={extra}"
        )

    if not 0 <= lower_bound <= 1:
        raise ValueError("detector_confidence должен быть от 0 до 1")

    values = {}
    for name in ALL_CLASSES:
        threshold = float(raw[name])
        if not 0 <= threshold <= 1:
            raise ValueError(f"Недопустимый confidence для {name}: {threshold}")
        if threshold < lower_bound:
            raise ValueError(
                f"Порог {name} ({threshold}) ниже общего "
                f"detector_confidence ({lower_bound}). "
                "Уменьшите detector_confidence, иначе YOLO "
                "отбросит такие обнаружения заранее."
            )
        values[name] = threshold
    return values


def equipment_for_people(people, equipment, min_person_height):
    """Привязывает каждую найденную каску/жилет к одному человеку.

    В отличие от старого PPEDetector, НЕ запускает нейросеть на crop.
    Координаты уже найдены общей YOLO на целом кадре.

    Правила соответствуют прежней PPE-логике:
      - каска: верхние 40% рамки человека;
      - жилет: от 15% до 80% высоты человека.
    При пересечении нескольких рамок выбираем ближайшего кандидата.
    """
    eligible = [
        person for person in people
        if person.track_id is not None
        and person.y2 - person.y1 >= min_person_height
    ]
    assigned = {
        person.track_id: {"helmets": [], "vests": []}
        for person in eligible
    }

    for item in equipment:
        cx = (item.x1 + item.x2) / 2
        cy = (item.y1 + item.y2) / 2
        best_person = None
        best_distance = float("inf")

        for person in eligible:
            h = person.y2 - person.y1
            w = max(1, person.x2 - person.x1)
            # Небольшой допуск по краям: каска иногда выходит за рамку person.
            margin = 0.12 * w
            if not (person.x1 - margin <= cx <= person.x2 + margin):
                continue

            vertical = (cy - person.y1) / h
            if item.class_name == "helmet":
                valid = -0.12 <= vertical <= 0.42
                expected = 0.13
            elif item.class_name == "vest":
                valid = 0.10 <= vertical <= 0.83
                expected = 0.45
            else:
                continue

            if not valid:
                continue

            horizontal = abs(cx - (person.x1 + person.x2) / 2) / w
            distance = horizontal + abs(vertical - expected)

            if distance < best_distance:
                best_person = person
                best_distance = distance

        if best_person is not None:
            kind = "helmets" if item.class_name == "helmet" else "vests"
            assigned[best_person.track_id][kind].append(item)

    return assigned


def draw_box(frame, det, color, label=None):
    """Рисует прямоугольник и подпись без повторного запуска YOLO."""
    cv2.rectangle(
        frame,
        (det.x1, det.y1),
        (det.x2, det.y2),
        color,
        2,
    )
    if label is None:
        label = f"{det.class_name}: {det.confidence:.2f}"
    cv2.putText(
        frame,
        label,
        (det.x1, max(20, det.y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def main():
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "configs" / "configuration.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    modules = config.get("modules", {})
    roi_enabled = modules.get("roi_enabled", True)
    ppe_enabled = modules.get("ppe_enabled", True)
    fire_enabled = modules.get("fire_smoke_enabled", True)
    spills_enabled = modules.get("spills_enabled", True)

    print("ROI:", roi_enabled)
    print("PPE:", ppe_enabled)
    print("Fire/Smoke:", fire_enabled)
    print("Spills:", spills_enabled)

    camera_config = config["camera"]
    source = camera_config["source"]
    if isinstance(source, str):
        if source.isdigit():
            source = int(source)  # Поддержка USB-камеры "0"
        elif "://" not in source:
            source = str(project_root / source)

    # Камеры: отдельный поток захвата хранит только последний кадр.
    # Для файла кадры читаются последовательно без пропусков.
    playback_cfg = config.get("playback", {})
    realtime_video = bool(playback_cfg.get("realtime_video", True))
    fallback_fps = float(playback_cfg.get("fallback_fps", 30.0))
    video = VideoSource(
        source=source,
        reconnect_sec=camera_config.get("reconnect_sec", 2),
        latest_frame=camera_config.get("latest_frame", True),
        read_timeout_sec=camera_config.get("read_timeout_sec", 0.5),
    )

    model_config = config["models"]
    # Для всех классов выполняется ТОЛЬКО одна нейросеть.
    detector = UnifiedDetector(
        model_path=project_root / model_config["detector_path"],
        confidence=model_config.get("detector_confidence", 0.20),
        image_size=model_config["image_size"],
        # Cross-class NMS: сравниваем ТОЛЬКО рамки транспорта.
        vehicle_nms_enabled=config.get("tracking", {}).get("vehicle_nms_enabled", True),
        vehicle_nms_iou=config.get("tracking", {}).get("vehicle_nms_iou", 0.60),
        vehicle_nms_debug=config.get("tracking", {}).get("vehicle_nms_debug", False),
        # Дополнительный NMS отдельно для людей, СИЗ, огня, дыма, разливов.
        class_nms_enabled=config.get("tracking", {}).get("class_nms_enabled", True),
        class_nms_iou=config.get("tracking", {}).get("class_nms_iou", {}),
        class_nms_debug=config.get("tracking", {}).get("class_nms_debug", False),
    )
    class_conf = get_class_thresholds(model_config)
    print("Пороги классов:", ", ".join(
        f"{name}={class_conf[name]:.2f}" for name in ALL_CLASSES
    ))

    roi_normalized = config["zones"]["restricted_01"]
    event_config = config["events"]
    event_logic = ZoneEventLogic(
        n=event_config["confirmation_n"],
        m=event_config["confirmation_m"],
        cooldown_sec=event_config["cooldown_sec"],
    )
    vehicle_event_logic = ZoneEventLogic(
        n=event_config["confirmation_n"],
        m=event_config["confirmation_m"],
        cooldown_sec=event_config["cooldown_sec"],
    )

    ppe_config = config["ppe"]
    fire_config = config["fire_smoke"]
    spill_config = config["spills"]

    # Ни один из этих объектов не загружает модель: это только логика событий.
    ppe_event_logic = (
        PPEEventLogic(confirmation_sec=ppe_config["confirmation_sec"])
        if ppe_enabled else None
    )
    fire_event_logic = (
        FireSmokeEventLogic(confirmation_sec=fire_config["confirmation_sec"])
        if fire_enabled else None
    )
    spill_event_logic = (
        SpillEventLogic(confirmation_sec=spill_config["confirmation_sec"])
        if spills_enabled else None
    )
    event_engine = EventEngine(
        project_root=project_root,
        camera_id=camera_config.get("id", "cam_01"),
    )
    fps_meter = FPSMeter(window=30)  # Реальная скорость показа, включая ожидание
    processing_durations = deque(maxlen=30)  # Скорость вычисления без ожидания

    # Стабилизация названия транспорта по Track ID (не запускает YOLO повторно).
    tracking_cfg = config.get("tracking", {})
    vehicle_stabilizer = (
        VehicleClassStabilizer(
            window_size=tracking_cfg.get("vehicle_class_window", 5),
            switch_votes=tracking_cfg.get("vehicle_class_switch_votes", 3),
            forget_after_frames=tracking_cfg.get("vehicle_class_forget_frames", 60),
        )
        if tracking_cfg.get("stabilize_vehicle_classes", True)
        else None
    )

    # Только плавность отображения. На ByteTrack, ROI и координаты обнаружений
    # эти фильтры не влияют. СИЗ кратко сохраняются при пропуске 1-3 кадров.
    visual_cfg = config.get("visual_smoothing", {})
    visual_enabled = bool(visual_cfg.get("enabled", True))
    person_visual = PersonBoxSmoother(
        alpha=float(visual_cfg.get("person_alpha", 0.65))
    ) if visual_enabled else None
    ppe_visual = PPEVisualSmoother(
        alpha=float(visual_cfg.get("ppe_alpha", 0.55)),
        hold_frames=int(visual_cfg.get("ppe_hold_frames", 3)),
    ) if visual_enabled and ppe_enabled else None

    try:
        video.open()
        print("Видеопоток успешно открыт")
        if video.is_live:
            print("Источник: камера/поток; режим свежих кадров:", video.latest_frame)
        else:
            print(f"Источник: видеофайл, FPS записи: {video.fps or fallback_fps:.2f}")
            print("Воспроизведение с исходной скоростью:", realtime_video)

        while True:
            frame_started_at = time.perf_counter()
            frame = video.read()
            if frame is None:
                if video.is_live:
                    # Камера временно не прислала кадр: ждём переподключения,
                    # но оставляем обработку клавиши Q доступной пользователю.
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Остановка пользователем")
                        break
                    continue
                print("Видеофайл завершён")
                break

            # Для файла учитываем также время декодирования кадра.
            # Для камеры исключаем ожидание очередного кадра от самой камеры.
            processing_started_at = (
                time.perf_counter() if video.is_live else frame_started_at
            )
            height, width = frame.shape[:2]
            original_frame = frame.copy()
            if vehicle_stabilizer is not None:
                vehicle_stabilizer.next_frame()
            if person_visual is not None:
                person_visual.next_frame()
            if ppe_visual is not None:
                ppe_visual.next_frame()

            # Один запуск YOLO + ByteTrack на один кадр.
            all_detections = detector.track(frame)

            people = [
                d for d in all_detections
                if d.class_name == "person"
                and d.confidence >= class_conf["person"]
            ]
            vehicles = [
                d for d in all_detections
                if d.class_name in VEHICLE_CLASSES
                and d.confidence >= class_conf[d.class_name]
            ]
            equipment = [
                d for d in all_detections
                if ppe_enabled
                and d.class_name in {"helmet", "vest"}
                and d.confidence >= class_conf[d.class_name]
            ]
            fire_detections = [
                d for d in all_detections
                if fire_enabled
                and d.class_name in {"fire", "smoke"}
                and d.confidence >= class_conf[d.class_name]
            ]
            spill_detections = [
                d for d in all_detections
                if spills_enabled
                and d.class_name == "spill"
                and d.confidence >= class_conf["spill"]
            ]

            # ROI относится только к людям и транспорту.
            objects_in_zone = set()
            vehicles_in_zone = set()
            polygon = (
                normalized_to_pixels(roi_normalized, width, height)
                if roi_enabled else None
            )

            # Рамки от модели не меняем: ROI и PPE используют оригинальные.
            display_people = {}
            for det in people + vehicles:
                visual_det = (
                    person_visual.update(det)
                    if det.class_name == "person" and person_visual is not None
                    else det
                )
                if det.class_name == "person" and det.track_id is not None:
                    display_people[det.track_id] = visual_det
                inside = (
                    bbox_intersects_polygon(det, polygon)
                    if roi_enabled else False
                )
                color = (0, 0, 255) if inside else (0, 255, 0)
                track_label = det.track_id if det.track_id is not None else "-"

                # Сглаживаем только НАДПИСЬ car/truck/bus/motorcycle,
                # не меняя класс, confidence и данные для ROI/Event Engine.
                display_class = det.class_name
                if vehicle_stabilizer is not None and det.class_name in VEHICLE_CLASSES:
                    display_class = vehicle_stabilizer.update(
                        det.track_id, det.class_name
                    )

                # Confidence относится к исходному результату нейросети.
                # При сглаженном названии не показываем чужую уверенность.
                if display_class == det.class_name:
                    label = f"{display_class} ID:{track_label} {det.confidence:.2f}"
                else:
                    label = f"{display_class} ID:{track_label} (stable)"

                draw_box(frame, visual_det, color, label)

                if inside and det.track_id is not None:
                    if det.class_name == "person":
                        objects_in_zone.add(det.track_id)
                    else:
                        vehicles_in_zone.add(det.track_id)

            if roi_enabled:
                for event in event_logic.update(objects_in_zone):
                    event_engine.register(
                        event_type="person_in_restricted_zone",
                        frame=original_frame,
                        track_id=event["track_id"],
                    )
                for event in vehicle_event_logic.update(vehicles_in_zone):
                    event_engine.register(
                        event_type="vehicle_in_restricted_zone",
                        frame=original_frame,
                        track_id=event["track_id"],
                    )

            # PPE: только геометрическое сопоставление, без второго YOLO.
            if ppe_enabled:
                assigned = equipment_for_people(
                    people,
                    equipment,
                    min_person_height=ppe_config["min_person_height"],
                )
                people_by_id = {
                    person.track_id: person for person in people
                    if person.track_id is not None
                }
                assigned_item_ids = {
                    id(item)
                    for matches in assigned.values()
                    for group in ("helmets", "vests")
                    for item in matches[group]
                }
                for track_id, items in assigned.items():
                    raw_person = people_by_id[track_id]
                    displayed_person = display_people.get(track_id, raw_person)
                    for class_name, key, color in (
                        ("helmet", "helmets", (0, 165, 255)),
                        ("vest", "vests", (255, 255, 0)),
                    ):
                        if ppe_visual is not None:
                            shown, held = ppe_visual.update(
                                raw_person, displayed_person,
                                class_name, items[key],
                            )
                        else:
                            shown = max(
                                items[key], key=lambda d: d.confidence, default=None
                            )
                            held = False
                        # Кеш предотвращает ложный сброс наличия СИЗ при
                        # краткой потере одного-двух обнаружений.
                        items[key] = [shown] if shown is not None else []
                        if shown is not None:
                            label = (
                                f"{class_name} (cached)" if held
                                else f"{class_name}: {shown.confidence:.2f}"
                            )
                            draw_box(frame, shown, color, label)

                    events = ppe_event_logic.update(
                        track_id=track_id,
                        helmet_missing=(len(items["helmets"]) == 0),
                        vest_missing=(len(items["vests"]) == 0),
                    )
                    for event in events:
                        event_engine.register(
                            event_type=event["event_type"],
                            frame=original_frame,
                            track_id=event["track_id"],
                        )

                # Убираем состояние людей, исчезнувших из кадра.
                ppe_event_logic.remove_missing_tracks(set(assigned))
                # Не теряем для диагностики обнаружения СИЗ, которые не
                # удалось связать с человеком (например, ещё нет Track ID).
                if ppe_config.get("show_unassigned", False):
                    for item in equipment:
                        if id(item) not in assigned_item_ids:
                            color = (0, 165, 255) if item.class_name == "helmet" else (255, 255, 0)
                            draw_box(frame, item, color, f"{item.class_name} (unassigned)")

            if fire_enabled:
                detected_fire = any(d.class_name == "fire" for d in fire_detections)
                detected_smoke = any(d.class_name == "smoke" for d in fire_detections)
                for event in fire_event_logic.update(
                    fire_detected=detected_fire,
                    smoke_detected=detected_smoke,
                ):
                    event_engine.register(
                        event_type=event["event_type"],
                        frame=original_frame,
                    )
                for det in fire_detections:
                    color = (0, 0, 255) if det.class_name == "fire" else (0, 255, 255)
                    draw_box(frame, det, color)

            if spills_enabled:
                for event in spill_event_logic.update(
                    spill_detected=bool(spill_detections),
                ):
                    event_engine.register(
                        event_type=event["event_type"],
                        frame=original_frame,
                    )
                for det in spill_detections:
                    draw_box(frame, det, (255, 0, 255))

            if roi_enabled:
                draw_roi(frame, polygon)

            # FPS обработки не зависит от пауз при воспроизведении файла.
            processing_durations.append(
                max(1e-9, time.perf_counter() - processing_started_at)
            )
            process_fps = len(processing_durations) / sum(processing_durations)
            display_fps = fps_meter.tick()
            shown_count = (
                len(people) + len(vehicles) + len(equipment)
                + len(fire_detections) + len(spill_detections)
            )
            cv2.putText(
                frame, f"Proc FPS: {process_fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )
            cv2.putText(
                frame, f"Display FPS: {display_fps:.1f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )
            cv2.putText(
                frame, f"Objects: {shown_count}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )
            if roi_enabled:
                cv2.putText(
                    frame,
                    f"In ROI: {len(objects_in_zone) + len(vehicles_in_zone)}",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                )

            cv2.imshow("Industrial Vision - Unified YOLO26n", frame)
            # Для файлов выдерживаем период 1/FPS с учётом уже затраченного
            # на чтение, YOLO и рисование времени. Для камер пауз нет.
            wait_ms = playback_wait_ms(
                is_live=video.is_live,
                realtime_video=realtime_video,
                source_fps=video.fps,
                frame_started_at=frame_started_at,
                now=time.perf_counter(),
                fallback_fps=fallback_fps,
            )
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                print("Остановка пользователем")
                break

    except Exception as error:
        print(f"Ошибка: {error}")
        raise  # Сохраняем traceback, чтобы не скрывать ошибки при интеграции.
    finally:
        video.close()
        cv2.destroyAllWindows()
        print("Программа завершена")


if __name__ == "__main__":
    main()
