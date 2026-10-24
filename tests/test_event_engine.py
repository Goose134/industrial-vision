
import json

import numpy as np

from src.events.event_engine import EventEngine


def test_event_registration(tmp_path):
    """
    Проверяет запись JSON и сохранение Evidence.
    """

    # Создаём временную папку для теста
    engine = EventEngine(
        project_root=tmp_path,
        camera_id="test_camera"
    )

    # Создаём тестовый чёрный кадр
    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8
    )

    # Регистрируем событие
    event = engine.register(
        event_type="fire_detected",
        frame=frame
    )

    # Проверяем основные поля
    assert event["event_type"] == "fire_detected"
    assert event["camera_id"] == "test_camera"
    assert event["track_id"] is None

    # Проверяем журнал
    assert engine.log_path.exists()

    with open(
        engine.log_path,
        "r",
        encoding="utf-8"
    ) as file:
        saved_event = json.loads(file.readline())

    assert saved_event["event_id"] == event["event_id"]

    # Проверяем наличие изображения
    evidence_file = (
        engine.output_dir / event["evidence_path"]
    )

    assert evidence_file.is_file()


def test_multiple_events(tmp_path):
    """
    Проверяет сохранение нескольких событий.
    """

    engine = EventEngine(
        project_root=tmp_path,
        camera_id="cam_01"
    )

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8
    )

    first = engine.register(
        "ppe_helmet_missing",
        frame,
        track_id=5
    )

    second = engine.register(
        "smoke_detected",
        frame
    )

    # У каждого события уникальный ID
    assert first["event_id"] != second["event_id"]

    # В журнале должно быть две записи
    with open(
        engine.log_path,
        "r",
        encoding="utf-8"
    ) as file:
        lines = file.readlines()

    assert len(lines) == 2

    # Проверяем Track ID первого события
    first_saved = json.loads(lines[0])

    assert first_saved["track_id"] == 5
