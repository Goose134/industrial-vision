
import json

import cv2
import numpy as np

from fastapi.testclient import TestClient

from src.api import server


# Клиент для тестирования HTTP-запросов
client = TestClient(server.app)


def test_home():
    """Проверяем главную страницу API."""

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "Industrial Vision API"
    assert data["status"] == "running"


def test_health():
    """Проверяем работоспособность API."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_events(tmp_path, monkeypatch):
    """Проверяем получение списка событий."""

    # Создаём временный журнал
    events_path = tmp_path / "events.jsonl"

    event = {
        "event_id": "test123",
        "camera_id": "cam_01",
        "event_type": "fire_detected",
        "track_id": None
    }

    events_path.write_text(
        json.dumps(event) + "\n",
        encoding="utf-8"
    )

    # Подменяем путь к журналу
    monkeypatch.setattr(
        server,
        "EVENTS_PATH",
        events_path
    )

    response = client.get("/events")

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["event_type"] == "fire_detected"


def test_event_by_id(tmp_path, monkeypatch):
    """Проверяем получение события по ID."""

    events_path = tmp_path / "events.jsonl"

    event = {
        "event_id": "abc123",
        "event_type": "spill_detected"
    }

    events_path.write_text(
        json.dumps(event) + "\n",
        encoding="utf-8"
    )

    monkeypatch.setattr(
        server,
        "EVENTS_PATH",
        events_path
    )

    response = client.get("/events/abc123")

    assert response.status_code == 200
    assert response.json()["event_type"] == "spill_detected"


def test_event_not_found(tmp_path, monkeypatch):
    """Проверяем ответ для несуществующего события."""

    events_path = tmp_path / "events.jsonl"
    events_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        server,
        "EVENTS_PATH",
        events_path
    )

    response = client.get("/events/unknown")

    assert response.status_code == 404


def test_stats(tmp_path, monkeypatch):
    """Проверяем подсчёт событий."""

    events_path = tmp_path / "events.jsonl"

    events = [
        {"event_type": "fire_detected"},
        {"event_type": "fire_detected"},
        {"event_type": "spill_detected"}
    ]

    with open(events_path, "w", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(event) + "\n")

    monkeypatch.setattr(
        server,
        "EVENTS_PATH",
        events_path
    )

    response = client.get("/stats")

    assert response.status_code == 200

    data = response.json()

    assert data["total_events"] == 3
    assert data["events_by_type"]["fire_detected"] == 2
    assert data["events_by_type"]["spill_detected"] == 1


def test_evidence(tmp_path, monkeypatch):
    """Проверяем получение фотографии через API."""

    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()

    # Создаём тестовое изображение
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    image_path = evidence_dir / "photo123.jpg"
    assert cv2.imwrite(str(image_path), frame)

    # Создаём событие
    events_path = tmp_path / "events.jsonl"

    event = {
        "event_id": "photo123",
        "event_type": "fire_detected",
        "evidence_path": "evidence/photo123.jpg"
    }

    events_path.write_text(
        json.dumps(event) + "\n",
        encoding="utf-8"
    )

    # Используем временные файлы
    monkeypatch.setattr(server, "EVENTS_PATH", events_path)
    monkeypatch.setattr(server, "EVIDENCE_DIR", evidence_dir)

    response = client.get("/evidence/photo123")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 0


def test_invalid_limit():
    """Проверяем неправильное значение limit."""

    response = client.get("/events?limit=0")

    assert response.status_code == 400
