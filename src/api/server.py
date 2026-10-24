
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse


# Находим корневую папку industrial-vision
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Путь к журналу событий
EVENTS_PATH = PROJECT_ROOT / "output" / "events.jsonl"

# Папка сохранённых кадров
EVIDENCE_DIR = PROJECT_ROOT / "output" / "evidence"

# Создаём приложение FastAPI
app = FastAPI(
    title="Industrial Vision API",
    description="REST API системы промышленного технического зрения",
    version="1.0"
)


def read_events():
    """
    Читает события из JSONL-файла
    и возвращает список словарей.
    """

    if not EVENTS_PATH.is_file():
        return []

    events = []

    with open(EVENTS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # Пропускаем повреждённую запись
                continue

    return events


@app.get("/")
def home():
    """Главная страница API."""
    return {
        "service": "Industrial Vision API",
        "status": "running"
    }


@app.get("/health")
def health():
    """Проверяет доступность API."""
    return {
        "status": "ok",
        "service": "Industrial Vision API"
    }


@app.get("/events")
def get_events(limit: int = 100):
    """
    Возвращает последние события.
    """

    if not 1 <= limit <= 1000:
        raise HTTPException(
            status_code=400,
            detail="limit должен быть от 1 до 1000"
        )

    events = read_events()

    return events[-limit:][::-1]


@app.get("/events/{event_id}")
def get_event(event_id: str):
    """Возвращает событие по его ID."""

    for event in read_events():
        if event.get("event_id") == event_id:
            return event

    raise HTTPException(
        status_code=404,
        detail="Событие не найдено"
    )


@app.get("/evidence/{event_id}")
def get_evidence(event_id: str):
    """Возвращает кадр-доказательство события."""

    event = None

    for item in read_events():
        if item.get("event_id") == event_id:
            event = item
            break

    if event is None:
        raise HTTPException(
            status_code=404,
            detail="Событие не найдено"
        )

    evidence_name = f"{event_id}.jpg"
    image_path = EVIDENCE_DIR / evidence_name

    if not image_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Изображение не найдено"
        )

    return FileResponse(
        path=str(image_path),
        media_type="image/jpeg"
    )


@app.get("/stats")
def get_stats():
    """Возвращает статистику по событиям."""

    events = read_events()
    counts = {}

    for event in events:
        event_type = event.get("event_type", "unknown")

        counts[event_type] = counts.get(event_type, 0) + 1

    return {
        "total_events": len(events),
        "events_by_type": counts
    }
