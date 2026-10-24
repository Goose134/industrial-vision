
import json
import uuid

from datetime import datetime
from pathlib import Path

import cv2


class EventEngine:
    """
    Единая система регистрации событий.

    Сохраняет:
    1. JSON-запись о событии.
    2. Изображение с камеры.
    """

    def __init__(self, project_root, camera_id="cam_01"):
        self.project_root = Path(project_root)
        self.camera_id = camera_id

        # Папка результатов
        self.output_dir = self.project_root / "output"

        # Папка кадров-доказательств
        self.evidence_dir = self.output_dir / "evidence"

        self.evidence_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Общий журнал событий
        self.log_path = self.output_dir / "events.jsonl"

    def register(
        self,
        event_type,
        frame,
        track_id=None,
        confidence=None
    ):
        """
        Регистрирует одно подтверждённое событие.
        """

        # Создаём уникальный ID
        event_id = uuid.uuid4().hex

        # Фиксируем время с часовым поясом
        timestamp = datetime.now().astimezone().isoformat()

        # Путь для сохранения кадра
        image_name = f"{event_id}.jpg"
        image_path = self.evidence_dir / image_name

        # Сохраняем кадр
        saved = cv2.imwrite(str(image_path), frame)

        if not saved:
            raise RuntimeError(
                f"Не удалось сохранить изображение: {image_path}"
            )

        # Относительный путь для JSON
        evidence_path = (
            Path("evidence") / image_name
        ).as_posix()

        # Собираем данные события
        event = {
            "event_id": event_id,
            "timestamp": timestamp,
            "camera_id": self.camera_id,
            "event_type": event_type,
            "track_id": track_id,
            "confidence": confidence,
            "evidence_path": evidence_path
        }

        # Записываем событие в журнал
        with open(
            self.log_path,
            "a",
            encoding="utf-8"
        ) as file:
            file.write(
                json.dumps(
                    event,
                    ensure_ascii=False
                ) + "\n"
            )

        # Сообщаем о событии в терминал
        print(
            f"[EVENT] {event_type} "
            f"Track ID: {track_id} "
            f"Evidence: {evidence_path}"
        )

        return event
