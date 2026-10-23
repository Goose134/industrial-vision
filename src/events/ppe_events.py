
import time


class PPEEventLogic:
    """
    Проверяет отсутствие каски и жилета.

    Нарушение подтверждается, если СИЗ
    непрерывно не обнаруживается заданное
    количество секунд.
    """

    def __init__(self, confirmation_sec=3.0):
        self.confirmation_sec = confirmation_sec

        # Время начала возможного нарушения
        self.violation_start = {}

        # Уже подтверждённые нарушения
        self.active_violations = set()

    def update(self, track_id, helmet_missing, vest_missing):
        if track_id is None:
            return []

        current_time = time.monotonic()
        events = []

        states = {
            "ppe_helmet_missing": helmet_missing,
            "ppe_vest_missing": vest_missing
        }

        for event_type, missing in states.items():
            key = (track_id, event_type)

            if missing:
                # Начинаем отсчёт при первом обнаружении
                if key not in self.violation_start:
                    self.violation_start[key] = current_time

                elapsed = current_time - self.violation_start[key]

                # Если нарушение продолжается 3 секунды
                if (
                    elapsed >= self.confirmation_sec
                    and key not in self.active_violations
                ):
                    events.append({
                        "event_type": event_type,
                        "track_id": track_id
                    })

                    self.active_violations.add(key)

            else:
                # СИЗ обнаружено: сбрасываем отсчёт
                self.violation_start.pop(key, None)
                self.active_violations.discard(key)

        return events
