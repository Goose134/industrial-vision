
import time


class FireSmokeEventLogic:
    """
    Подтверждает обнаружение дыма и огня
    в течение заданного времени.

    Для каждого типа опасности отсчёт
    ведётся независимо.
    """

    def __init__(self, confirmation_sec=3.0):
        if confirmation_sec < 0:
            raise ValueError(
                "Время подтверждения не может быть отрицательным"
            )

        self.confirmation_sec = confirmation_sec

        # Начало обнаружения опасности
        self.detection_start = {}

        # Уже подтверждённые опасности
        self.active_events = set()

    def update(self, fire_detected, smoke_detected):
        current_time = time.monotonic()

        states = {
            "fire_detected": fire_detected,
            "smoke_detected": smoke_detected
        }

        events = []

        for event_type, detected in states.items():

            if detected:
                # Начинаем отсчёт при первом обнаружении
                if event_type not in self.detection_start:
                    self.detection_start[event_type] = current_time

                elapsed = (
                    current_time
                    - self.detection_start[event_type]
                )

                # Подтверждаем опасность
                if (
                    elapsed >= self.confirmation_sec
                    and event_type not in self.active_events
                ):
                    events.append({
                        "event_type": event_type
                    })

                    # Больше не повторяем событие
                    self.active_events.add(event_type)

            else:
                # Опасность перестала обнаруживаться
                self.detection_start.pop(event_type, None)
                self.active_events.discard(event_type)

        return events
