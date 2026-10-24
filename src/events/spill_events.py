
import time


class SpillEventLogic:
    """
    Создаёт событие разлива после подтверждения
    в течение заданного времени.
    """

    def __init__(self, confirmation_sec=3.0):
        if confirmation_sec < 0:
            raise ValueError(
                "Время подтверждения должно быть >= 0"
            )

        self.confirmation_sec = confirmation_sec
        self.detection_start = None
        self.active = False

    def update(self, spill_detected):
        current_time = time.monotonic()

        if not spill_detected:
            # Разлив перестал обнаруживаться
            self.detection_start = None
            self.active = False
            return []

        # Первое обнаружение
        if self.detection_start is None:
            self.detection_start = current_time

        elapsed = current_time - self.detection_start

        # Подтверждаем нарушение
        if (
            elapsed >= self.confirmation_sec
            and not self.active
        ):
            self.active = True

            return [{
                "event_type": "spill_detected"
            }]

        return []
