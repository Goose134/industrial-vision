
from collections import defaultdict, deque
import time


class ZoneEventLogic:
    """
    Проверка нарушений зоны с временным
    подтверждением и cooldown.
    """

    def __init__(self, n=3, m=5, cooldown_sec=10):

        if not 1 <= n <= m:
            raise ValueError("Должно выполняться 1 <= n <= m")

        self.n = n
        self.m = m
        self.cooldown_sec = cooldown_sec

        # История нахождения объектов в зоне
        self.history = defaultdict(
            lambda: deque(maxlen=m)
        )

        # Время предыдущих событий
        self.last_events = {}

    def update(self, objects_in_zone):
        """
        objects_in_zone: множество Track ID,
        находящихся в зоне на текущем кадре.

        Возвращает список новых событий.
        """

        now = time.monotonic()
        new_events = []

        # Проверяем как присутствующие,
        # так и ранее отслеживаемые объекты
        all_ids = set(self.history) | set(objects_in_zone)

        for track_id in all_ids:

            # Запоминаем состояние на текущем кадре
            inside = track_id in objects_in_zone
            self.history[track_id].append(inside)

            history = self.history[track_id]

            # Нужно накопить полное окно из m кадров
            confirmed = (
                len(history) == self.m
                and sum(history) >= self.n
                and inside
            )

            if not confirmed:
                continue

            last_time = self.last_events.get(track_id)

            # Проверяем cooldown
            if (
                last_time is not None
                and now - last_time < self.cooldown_sec
            ):
                continue

            # Создаём событие
            new_events.append({
                "event_type": "person_in_restricted_zone",
                "track_id": track_id
            })

            self.last_events[track_id] = now

        return new_events
