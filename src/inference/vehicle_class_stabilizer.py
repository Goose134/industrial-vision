"""Сглаживание названия класса транспорта на экране по Track ID.

Нейросеть и ByteTrack продолжают работать как прежде: стабилизация влияет
только на подпись рамки в main.py, а не на сами детекции или события.
"""

from collections import Counter, deque


class VehicleClassStabilizer:
    def __init__(self, window_size=5, switch_votes=3, forget_after_frames=60):
        window_size = int(window_size)
        switch_votes = int(switch_votes)
        forget_after_frames = int(forget_after_frames)

        if window_size < 1:
            raise ValueError("vehicle_class_window должен быть >= 1")
        if not 1 <= switch_votes <= window_size:
            raise ValueError("vehicle_class_switch_votes должен быть от 1 до window_size")
        if forget_after_frames < 1:
            raise ValueError("vehicle_class_forget_frames должен быть >= 1")

        self.window_size = window_size
        self.switch_votes = switch_votes
        self.forget_after_frames = forget_after_frames
        self._history = {}
        self._stable = {}
        self._last_seen = {}
        self._frame = 0

    def next_frame(self):
        """Вызывать один раз на каждом прочитанном кадре."""
        self._frame += 1
        expired = [
            track_id for track_id, last_frame in self._last_seen.items()
            if self._frame - last_frame > self.forget_after_frames
        ]
        for track_id in expired:
            self._history.pop(track_id, None)
            self._stable.pop(track_id, None)
            self._last_seen.pop(track_id, None)

    def update(self, track_id, observed_class):
        """Возвращает более стабильное имя, если объект имеет Track ID."""
        if track_id is None:
            return observed_class

        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=self.window_size)
            self._stable[track_id] = observed_class

        self._history[track_id].append(observed_class)
        self._last_seen[track_id] = self._frame

        count = Counter(self._history[track_id])
        winner, votes = count.most_common(1)[0]
        if winner != self._stable[track_id] and votes >= self.switch_votes:
            self._stable[track_id] = winner

        return self._stable[track_id]
