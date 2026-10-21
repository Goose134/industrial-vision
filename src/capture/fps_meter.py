
import time
from collections import deque


class FPSMeter:
    """
    Класс для расчёта текущего FPS.
    """

    def __init__(self, window=30):
        # Храним время обработки последних кадров
        self.times = deque(maxlen=window)

    def tick(self):
        """
        Вызывается после обработки каждого кадра.
        Возвращает текущий FPS.
        """

        # Получаем текущее время
        now = time.perf_counter()

        # Добавляем время в очередь
        self.times.append(now)

        # Для расчёта нужно минимум два кадра
        if len(self.times) < 2:
            return 0.0

        # Временной интервал между первым и последним
        dt = self.times[-1] - self.times[0]

        # Вычисляем количество кадров в секунду
        if dt > 0:
            return (len(self.times) - 1) / dt

        return 0.0
