
import time
import cv2


class VideoSource:
    """
    Класс для получения кадров из видеофайла или камеры.
    """

    def __init__(self, source, reconnect_sec=2.0):
        # Источник: путь к видеофайлу или номер камеры
        self.source = source

        # Задержка перед повторным подключением
        self.reconnect_sec = reconnect_sec

        # Объект видеозахвата OpenCV
        self.cap = None

    def open(self):
        """
        Открывает источник видеопотока.
        """

        # Освобождаем старый источник, если он существует
        if self.cap is not None:
            self.cap.release()

        # Пытаемся открыть видео или камеру
        self.cap = cv2.VideoCapture(self.source)

        # Проверяем успешность открытия
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Не удалось открыть источник видео: {self.source}"
            )

    def read(self):
        """
        Возвращает следующий кадр видеопотока.
        """

        # Если источник не открыт, открываем его
        if self.cap is None or not self.cap.isOpened():
            self.open()

        # Считываем очередной кадр
        ok, frame = self.cap.read()

        # Если кадр успешно получен
        if ok and frame is not None:
            return frame

        # Для обычного видеофайла конец видео — не ошибка
        if isinstance(self.source, str) and not self.source.lower().startswith(
            ("rtsp://", "http://", "https://")
        ):
            return None

        # При сбое камеры или сетевого потока ждём
        time.sleep(self.reconnect_sec)

        # Пробуем подключиться заново
        self.open()

        # Повторная попытка чтения
        ok, frame = self.cap.read()

        if not ok or frame is None:
            raise RuntimeError(
                "Не удалось получить кадр после переподключения"
            )

        return frame

    def close(self):
        """
        Освобождает источник видеопотока.
        """

        if self.cap is not None:
            self.cap.release()
            self.cap = None
