"""Reading files normally, and live sources with a latest-frame background reader.

The capture thread continuously drains a camera/RTSP stream into one slot.
Inference consumes the newest image and does not accumulate an outdated queue.
"""

import math
import threading
import time

import cv2


class VideoSource:
    """OpenCV video source for file, USB camera (int), or network stream (URL).

    For cameras ``latest_frame=True`` discards frames while inference is busy.
    For files frames are NEVER discarded; frame pacing is handled by the caller.

    ``read()`` on a live stream returns None after a short timeout if no new
    frame is ready. Unlike a file, None on a camera is NOT an end-of-file.
    """

    def __init__(
        self,
        source,
        reconnect_sec=2.0,
        latest_frame=True,
        read_timeout_sec=0.5,
    ):
        if isinstance(source, str) and source.isdecimal():
            source = int(source)
        self.source = source
        self.reconnect_sec = float(reconnect_sec)
        self.latest_frame = bool(latest_frame)
        self.read_timeout_sec = float(read_timeout_sec)
        if self.reconnect_sec < 0:
            raise ValueError("reconnect_sec не может быть отрицательным")
        if self.read_timeout_sec <= 0:
            raise ValueError("read_timeout_sec должен быть положительным")

        self.is_live = isinstance(self.source, int) or (
            isinstance(self.source, str) and "://" in self.source
        )
        self.cap = None
        self.fps = 0.0
        self._thread = None
        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._latest_frame = None
        self._frame_sequence = 0
        self._last_consumed_sequence = 0

    def _new_capture(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Не удалось открыть источник видео: {self.source}")
        if self.is_live and self.latest_frame:
            # Supported only by some OpenCV backends; the reader thread is
            # the primary means of keeping the picture fresh.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def open(self):
        """Opens the source and starts the latest-frame reader for cameras."""
        self.close()
        self._stop_event.clear()
        with self._condition:
            self._latest_frame = None
            self._frame_sequence = 0
            self._last_consumed_sequence = 0

        self.cap = self._new_capture()
        reported_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = reported_fps if math.isfinite(reported_fps) and reported_fps > 0 else 0.0

        if self.is_live and self.latest_frame:
            self._thread = threading.Thread(
                target=self._camera_reader,
                name="VideoSourceLatestFrame",
                daemon=True,
            )
            self._thread.start()

    def _camera_reader(self):
        """Continuously reads and overwrites the previous unconsumed frame."""
        was_disconnected = False
        while not self._stop_event.is_set():
            if self.cap is None:
                try:
                    self.cap = self._new_capture()
                    if was_disconnected:
                        print("[VideoSource] Камера переподключена")
                    was_disconnected = False
                except Exception as exc:
                    print(f"[VideoSource] Повторное подключение: {exc}")
                    if self._stop_event.wait(self.reconnect_sec):
                        break
                    continue

            try:
                ok, frame = self.cap.read()
            except Exception as exc:
                print(f"[VideoSource] Ошибка чтения камеры: {exc}")
                ok, frame = False, None

            if ok and frame is not None:
                with self._condition:
                    self._latest_frame = frame
                    self._frame_sequence += 1
                    self._condition.notify_all()
                continue

            if self._stop_event.is_set():
                break
            if not was_disconnected:
                print("[VideoSource] Потеря кадров камеры. Переподключение...")
            was_disconnected = True
            try:
                if self.cap is not None:
                    self.cap.release()
            finally:
                self.cap = None
            if self._stop_event.wait(self.reconnect_sec):
                break

        with self._condition:
            self._condition.notify_all()

    def read(self):
        """Next file frame, or the most recent *new* live frame.

        Live None means temporary lack of a new frame, not EOF.
        """
        if self.cap is None and self._thread is None:
            self.open()

        if self.is_live and self.latest_frame:
            with self._condition:
                available = self._condition.wait_for(
                    lambda: self._frame_sequence > self._last_consumed_sequence
                    or self._stop_event.is_set(),
                    timeout=self.read_timeout_sec,
                )
                if not available or self._stop_event.is_set():
                    return None
                self._last_consumed_sequence = self._frame_sequence
                return self._latest_frame.copy()

        ok, frame = self.cap.read()
        if ok and frame is not None:
            return frame
        if not self.is_live:
            return None
        # Optional synchronous camera mode (latest_frame=False).
        if self._stop_event.wait(self.reconnect_sec):
            return None
        self.open()
        ok, frame = self.cap.read()
        return frame if ok and frame is not None else None

    def close(self):
        """Stops the reader and releases the capture; safe to call repeatedly."""
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        worker = self._thread
        if worker is not None and worker.is_alive():
            worker.join(timeout=0.5)
        # On some backends, releasing the capture helps unblock read().
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if worker is not None and worker.is_alive():
            worker.join(timeout=0.5)
        self._thread = None
        with self._condition:
            self._latest_frame = None
