"""Lightweight tests: no ONNX model, camera, or actual video files required."""

import time

import numpy as np
import pytest

from src.capture.playback import playback_wait_ms
from src.capture.video_capture import VideoSource


class FakeCapture:
    def __init__(self, source, *, limit=500, delay=0.0, fps=25.0):
        self.source = source
        self.limit = limit
        self.delay = delay
        self.fps = fps
        self.read_count = 0
        self.opened = True
        self.buffer_setting = None

    def isOpened(self):
        return self.opened

    def read(self):
        if self.delay:
            time.sleep(self.delay)
        if not self.opened or self.read_count >= self.limit:
            return False, None
        self.read_count += 1
        return True, np.full((3, 3, 3), self.read_count % 256, dtype=np.uint8)

    def get(self, _prop):
        return self.fps

    def set(self, key, value):
        self.buffer_setting = (key, value)
        return True

    def release(self):
        self.opened = False


def test_local_file_read_to_eof(monkeypatch):
    monkeypatch.setattr("src.capture.video_capture.cv2.VideoCapture",
                        lambda src: FakeCapture(src, limit=3))
    video = VideoSource("data/demo.mp4")
    video.open()
    try:
        assert video.is_live is False
        assert video.fps == 25
        assert [int(video.read()[0, 0, 0]) for _ in range(3)] == [1, 2, 3]
        assert video.read() is None
        assert video._thread is None
    finally:
        video.close()


@pytest.mark.parametrize("source", [0, "0", 1, "rtsp://example.test/live", "http://example.test/mjpg"])
def test_live_source_recognition(source):
    stream = VideoSource(source)
    assert stream.is_live is True
    if source == "0":
        assert stream.source == 0


def test_live_reader_discards_old_frames(monkeypatch):
    monkeypatch.setattr("src.capture.video_capture.cv2.VideoCapture",
                        lambda src: FakeCapture(src, limit=1000, delay=0.003, fps=30))
    video = VideoSource(0, reconnect_sec=0.01, read_timeout_sec=0.3)
    video.open()
    try:
        first = video.read()
        assert first is not None
        first_number = int(first[0, 0, 0])
        time.sleep(0.055)  # Simulates slow inference in main.py.
        second = video.read()
        assert second is not None
        assert int(second[0, 0, 0]) >= first_number + 6
    finally:
        video.close()
    assert video.cap is None
    assert video._thread is None


def test_live_read_timeout_allows_gui_poll(monkeypatch):
    # A camera with a slow cadence: no new frame during the short timeout.
    monkeypatch.setattr("src.capture.video_capture.cv2.VideoCapture",
                        lambda src: FakeCapture(src, delay=0.15, fps=30))
    video = VideoSource(0, read_timeout_sec=0.02)
    video.open()
    try:
        assert video.read() is None
    finally:
        video.close()


def test_playback_timing_file_25_fps():
    assert playback_wait_ms(is_live=False, realtime_video=True, source_fps=25,
                            frame_started_at=100.0, now=100.020) == 20
    assert playback_wait_ms(is_live=False, realtime_video=True, source_fps=25,
                            frame_started_at=100.0, now=100.050) == 1


def test_camera_and_fast_mode_never_add_delay():
    base = dict(source_fps=25, frame_started_at=100.0, now=100.010)
    assert playback_wait_ms(is_live=True, realtime_video=True, **base) == 1
    assert playback_wait_ms(is_live=False, realtime_video=False, **base) == 1


def test_invalid_file_fps_uses_fallback():
    assert playback_wait_ms(is_live=False, realtime_video=True, source_fps=0,
                            frame_started_at=0.0, now=0.010,
                            fallback_fps=20) == 40
    with pytest.raises(ValueError):
        playback_wait_ms(is_live=False, realtime_video=True, source_fps=0,
                         frame_started_at=0.0, now=0.010, fallback_fps=0)
