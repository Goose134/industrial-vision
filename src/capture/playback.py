"""Timing helpers independent of the machine-learning inference engine."""

import math


def playback_wait_ms(*, is_live, realtime_video, source_fps, frame_started_at, now,
                     fallback_fps=30.0):
    """Return OpenCV waitKey milliseconds, accounting for processing time.

    For live sources or maximum-speed files, return the minimum GUI poll wait.
    For real-time file playback, budget one source-frame period per iteration.
    """
    if is_live or not realtime_video:
        return 1
    fps = float(source_fps)
    if not math.isfinite(fps) or fps <= 0:
        fps = float(fallback_fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("playback.fallback_fps должен быть больше нуля")
    spent = max(0.0, now - frame_started_at)
    remaining = max(0.0, (1.0 / fps) - spent)
    return max(1, int(round(remaining * 1000)))
