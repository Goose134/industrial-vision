
from unittest.mock import patch

from src.events.zone_events import ZoneEventLogic


def test_temporal_confirmation():
    """
    Проверяет, что событие формируется только после
    подтверждения на нескольких кадрах.
    """

    logic = ZoneEventLogic(
        n=3,
        m=5,
        cooldown_sec=10
    )

    # Человек с ID=7 находится в зоне
    # на первых двух кадрах
    assert logic.update({7}) == []
    assert logic.update({7}) == []

    # На третьем кадре человек вне зоны
    assert logic.update(set()) == []

    # На четвёртом кадре снова в зоне
    assert logic.update({7}) == []

    # На пятом кадре событие подтверждается
    events = logic.update({7})

    assert len(events) == 1
    assert events[0]["track_id"] == 7
    assert events[0]["event_type"] == "person_in_restricted_zone"


def test_cooldown():
    """
    Проверяет подавление повторного события
    в течение заданного времени.
    """

    logic = ZoneEventLogic(
        n=3,
        m=5,
        cooldown_sec=10
    )

    # Подменяем время, чтобы тест был воспроизводимым
    with patch(
        "src.events.zone_events.time.monotonic",
        return_value=100.0
    ):
        for _ in range(4):
            assert logic.update({5}) == []

        # Пятый кадр: первое событие
        events = logic.update({5})
        assert len(events) == 1

    # Спустя 5 секунд повторное событие запрещено
    with patch(
        "src.events.zone_events.time.monotonic",
        return_value=105.0
    ):
        events = logic.update({5})
        assert events == []

    # Спустя 11 секунд cooldown закончился
    with patch(
        "src.events.zone_events.time.monotonic",
        return_value=111.0
    ):
        events = logic.update({5})
        assert len(events) == 1
