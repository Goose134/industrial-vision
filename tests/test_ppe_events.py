
from unittest.mock import patch

from src.events.ppe_events import PPEEventLogic


def update_at(logic, seconds, track_id, helmet, vest):
    """
    Вызывает проверку СИЗ в заданный момент времени.

    helmet и vest:
    True — СИЗ отсутствует.
    False — СИЗ обнаружено.
    """
    with patch(
        "src.events.ppe_events.time.monotonic",
        return_value=seconds
    ):
        return logic.update(
            track_id=track_id,
            helmet_missing=helmet,
            vest_missing=vest
        )


def test_helmet_confirmation():
    """Предупреждение о каске появляется через 3 секунды."""

    logic = PPEEventLogic(confirmation_sec=3.0)

    assert update_at(logic, 10.0, 1, True, False) == []
    assert update_at(logic, 11.0, 1, True, False) == []
    assert update_at(logic, 12.9, 1, True, False) == []

    events = update_at(logic, 13.0, 1, True, False)

    assert len(events) == 1
    assert events[0]["event_type"] == "ppe_helmet_missing"
    assert events[0]["track_id"] == 1


def test_vest_confirmation():
    """Предупреждение о жилете появляется через 3 секунды."""

    logic = PPEEventLogic(confirmation_sec=3.0)

    assert update_at(logic, 20.0, 2, False, True) == []
    assert update_at(logic, 22.0, 2, False, True) == []

    events = update_at(logic, 23.0, 2, False, True)

    assert len(events) == 1
    assert events[0]["event_type"] == "ppe_vest_missing"


def test_timer_reset():
    """Если каска появилась, таймер сбрасывается."""

    logic = PPEEventLogic(confirmation_sec=3.0)

    assert update_at(logic, 0.0, 3, True, False) == []
    assert update_at(logic, 2.0, 3, True, False) == []

    # Каска обнаружена: нарушение прервалось
    assert update_at(logic, 2.5, 3, False, False) == []

    # Отсчёт начинается заново
    assert update_at(logic, 3.0, 3, True, False) == []
    assert update_at(logic, 5.0, 3, True, False) == []

    events = update_at(logic, 6.0, 3, True, False)

    assert len(events) == 1


def test_no_duplicate_events():
    """Одно непрерывное нарушение создаёт одно событие."""

    logic = PPEEventLogic(confirmation_sec=3.0)

    all_events = []

    for second in range(10):
        events = update_at(
            logic,
            float(second),
            4,
            True,
            False
        )
        all_events.extend(events)

    assert len(all_events) == 1


def test_no_violation():
    """Если СИЗ присутствуют, предупреждений нет."""

    logic = PPEEventLogic(confirmation_sec=3.0)

    for second in range(10):
        events = update_at(
            logic,
            float(second),
            5,
            False,
            False
        )

        assert events == []


def test_independent_timers():
    """Каска и жилет проверяются независимо."""

    logic = PPEEventLogic(confirmation_sec=3.0)

    # Сначала пропадает каска
    assert update_at(logic, 0.0, 6, True, False) == []

    # Через секунду пропадает и жилет
    assert update_at(logic, 1.0, 6, True, True) == []

    # Через 3 секунды от первого кадра
    # подтверждается отсутствие каски
    events = update_at(logic, 3.0, 6, True, True)

    assert len(events) == 1
    assert events[0]["event_type"] == "ppe_helmet_missing"

    # Ещё через секунду — отсутствие жилета
    events = update_at(logic, 4.0, 6, True, True)

    assert len(events) == 1
    assert events[0]["event_type"] == "ppe_vest_missing"
