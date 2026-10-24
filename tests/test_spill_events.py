
from unittest.mock import patch

from src.events.spill_events import SpillEventLogic


def update_at(logic, seconds, detected):
    with patch(
        "src.events.spill_events.time.monotonic",
        return_value=seconds
    ):
        return logic.update(
            spill_detected=detected
        )


def test_spill_confirmation():
    # Подтверждение через 3 секунды
    logic = SpillEventLogic(confirmation_sec=3.0)

    assert update_at(logic, 0, True) == []
    assert update_at(logic, 1, True) == []
    assert update_at(logic, 2, True) == []

    events = update_at(logic, 3, True)

    assert len(events) == 1
    assert events[0]["event_type"] == "spill_detected"


def test_no_spill():
    # Если жидкости нет, предупреждение не создаётся
    logic = SpillEventLogic(confirmation_sec=3.0)

    for second in range(10):
        assert update_at(logic, second, False) == []


def test_no_duplicate_events():
    # Одно непрерывное обнаружение — одно событие
    logic = SpillEventLogic(confirmation_sec=3.0)

    events = []

    for second in range(10):
        events.extend(update_at(logic, second, True))

    assert len(events) == 1


def test_timer_reset():
    # Кратковременное обнаружение
    # не должно создавать событие
    logic = SpillEventLogic(confirmation_sec=3.0)

    assert update_at(logic, 0, True) == []
    assert update_at(logic, 1, False) == []

    assert update_at(logic, 2, True) == []
    assert update_at(logic, 4, True) == []

    events = update_at(logic, 5, True)

    assert len(events) == 1


def test_new_event_after_disappearance():
    # После исчезновения разлива
    # и повторного обнаружения возможно новое событие
    logic = SpillEventLogic(confirmation_sec=3.0)

    update_at(logic, 0, True)
    first = update_at(logic, 3, True)

    assert len(first) == 1

    update_at(logic, 4, False)

    update_at(logic, 5, True)
    second = update_at(logic, 8, True)

    assert len(second) == 1
