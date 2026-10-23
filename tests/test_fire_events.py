
from unittest.mock import patch

from src.events.fire_events import FireSmokeEventLogic


def update_at(logic, seconds, fire=False, smoke=False):
    """
    Имитирует проверку дыма и огня
    в заданный момент времени.
    """
    with patch(
        "src.events.fire_events.time.monotonic",
        return_value=seconds
    ):
        return logic.update(
            fire_detected=fire,
            smoke_detected=smoke
        )


def test_fire_confirmation():
    # Проверяем задержку обнаружения огня
    logic = FireSmokeEventLogic(confirmation_sec=3.0)

    assert update_at(logic, 0, fire=True) == []
    assert update_at(logic, 1, fire=True) == []
    assert update_at(logic, 2, fire=True) == []

    events = update_at(logic, 3, fire=True)

    assert len(events) == 1
    assert events[0]["event_type"] == "fire_detected"


def test_smoke_confirmation():
    # Проверяем задержку обнаружения дыма
    logic = FireSmokeEventLogic(confirmation_sec=3.0)

    assert update_at(logic, 0, smoke=True) == []
    assert update_at(logic, 2, smoke=True) == []

    events = update_at(logic, 3, smoke=True)

    assert len(events) == 1
    assert events[0]["event_type"] == "smoke_detected"


def test_false_alarm():
    # Кратковременное обнаружение не должно
    # приводить к предупреждению
    logic = FireSmokeEventLogic(confirmation_sec=3.0)

    assert update_at(logic, 0, fire=True) == []
    assert update_at(logic, 1, fire=False) == []
    assert update_at(logic, 2, fire=True) == []

    # С момента повторного обнаружения
    # прошло недостаточно времени
    assert update_at(logic, 3, fire=True) == []


def test_no_duplicate_events():
    # При постоянном пожаре событие
    # создаётся только один раз
    logic = FireSmokeEventLogic(confirmation_sec=3.0)

    all_events = []

    for second in range(10):
        all_events.extend(
            update_at(logic, second, fire=True)
        )

    assert len(all_events) == 1


def test_independent_detection():
    # Огонь и дым должны иметь отдельные таймеры
    logic = FireSmokeEventLogic(confirmation_sec=3.0)

    assert update_at(logic, 0, fire=True) == []
    assert update_at(logic, 1, fire=True, smoke=True) == []

    # Огонь подтверждён через 3 секунды
    events = update_at(logic, 3, fire=True, smoke=True)

    assert len(events) == 1
    assert events[0]["event_type"] == "fire_detected"

    # Дым подтверждён через 3 секунды
    # после своего первого обнаружения
    events = update_at(logic, 4, fire=True, smoke=True)

    assert len(events) == 1
    assert events[0]["event_type"] == "smoke_detected"


def test_no_detection():
    # Если опасностей нет, событий тоже нет
    logic = FireSmokeEventLogic(confirmation_sec=3.0)

    for second in range(10):
        assert update_at(logic, second) == []
