
from dataclasses import dataclass


@dataclass
class Detection:
    """
    Данные об объекте, найденном нейросетью.
    """

    class_id: int
    class_name: str
    confidence: float

    # Координаты прямоугольника объекта
    x1: int
    y1: int
    x2: int
    y2: int

    # Идентификатор объекта при отслеживании
    track_id: int | None = None
