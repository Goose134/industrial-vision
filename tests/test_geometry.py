
from src.zones.geometry import point_in_polygon


def test_point_inside():
    polygon = [
        (0, 0),
        (100, 0),
        (100, 100),
        (0, 100)
    ]

    assert point_in_polygon((50, 50), polygon) is True


def test_point_outside():
    polygon = [
        (0, 0),
        (100, 0),
        (100, 100),
        (0, 100)
    ]

    assert point_in_polygon((150, 50), polygon) is False
