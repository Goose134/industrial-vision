
import cv2
import numpy as np


def normalized_to_pixels(points, width, height):
    """
    Преобразует нормированные координаты ROI
    в координаты пикселей изображения.
    """
    return [
        (
            int(x * width),
            int(y * height)
        )
        for x, y in points
    ]


def bottom_center(det):
    """
    Нижняя центральная точка Bounding Box.
    Приблизительно соответствует положению ног.
    """
    return (
        (det.x1 + det.x2) // 2,
        det.y2
    )


def point_in_polygon(point, polygon):
    """
    Проверяет, находится ли точка
    внутри многоугольника или на его границе.
    """
    contour = np.array(polygon, dtype=np.int32)

    result = cv2.pointPolygonTest(
        contour,
        point,
        False
    )

    return result >= 0


def draw_roi(frame, polygon):
    """
    Рисует границу запрещённой зоны.
    """
    contour = np.array(polygon, dtype=np.int32)

    cv2.polylines(
        frame,
        [contour],
        isClosed=True,
        color=(0, 0, 255),
        thickness=3
    )

    return frame
