
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


def segments_intersect(a, b, c, d):
    """
    Проверяет пересечение двух отрезков AB и CD,
    включая касание и совпадение границ.
    """

    def orientation(p, q, r):
        return (
            (q[0] - p[0]) * (r[1] - p[1])
            - (q[1] - p[1]) * (r[0] - p[0])
        )

    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)

    if (o1 > 0 > o2 or o1 < 0 < o2) and \
       (o3 > 0 > o4 or o3 < 0 < o4):
        return True

    if o1 == 0 and on_segment(a, c, b):
        return True
    if o2 == 0 and on_segment(a, d, b):
        return True
    if o3 == 0 and on_segment(c, a, d):
        return True
    if o4 == 0 and on_segment(c, b, d):
        return True

    return False


def bbox_intersects_polygon(det, polygon):
    """
    Проверяет пересечение всей рамки объекта
    с запрещённой зоной.
    """

    x1, y1 = int(det.x1), int(det.y1)
    x2, y2 = int(det.x2), int(det.y2)

    # Четыре угла Bounding Box
    box = [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2)
    ]

    # 1. Хотя бы один угол рамки внутри зоны
    for point in box:
        if point_in_polygon(point, polygon):
            return True

    # 2. Хотя бы одна вершина зоны внутри рамки
    for px, py in polygon:
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True

    # 3. Пересечение границ рамки и зоны
    for i in range(4):
        a = box[i]
        b = box[(i + 1) % 4]

        for j in range(len(polygon)):
            c = polygon[j]
            d = polygon[(j + 1) % len(polygon)]

            if segments_intersect(a, b, c, d):
                return True

    return False
