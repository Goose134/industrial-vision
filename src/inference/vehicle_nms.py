"""Cross-class NMS exclusively for vehicles (other classes are untouched).

This is applied to Ultralytics Boxes before passing them to ByteTrack.
"""

import numpy as np

VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle"})


def box_iou(a, b):
    """Intersection-over-union for two xyxy boxes."""
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0:
        return 0.0

    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def vehicle_nms_mask(boxes, names, iou_threshold=0.60):
    """Return a boolean mask and suppressed pairs for Ultralytics Boxes.

    Only vehicle-vs-vehicle comparisons; compare classes across the group.
    The higher-confidence box survives. Does not modify the input Boxes.
    """
    threshold = float(iou_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("vehicle_nms_iou должен быть в диапазоне [0, 1]")

    n = len(boxes)
    keep = np.ones(n, dtype=bool)
    if n < 2:
        return keep, []

    coords = np.asarray(boxes.xyxy, dtype=float)
    classes = np.asarray(boxes.cls, dtype=int)
    scores = np.asarray(boxes.conf, dtype=float)

    vehicle_indices = [
        i for i in range(n)
        if str(names[int(classes[i])]) in VEHICLE_CLASSES
    ]
    # Deterministic order: stronger detections first; tie => lower index.
    order = sorted(vehicle_indices, key=lambda i: (-scores[i], i))
    chosen = []
    suppressed = []

    for i in order:
        duplicate_of = next(
            (j for j in chosen if box_iou(coords[i], coords[j]) > threshold),
            None,
        )
        if duplicate_of is None:
            chosen.append(i)
        else:
            keep[i] = False
            suppressed.append((i, duplicate_of))

    return keep, suppressed
