"""Additional per-class NMS for an already decoded Ultralytics Boxes object.

The model still does its normal NMS. This module optionally applies stricter
thresholds *within* selected classes after inference. It never compares
helmet to person, fire to smoke, or spills to other object categories.
"""

import numpy as np

from .vehicle_nms import box_iou


def _area(box):
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _intersection_area(a, b):
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    return max(0.0, right - left) * max(0.0, bottom - top)


def _nested_fire_duplicate(candidate, retained, candidate_score, retained_score):
    """Remove weak small fire box almost fully inside a stronger large one.

    Requires >=90% of the smaller box to overlap, <=20% area ratio,
    and the larger box to have at least 1.5x the confidence.
    This is a presentation/postprocessing heuristic, not a model fix.
    """
    small = _area(candidate)
    large = _area(retained)
    if small <= 0 or large <= 0 or small > 0.2 * large:
        return False
    if retained_score < 1.5 * candidate_score:
        return False
    return _intersection_area(candidate, retained) / small >= 0.90


def class_nms_mask(boxes, names, class_iou):
    """Return (keep_mask, removed_pairs), with indices into original `boxes`.

    `class_iou` maps class name to IoU threshold, e.g. {"fire": 0.5}.
    Only classes in the mapping are filtered. Suppression is SAME-CLASS only.
    """
    for name, value in class_iou.items():
        if not isinstance(name, str) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"Неверное значение class_nms_iou[{name!r}]: {value!r}")

    n = len(boxes)
    keep = np.ones(n, dtype=bool)
    if n < 2 or not class_iou:
        return keep, []

    coords = np.asarray(boxes.xyxy, dtype=float)
    classes = np.asarray(boxes.cls, dtype=int)
    scores = np.asarray(boxes.conf, dtype=float)
    class_names = [str(names[int(class_id)]) for class_id in classes]
    removed = []

    for name, threshold in class_iou.items():
        # Higher confidence goes first; stable order if confidences are equal.
        indices = [i for i, cls_name in enumerate(class_names) if cls_name == name]
        indices.sort(key=lambda i: (-float(scores[i]), i))
        selected = []
        for i in indices:
            stronger = next(
                (j for j in selected if (
                    box_iou(coords[i], coords[j]) > float(threshold)
                    or (name == "fire" and _nested_fire_duplicate(
                        coords[i], coords[j], float(scores[i]), float(scores[j])
                    ))
                )),
                None,
            )
            if stronger is None:
                selected.append(i)
            else:
                keep[i] = False
                removed.append((i, stronger))

    return keep, removed
