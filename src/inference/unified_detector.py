"""One YOLO inference for 10 classes with group-aware duplicate removal.

Vehicles: cross-class NMS. Other selected classes: separate same-class NMS.
Nothing is ever compared across groups (e.g., person/helmet or fire/smoke).
"""

import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

from .types import Detection
from .vehicle_nms import vehicle_nms_mask
from .class_nms import class_nms_mask


class UnifiedDetector:
    TRACKED_CLASSES = {"person", "car", "truck", "bus", "motorcycle"}

    def __init__(
        self,
        model_path,
        confidence=0.20,
        image_size=320,
        vehicle_nms_enabled=True,
        vehicle_nms_iou=0.60,
        vehicle_nms_debug=False,
        class_nms_enabled=True,
        class_nms_iou=None,
        class_nms_debug=False,
    ):
        self.model = YOLO(str(model_path), task="detect")
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.vehicle_nms_enabled = bool(vehicle_nms_enabled)
        self.vehicle_nms_iou = float(vehicle_nms_iou)
        self.vehicle_nms_debug = bool(vehicle_nms_debug)
        self.class_nms_enabled = bool(class_nms_enabled)
        self.class_nms_iou = dict(class_nms_iou or {})
        self.class_nms_debug = bool(class_nms_debug)

        allowed = {"person", "helmet", "vest", "fire", "smoke", "spill"}
        unknown = set(self.class_nms_iou) - allowed
        if unknown:
            raise ValueError(
                f"class_nms_iou: недопустимые классы {sorted(unknown)}. "
                "Для транспорта используется vehicle_nms_iou."
            )
        for name, threshold in self.class_nms_iou.items():
            if not 0.0 <= float(threshold) <= 1.0:
                raise ValueError(f"class_nms_iou[{name}] должен быть от 0 до 1")

        if not 0 <= self.vehicle_nms_iou <= 1:
            raise ValueError("vehicle_nms_iou должен быть в диапазоне [0, 1]")

        settings = YAML.load(check_yaml("bytetrack.yaml"))
        self.tracker = BYTETracker(IterableSimpleNamespace(**settings))

    def track(self, frame):
        # Only one model inference. Standard YOLO NMS is still class-aware.
        result = self.model.predict(
            source=frame,
            imgsz=self.image_size,
            conf=self.confidence,
            verbose=False,
        )[0]
        detections = []
        if result.boxes is None:
            return detections

        boxes = result.boxes.cpu().numpy()
        names = result.names

        # 1) Remove duplicate vehicle classes BEFORE tracking.
        if self.vehicle_nms_enabled:
            keep, dropped = vehicle_nms_mask(
                boxes, names, iou_threshold=self.vehicle_nms_iou,
            )
            if self.vehicle_nms_debug and dropped:
                for discarded, retained in dropped:
                    a = names[int(boxes.cls[discarded])]
                    b = names[int(boxes.cls[retained])]
                    value = float(boxes.conf[discarded])
                    print(f"[VehicleNMS] удалено {a} conf={value:.2f}, оставлено {b}")
            boxes = boxes[keep]

        # 2) Optional NMS within EACH selected class (never across classes).
        # Fire and smoke can coexist, as can people and their PPE.
        if self.class_nms_enabled and self.class_nms_iou:
            keep, dropped = class_nms_mask(
                boxes, names, class_iou=self.class_nms_iou,
            )
            if self.class_nms_debug and dropped:
                for discarded, retained in dropped:
                    class_name = str(names[int(boxes.cls[discarded])])
                    score = float(boxes.conf[discarded])
                    print(
                        f"[ClassNMS] удалён дубликат {class_name} "
                        f"conf={score:.2f}, оставлено обнаружение того же класса"
                    )
            boxes = boxes[keep]

        # 3) Convert every surviving detection, including PPE/fire/spill.
        for box in boxes:
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=str(names[class_id]),
                    confidence=float(box.conf.item()),
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    track_id=None,
                )
            )

        # 4) ByteTrack receives only people + the filtered vehicles.
        track_keep = np.array(
            [str(names[int(c)]) in self.TRACKED_CLASSES for c in boxes.cls],
            dtype=bool,
        )
        selected_indices = np.flatnonzero(track_keep)
        tracks = self.tracker.update(boxes[track_keep], frame)

        # Current Ultralytics ByteTrack rows:
        # [x1, y1, x2, y2, track_id, score, class_id, original_track_input_index]
        if tracks is not None:
            for row in tracks:
                index_in_selected = int(row[7])
                if 0 <= index_in_selected < len(selected_indices):
                    original_index = int(selected_indices[index_in_selected])
                    detections[original_index].track_id = int(row[4])

        return detections
