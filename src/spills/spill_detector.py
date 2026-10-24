
from pathlib import Path

import torch
from ultralytics import YOLO


class SpillDetector:
    """
    Обнаруживает разливы жидкостей с помощью YOLO.
    """

    def __init__(
        self,
        model_path,
        confidence=0.35,
        image_size=640
    ):
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"Не найдена модель разливов: {model_path}"
            )

        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.image_size = image_size

        self.device = (
            0 if torch.cuda.is_available() else "cpu"
        )

    def detect(self, frame):
        """
        Принимает кадр и возвращает список
        обнаруженных разливов.
        """

        result = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False
        )[0]

        detections = []

        for box in result.boxes:
            class_id = int(box.cls.item())

            # Исходные классы: Spill и spillage
            original_name = str(
                result.names[class_id]
            ).lower()

            # Объединяем оба класса в категорию spill
            if original_name not in ("spill", "spillage"):
                continue

            confidence = float(box.conf.item())

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            detections.append({
                "class_name": "spill",
                "original_class": original_name,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2)
            })

        return detections
