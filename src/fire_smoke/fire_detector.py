
from pathlib import Path

import torch
from ultralytics import YOLO


class FireSmokeDetector:
    """
    Обнаруживает дым и огонь с помощью
    отдельной обученной YOLO-модели.
    """

    def __init__(
        self,
        model_path,
        confidence=0.35,
        image_size=640
    ):
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"Не найдена модель Fire/Smoke: {model_path}"
            )

        # Загружаем нейросеть один раз
        self.model = YOLO(str(model_path))

        self.confidence = confidence
        self.image_size = image_size

        # GPU при наличии CUDA, иначе CPU
        self.device = 0 if torch.cuda.is_available() else "cpu"

    def detect(self, frame):
        """
        Анализирует один кадр.

        Возвращает список обнаружений:
        {
            "class_name": "fire" или "smoke",
            "confidence": число,
            "bbox": (x1, y1, x2, y2)
        }
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
            class_name = str(
                result.names[class_id]
            ).lower()

            # Нам нужны только дым и огонь
            if class_name not in ("fire", "smoke"):
                continue

            confidence = float(box.conf.item())

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            detections.append({
                "class_name": class_name,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2)
            })

        return detections
