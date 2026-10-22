
from ultralytics import YOLO

from .types import Detection


class Detector:
    """
    Модуль обнаружения и отслеживания объектов.
    """

    def __init__(self, model_path, confidence=0.35, image_size=640):

        # Загружаем модель нейросети
        self.model = YOLO(model_path)

        # Порог уверенности модели
        self.confidence = confidence

        # Размер входного изображения
        self.image_size = image_size

        # Классы людей и транспорта
        self.target_classes = {
            "person",
            "car",
            "truck",
            "bus",
            "motorcycle"
        }

    def predict(self, frame):
        """
        Обычное обнаружение объектов без tracking.
        Сохраняем для совместимости с прошлым этапом.
        """
        result = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False
        )[0]

        return self._convert_results(result)

    def track(self, frame):
        """
        Обнаружение и отслеживание объектов.
        """

        result = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False
        )[0]

        return self._convert_results(result)

    def _convert_results(self, result):
        """
        Преобразует результат YOLO в наши Detection.
        """

        detections = []

        # Если прямоугольников нет, возвращаем пустой список
        if result.boxes is None:
            return detections

        names = result.names

        for box in result.boxes:

            class_id = int(box.cls.item())
            class_name = str(names[class_id])

            # Пропускаем ненужные классы
            if class_name not in self.target_classes:
                continue

            confidence = float(box.conf.item())

            # Координаты объекта
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # По умолчанию идентификатор отсутствует
            track_id = None

            # Проверяем, выдал ли трекер идентификатор
            if box.id is not None:
                track_id = int(box.id.item())

            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                track_id=track_id
            )

            detections.append(detection)

        return detections
