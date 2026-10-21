
from ultralytics import YOLO

from .types import Detection


class Detector:
    """
    Класс для обнаружения людей и транспорта.
    """

    def __init__(self, model_path, confidence=0.5, image_size=640):

        # Загружаем предварительно обученную модель
        self.model = YOLO(model_path)

        # Минимальная уверенность детекции
        self.confidence = confidence

        # Размер изображения для нейросети
        self.image_size = image_size

        # Классы объектов, которые нас интересуют
        self.target_classes = {
            "person",
            "car",
            "truck",
            "bus",
            "motorcycle"
        }

    def predict(self, frame):
        """
        Принимает изображение и возвращает список объектов.
        """

        # Передаём кадр нейросети
        result = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False
        )[0]

        # Названия классов модели
        names = result.names

        # Список обнаруженных объектов
        detections = []

        # Перебираем прямоугольники обнаруженных объектов
        for box in result.boxes:

            # Получаем ID класса
            class_id = int(box.cls.item())

            # Название класса
            class_name = str(names[class_id])

            # Пропускаем ненужные классы
            if class_name not in self.target_classes:
                continue

            # Уверенность модели
            confidence = float(box.conf.item())

            # Координаты прямоугольника
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # Создаём объект Detection
            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2
            )

            # Добавляем объект в список
            detections.append(detection)

        return detections
