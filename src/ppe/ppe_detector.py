
from pathlib import Path

import cv2
from ultralytics import YOLO


class PPEDetector:
    """
    Проверяет наличие каски и жилета
    у отдельно обнаруженного человека.
    """

    def __init__(
        self,
        model_path,
        confidence=0.35,
        min_person_height=120
    ):
        # Проверяем существование обученной модели
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"Не найдена PPE-модель: {model_path}"
            )

        # Загружаем нашу обученную YOLO
        self.model = YOLO(str(model_path))

        self.confidence = confidence

        # Минимальная высота человека в пикселях
        self.min_person_height = min_person_height

    def check_person(self, frame, person):
        """
        Проверяет каску и жилет у одного человека.

        Возвращает:
        - None, если человек слишком мал;
        - словарь с результатами проверки.
        """

        height, width = frame.shape[:2]

        # Ограничиваем координаты границами кадра
        x1 = max(0, person.x1)
        y1 = max(0, person.y1)
        x2 = min(width, person.x2)
        y2 = min(height, person.y2)

        # Проверяем размер обнаруженного человека
        person_height = y2 - y1

        if person_height < self.min_person_height:
            return None

        # Вырезаем изображение человека
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        # Передаём изображение в PPE-модель
        result = self.model.predict(
            source=crop,
            conf=self.confidence,
            imgsz=640,
            verbose=False
        )[0]

        # Начальные результаты
        has_helmet = False
        has_vest = False

        # Координаты обнаруженных СИЗ
        helmets = []
        vests = []

        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = str(result.names[class_id])

            confidence = float(box.conf.item())

            bx1, by1, bx2, by2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # Координаты объекта внутри crop
            center_x = (bx1 + bx2) / 2
            center_y = (by1 + by2) / 2

            crop_width = x2 - x1

            # Каска должна находиться в верхней части
            # изображения человека
            if (
                class_name == "helmet"
                and center_y < person_height * 0.40
            ):
                has_helmet = True

                helmets.append({
                    "confidence": confidence,
                    "bbox": (
                        bx1 + x1,
                        by1 + y1,
                        bx2 + x1,
                        by2 + y1
                    )
                })

            # Жилет должен находиться в области туловища
            elif (
                class_name == "vest"
                and person_height * 0.15
                < center_y
                < person_height * 0.80
            ):
                has_vest = True

                vests.append({
                    "confidence": confidence,
                    "bbox": (
                        bx1 + x1,
                        by1 + y1,
                        bx2 + x1,
                        by2 + y1
                    )
                })

        return {
            "track_id": person.track_id,
            "has_helmet": has_helmet,
            "has_vest": has_vest,
            "helmet_missing": not has_helmet,
            "vest_missing": not has_vest,
            "helmets": helmets,
            "vests": vests
        }
