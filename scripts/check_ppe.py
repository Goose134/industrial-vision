
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def main():
    # Определяем корневую папку проекта
    project_root = Path(__file__).resolve().parent.parent

    # Путь к нашей обученной модели
    model_path = (
        project_root
        / "runs"
        / "ppe"
        / "experiment_01"
        / "weights"
        / "best.pt"
    )

    # Путь к тестовому видео
    video_path = project_root / "data" / "ppe_test.mp4"

    # Проверяем существование файлов
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Не найдена модель: {model_path}"
        )

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Не найдено видео: {video_path}"
        )

    # Определяем устройство для нейросети
    device = 0 if torch.cuda.is_available() else "cpu"

    print("Устройство:", device)

    # Загружаем обученную модель
    model = YOLO(str(model_path))

    print("Модель PPE успешно загружена")

    # Открываем видео
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть PPE-видео")

    try:
        while True:
            # Получаем очередной кадр
            ok, frame = cap.read()

            if not ok:
                print("Видео завершено")
                break

            # Запускаем обученную нейросеть
            result = model.predict(
                source=frame,
                conf=0.35,
                imgsz=640,
                device=device,
                verbose=False
            )[0]

            # Подсчитываем каски и жилеты
            helmets = 0
            vests = 0

            # Перебираем найденные объекты
            for box in result.boxes:

                class_id = int(box.cls.item())
                class_name = str(result.names[class_id])
                confidence = float(box.conf.item())

                # Координаты объекта
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                # Выбираем цвет
                if class_name == "helmet":
                    color = (0, 255, 0)
                    helmets += 1

                elif class_name == "vest":
                    color = (255, 255, 0)
                    vests += 1

                else:
                    # Пока остальные классы не отображаем
                    continue

                # Рисуем Bounding Box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # Подписываем объект
                label = f"{class_name}: {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            # Показываем количество объектов
            cv2.putText(
                frame,
                f"Helmets: {helmets}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Vests: {vests}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2
            )

            # Показываем кадр
            cv2.imshow(
                "Industrial Vision - PPE Detection",
                frame
            )

            # Выход по клавише Q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Проверка PPE завершена")


if __name__ == "__main__":
    main()
