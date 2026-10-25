
from pathlib import Path
import cv2
from ultralytics import YOLO

root = Path(__file__).resolve().parents[1]

model = YOLO(str(root / "yolo26n.onnx"))

video = cv2.VideoCapture(
    str(root / "data" / "ppe_test.mp4")
)

success, frame = video.read()
video.release()

if not success:
    raise RuntimeError("Не удалось прочитать видео")

results = model.predict(
    frame,
    imgsz=320,
    device="cpu",
    verbose=False
)

print("ONNX успешно обработал кадр!")
print("Количество обнаружений:", len(results[0].boxes))
