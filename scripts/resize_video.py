
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]

input_path = ROOT / "data" / "fire_test.mp4"
output_path = ROOT / "data" / "spill_test_480p.mp4"

cap = cv2.VideoCapture(str(input_path))

if not cap.isOpened():
    raise RuntimeError("Не удалось открыть исходное видео")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25

width = 854
height = 480

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    cap.release()
    raise RuntimeError("Не удалось создать выходное видео")

while True:
    success, frame = cap.read()

    if not success:
        break

    resized = cv2.resize(
        frame,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    out.write(resized)

cap.release()
out.release()

print("Готово!")
print("Видео сохранено:", output_path)
