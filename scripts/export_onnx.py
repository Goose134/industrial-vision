
from pathlib import Path
from ultralytics import YOLO

# Корневая папка проекта
project_root = Path(__file__).resolve().parents[1]

# Путь к базовой модели YOLO
model_path = project_root / "yolo26n.pt"

if not model_path.is_file():
    raise FileNotFoundError(
        f"Модель не найдена: {model_path}"
    )

print("Загружаем PyTorch-модель...")

model = YOLO(str(model_path))

print("Начинаем экспорт в ONNX...")

# Экспорт с разрешением 320x320 для CPU ноутбука
onnx_path = model.export(
    format="onnx",
    imgsz=320,
    device="cpu",
    dynamic=False,
    simplify=True
)

print("Экспорт завершён!")
print("ONNX-модель:", onnx_path)
