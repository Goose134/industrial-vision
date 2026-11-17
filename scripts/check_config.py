"""Проверка YAML и индивидуальных порогов общей модели."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "configuration.yaml"
CLASSES = (
    "person", "car", "truck", "bus", "motorcycle",
    "helmet", "vest", "fire", "smoke", "spill",
)

with CONFIG_PATH.open(encoding="utf-8") as file:
    config = yaml.safe_load(file)

model_config = config["models"]
minimum = float(model_config["detector_confidence"])
limits = model_config["class_confidence"]

if not isinstance(limits, dict) or set(limits) != set(CLASSES):
    raise ValueError("models.class_confidence должен содержать ровно 10 классов")
if not 0 <= minimum <= 1:
    raise ValueError("Неверный detector_confidence")
for cls in CLASSES:
    confidence = float(limits[cls])
    if not minimum <= confidence <= 1:
        raise ValueError(f"Порог {cls} ({confidence}) должен быть от {minimum} до 1")

print("Камера:", config["camera"]["id"])
print("Источник:", config["camera"]["source"])
print("Общий порог:", minimum)
for cls in CLASSES:
    print(f"{cls}: {float(limits[cls]):.2f}")
print("Конфигурация прочитана успешно!")
