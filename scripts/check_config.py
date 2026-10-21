
from pathlib import Path
import yaml

# Определяем корень проекта
project_root = Path(__file__).resolve().parent.parent

# Путь к конфигурации
config_path = project_root / "configs" / "configuration.yaml"

# Открываем и читаем YAML
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# Выводим некоторые настройки
print("Камера:", config["camera"]["id"])
print("Источник видео:", config["camera"]["source"])
print("Порог человека:", config["models"]["person_confidence"])
print("Время cooldown:", config["events"]["cooldown_sec"])

print("Конфигурация прочитана успешно!")
