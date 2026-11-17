from pathlib import Path
from collections import Counter
import yaml

DATASETS = {
    "PPE": Path("datasets/ppe"),
    "Fire/Smoke": Path("datasets/fire_smoke"),
    "Spills": Path("datasets/spills"),
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]

for name, relative_path in DATASETS.items():
    folder = (PROJECT_ROOT / relative_path).resolve()

    print(f"\n{'=' * 40}")
    print(f"Датасет: {name}")
    print(f"Путь: {folder}")

    if not folder.exists():
        print("Папка не найдена!")
        continue

    yaml_file = folder / "data.yaml"

    if not yaml_file.exists():
        print("data.yaml не найден!")
        continue

    with open(yaml_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    classes = config.get("names", {})
    if isinstance(classes, list):
        classes = dict(enumerate(classes))
    else:
        classes = {int(k): v for k, v in classes.items()}

    print("Классы:", classes)

    counter = Counter()
    label_files = [
        p for p in folder.rglob("*.txt")
        if "labels" in p.relative_to(folder).parts
    ]

    for file in label_files:
        for line in file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
                counter[class_id] += 1
            except ValueError:
                print("Ошибка разметки:", file)

    images = [
        p for p in folder.rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    ]

    print("Изображений:", len(images))
    print("Файлов разметки:", len(label_files))
    print("\nКоличество размеченных объектов:")

    for class_id, class_name in classes.items():
        print(f"  {class_id}: {class_name} — {counter[class_id]}")

    unknown = set(counter) - set(classes)
    if unknown:
        print("Неизвестные ID классов:", sorted(unknown))