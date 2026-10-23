
from pathlib import Path

from huggingface_hub import hf_hub_download


def main():
    # Корневая папка нашего проекта
    project_root = Path(__file__).resolve().parent.parent

    # Папка для модели дыма и огня
    model_dir = project_root / "models" / "fire_smoke"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Скачиваем модель Fire/Smoke...")

    # Скачиваем обученные веса с Hugging Face
    model_path = hf_hub_download(
        repo_id="rabahdev/fire-smoke-yolov8n",
        filename="best.pt",
        local_dir=str(model_dir)
    )

    print("Модель успешно скачана!")
    print("Путь:", model_path)


if __name__ == "__main__":
    main()
