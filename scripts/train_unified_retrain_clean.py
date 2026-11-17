"""Train 10-class YOLO26n anew from official pretrained weights, not old unified best.pt.

Install as industrial-vision/scripts/train_unified_retrain_clean.py
Run from project root: python scripts/train_unified_retrain_clean.py
"""
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets' / 'unified_retrain_clean' / 'data.yaml'
PROJECT = ROOT / 'runs' / 'retrain'
NAME = 'unified_clean_50'
EXPECTED = [
    'person', 'car', 'truck', 'bus', 'motorcycle',
    'helmet', 'vest', 'fire', 'smoke', 'spill'
]


def main():
    if not DATA.is_file():
        raise SystemExit(f'Сначала создай датасет: {DATA}')
    if (PROJECT / NAME).exists():
        raise SystemExit(f'Результат уже существует: {PROJECT / NAME}. Не перезаписываем.')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA недоступна. Проверь активную .venv и драйвер GPU.')
    cfg = yaml.safe_load(DATA.read_text(encoding='utf-8'))
    if cfg.get('names') != EXPECTED or cfg.get('nc') != 10:
        raise SystemExit(f'Неожиданные классы: {cfg.get("names")}')

    print('Датасет:', DATA)
    print('GPU:', torch.cuda.get_device_name(0))
    print('Стартуем с официальных предобученных весов yolo26n.pt, НЕ с unified/best.pt')
    print('Результаты:', PROJECT / NAME, flush=True)

    model = YOLO('yolo26n.pt')
    model.train(
        data=str(DATA), epochs=50, imgsz=640, batch=8,
        device=0, workers=2, cache=False,
        optimizer='AdamW', lr0=0.001, lrf=0.01,
        warmup_epochs=3.0, cos_lr=True,
        patience=15, amp=True, seed=42,
        resume=False, save=True, save_period=10,
        project=str(PROJECT), name=NAME,
        exist_ok=False, verbose=True,
    )
    print('Лучшие веса:', PROJECT / NAME / 'weights' / 'best.pt')
    print('Старые ONNX и best.pt не изменены.')


if __name__ == '__main__':
    main()
