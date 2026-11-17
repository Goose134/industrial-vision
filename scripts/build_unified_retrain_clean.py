"""Build YOLO data lists excluding OLD fire/smoke examples, without copying images.

Install as industrial-vision/scripts/build_unified_retrain_clean.py
Run from the project root: python scripts/build_unified_retrain_clean.py

Uses the verified datasets/unified_finetune/manifest.csv created previously.
Does not modify any existing images, labels, runs, or model weights.
"""

from collections import Counter
from pathlib import Path
import csv
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'datasets' / 'unified_finetune'
MANIFEST = INPUT / 'manifest.csv'
OUTPUT = ROOT / 'datasets' / 'unified_retrain_clean'
SPLITS = ('train', 'val', 'test')
CLASSES = [
    'person', 'car', 'truck', 'bus', 'motorcycle',
    'helmet', 'vest', 'fire', 'smoke', 'spill'
]


def label_classes(path):
    result = []
    for n, text in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not text.strip():
            continue
        parts = text.split()
        if len(parts) != 5:
            raise ValueError(f'{path}, строка {n}: должно быть 5 значений')
        try:
            number = int(parts[0])
        except ValueError as exc:
            raise ValueError(f'{path}, строка {n}: неверный ID класса') from exc
        if number not in range(len(CLASSES)):
            raise ValueError(f'{path}, строка {n}: неизвестный класс {number}')
        result.append(number)
    return result


def is_old_fire_smoke(row, ids):
    if row['source'] != 'old_unified':
        return False
    # Primary rule: old unified image has annotated fire or smoke.
    if 7 in ids or 8 in ids:
        return True
    # Also catch background/negative examples explicitly named for the OLD
    # fire/smoke source, where no fire/smoke annotation was present.
    original_name = Path(row['original_image'].replace('\\', '/')).name.lower()
    return any(tag in original_name for tag in ('fire_smoke', 'fire-smoke', 'firesmoke'))


def main():
    if not MANIFEST.is_file():
        raise SystemExit(f'Нет {MANIFEST}. Сначала нужна готовая сборка unified_finetune.')
    if OUTPUT.exists():
        raise SystemExit(f'Папка {OUTPUT} уже существует. Скрипт ничего не перезаписывает.')

    selected = {split: [] for split in SPLITS}
    classes_by_split = {split: Counter() for split in SPLITS}
    sources_kept = Counter()
    excluded = Counter()
    excluded_mixed = Counter()
    seen = set()
    total = 0

    print('ФОРМИРУЕМ НОВЫЙ НАБОР БЕЗ СТАРОГО FIRE/SMOKE')
    print('Исходные изображения и старые модели не изменяются.\n')

    with MANIFEST.open('r', newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        required = {'source', 'split', 'original_image', 'new_image'}
        if not required.issubset(reader.fieldnames or []):
            raise SystemExit(f'В manifest.csv нет столбцов: {sorted(required - set(reader.fieldnames or []))}')
        for row in reader:
            total += 1
            split = row['split']
            if split not in SPLITS:
                raise SystemExit(f'Некорректный split: {split}')
            name = row['new_image']
            image_path = INPUT / 'images' / split / name
            label_path = INPUT / 'labels' / split / (Path(name).stem + '.txt')
            if not image_path.is_file() or not label_path.is_file():
                raise SystemExit(f'Нет пары изображения/разметки: {image_path} / {label_path}')
            try:
                ids = label_classes(label_path)
            except (ValueError, UnicodeError, OSError) as exc:
                raise SystemExit(str(exc)) from exc

            if is_old_fire_smoke(row, ids):
                excluded[split] += 1
                if any(c not in (7, 8) for c in ids):
                    excluded_mixed[split] += 1
                continue

            if row['source'] == 'old_unified' and any(c in (7, 8) for c in ids):
                raise SystemExit(f'Неисключённый старый огонь/дым: {label_path}')

            if image_path in seen:
                raise SystemExit(f'Повтор записи в manifest: {image_path}')
            seen.add(image_path)
            selected[split].append(image_path.resolve().as_posix())
            classes_by_split[split].update(ids)
            sources_kept[(row['source'], split)] += 1

    if not selected['train'] or not selected['val'] or not selected['test']:
        raise SystemExit('Один из наборов оказался пустым — обучение не запускай.')
    missing = [CLASSES[i] for i in range(len(CLASSES)) if classes_by_split['train'][i] == 0]
    if missing:
        raise SystemExit(f'В train нет размеченных классов: {missing}')
    if not sources_kept[('new_fire_smoke', 'train')]:
        raise SystemExit('Не попали НОВЫЕ изображения огня и дыма. Остановлено.')

    # No image copy: YOLO reads existing unified_finetune paths from txt lists.
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        (OUTPUT / f'{split}.txt').write_text(
            '\n'.join(selected[split]) + '\n', encoding='utf-8'
        )

    data_yaml = {
        'path': OUTPUT.resolve().as_posix(),
        'train': 'train.txt',
        'val': 'val.txt',
        'test': 'test.txt',
        'nc': len(CLASSES),
        'names': CLASSES,
    }
    (OUTPUT / 'data.yaml').write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False),
        encoding='utf-8'
    )

    lines = [
        f'Исходных изображений по манифесту: {total}',
        f'Исключено старых изображений Fire/Smoke: {sum(excluded.values())}',
        f'Среди исключённых были другие размеченные классы: {sum(excluded_mixed.values())}',
    ]
    for split in SPLITS:
        lines.append(f'{split}: {len(selected[split])} изображений; исключено {excluded[split]}')
    lines.append('Оставшиеся источники:')
    for (source, split), count in sorted(sources_kept.items()):
        lines.append(f'  {source} / {split}: {count}')
    lines.append('Классы в train:')
    for i, name in enumerate(CLASSES):
        lines.append(f'  {i} {name}: {classes_by_split["train"][i]} объектов')
    lines.append(f'Датасет: {OUTPUT}')
    lines.append('Оригиналы и веса не изменялись.')
    (OUTPUT / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print('\nГОТОВО. Сначала проверь цифры, затем запускай обучение.')


if __name__ == '__main__':
    main()
