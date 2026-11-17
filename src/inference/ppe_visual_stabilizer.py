"""Сглаживание рамок человека и СИЗ только средствами Python.

Одна YOLO-модель по-прежнему вызывается один раз на кадр. Сглаживание
не меняет координаты исходных детекций, используемые для ROI и ByteTrack.
"""
from dataclasses import dataclass, replace

from .types import Detection


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1)*max(0, y2-y1)
    aa = max(0, a[2]-a[0])*max(0, a[3]-a[1])
    bb = max(0, b[2]-b[0])*max(0, b[3]-b[1])
    union = aa+bb-inter
    return inter/union if union > 0 else 0.0


def _coords(det):
    return (det.x1, det.y1, det.x2, det.y2)


def _blend(old, new, alpha):
    return tuple((1-alpha)*a + alpha*b for a,b in zip(old,new))


def _updated(det, bounds):
    x1,y1,x2,y2 = (int(round(v)) for v in bounds)
    return replace(det, x1=x1, y1=y1, x2=x2, y2=y2)


class PersonBoxSmoother:
    """Сглаживает только отображение людей, если ByteTrack сохраняет их ID."""

    def __init__(self, alpha=0.65, forget_frames=45):
        if not 0 < alpha <= 1:
            raise ValueError('person_box_alpha должен быть в диапазоне (0,1]')
        self.alpha = float(alpha)
        self.forget_frames = max(1, int(forget_frames))
        self.frame_index = 0
        self.history = {}  # track_id -> (bbox float, last_frame)

    def next_frame(self):
        self.frame_index += 1
        for k, (_, last) in list(self.history.items()):
            if self.frame_index - last > self.forget_frames:
                del self.history[k]

    def update(self, person: Detection) -> Detection:
        if person.track_id is None:
            return person
        key = person.track_id
        now = tuple(float(v) for v in _coords(person))
        old = self.history.get(key)
        if old and self.frame_index - old[1] <= 2 and _iou(old[0], now) >= 0.20:
            now = _blend(old[0], now, self.alpha)
        self.history[key] = (now, self.frame_index)
        return _updated(person, now)


@dataclass
class _EquipmentState:
    relative_box: tuple
    confidence: float
    class_id: int
    last_seen_frame: int


class PPEVisualSmoother:
    """Привязывает одну каску и один жилет к человеку по Track ID.

    Координаты СИЗ хранятся относительно рамки человека, чтобы кешированная
    каска перемещалась вместе с человеком, а не зависала на экране.
    """

    def __init__(self, alpha=0.55, hold_frames=3, forget_frames=45):
        if not 0 < alpha <= 1:
            raise ValueError('ppe_box_alpha должен быть в диапазоне (0,1]')
        if not 0 <= hold_frames < forget_frames:
            raise ValueError('ppe_hold_frames должен быть >=0 и < forget_frames')
        self.alpha = float(alpha)
        self.hold_frames = int(hold_frames)
        self.forget_frames = int(forget_frames)
        self.frame_index = 0
        self.history = {}  # (track_id, "helmet" | "vest") -> _EquipmentState

    def next_frame(self):
        self.frame_index += 1
        for key, state in list(self.history.items()):
            if self.frame_index - state.last_seen_frame > self.forget_frames:
                del self.history[key]

    @staticmethod
    def _to_relative(item, person):
        w = max(1, person.x2-person.x1)
        h = max(1, person.y2-person.y1)
        return (
            (item.x1-person.x1)/w, (item.y1-person.y1)/h,
            (item.x2-person.x1)/w, (item.y2-person.y1)/h,
        )

    @staticmethod
    def _from_relative(box, person):
        w = max(1, person.x2-person.x1)
        h = max(1, person.y2-person.y1)
        return (
            person.x1+box[0]*w, person.y1+box[1]*h,
            person.x1+box[2]*w, person.y1+box[3]*h,
        )

    def update(self, raw_person, visual_person, class_name, matches):
        """Возвращает (рамка Detection или None, cached:bool).

        Если пропало обнаружение СИЗ на 1-3 кадра, используем последнюю
        относительную рамку. По истечении hold_frames СИЗ считается отсутствующим.
        """
        if raw_person.track_id is None:
            return None, False
        if class_name not in {'helmet','vest'}:
            raise ValueError('Only helmet or vest can be smoothed')
        key = (raw_person.track_id, class_name)
        prev = self.history.get(key)
        best = max(matches, key=lambda d: d.confidence, default=None)
        if best is not None:
            relative = self._to_relative(best, raw_person)
            if prev and self.frame_index - prev.last_seen_frame <= self.hold_frames + 1:
                # Reacquisition at a different location resets smoothing.
                if _iou(prev.relative_box, relative) >= 0.08:
                    relative = _blend(prev.relative_box, relative, self.alpha)
            self.history[key] = _EquipmentState(
                relative_box=relative,
                confidence=float(best.confidence),
                class_id=int(best.class_id),
                last_seen_frame=self.frame_index,
            )
            cached = False
        elif prev is not None and self.frame_index - prev.last_seen_frame <= self.hold_frames:
            cached = True
        else:
            self.history.pop(key, None)
            return None, False

        state = self.history[key]
        result = Detection(
            class_id=state.class_id,
            class_name=class_name,
            confidence=state.confidence,
            x1=0, y1=0, x2=0, y2=0,
            track_id=None,
        )
        result = _updated(result, self._from_relative(state.relative_box, visual_person))
        return result, cached
