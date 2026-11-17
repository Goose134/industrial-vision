"""Визуальный редактор многоугольника ROI для Industrial Vision.

Работает с одним кадром видеофайла или камеры. Координаты хранятся в
нормированном виде [0..1] и поэтому не зависят от разрешения кадра.
Никакую YOLO-модель этот модуль не загружает.
"""

from __future__ import annotations

import base64
import math
import os
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import cv2


MAX_VERTICES = 32
MIN_POLYGON_AREA = 0.0005  # Доля площади кадра (в нормированных координатах).


def validate_polygon(points: list[list[float]]) -> list[list[float]]:
    """Проверяет корректность ROI; в частности запрещает самопересечения."""
    if not 3 <= len(points) <= MAX_VERTICES:
        raise ValueError(f"Укажи от 3 до {MAX_VERTICES} точек запретной зоны.")
    result = []
    for p in points:
        if len(p) != 2:
            raise ValueError("Каждая точка ROI должна содержать координаты X и Y.")
        x, y = (float(p[0]), float(p[1]))
        if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("Координаты ROI должны находиться от 0 до 1.")
        result.append([x, y])

    # Исключаем совпадающие вершины и очень короткие стороны.
    for i in range(len(result)):
        a, b = result[i], result[(i + 1) % len(result)]
        if math.dist(a, b) < 0.005:
            raise ValueError("Две соседние точки ROI слишком близко друг к другу.")

    area = abs(sum(
        result[i][0] * result[(i + 1) % len(result)][1]
        - result[(i + 1) % len(result)][0] * result[i][1]
        for i in range(len(result))
    )) / 2
    if area < MIN_POLYGON_AREA:
        raise ValueError("ROI слишком маленькая или почти лежит на одной линии.")

    def orientation(a, b, c):
        return ((b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0]))

    def on_segment(a, b, c):
        return (min(a[0], b[0]) - 1e-10 <= c[0] <= max(a[0], b[0]) + 1e-10
                and min(a[1], b[1]) - 1e-10 <= c[1] <= max(a[1], b[1]) + 1e-10)

    def intersect(a, b, c, d):
        o1, o2 = orientation(a, b, c), orientation(a, b, d)
        o3, o4 = orientation(c, d, a), orientation(c, d, b)
        if o1 * o2 < -1e-12 and o3 * o4 < -1e-12:
            return True
        for o, p, q, r in ((o1, a, b, c), (o2, a, b, d),
                           (o3, c, d, a), (o4, c, d, b)):
            if abs(o) < 1e-10 and on_segment(p, q, r):
                return True
        return False

    n = len(result)
    for i in range(n):
        for j in range(i + 1, n):
            # У соседних сторон общая вершина, это корректно.
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if intersect(result[i], result[(i + 1) % n],
                         result[j], result[(j + 1) % n]):
                raise ValueError("Линии ROI пересекаются. Передвинь точки, чтобы убрать пересечение.")
    return [[round(x, 5), round(y, 5)] for x, y in result]


class ROIPolygonEditor:
    """Окно статического кадра с редактированием нормированных точек ROI."""

    def __init__(self, parent: tk.Tk, source: int | str,
                 initial_points: list[list[float]], on_save):
        self.source = source
        self.on_save = on_save
        self.points = [list(map(float, pair)) for pair in initial_points]
        self.dragging_index: int | None = None
        self.photo = None
        self.frame = None
        self.preview_width = 800
        self.preview_height = 450
        self.loading = False
        self.results: queue.Queue = queue.Queue()
        self.duration = 0.0
        self.is_file = isinstance(source, str) and "://" not in source

        self.window = tk.Toplevel(parent)
        self.window.title("Industrial Vision — настройка запретной зоны ROI")
        self.window.geometry("1020x780")
        self.window.minsize(800, 600)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self.window.destroy)

        content = ttk.Frame(self.window, padding=12)
        content.pack(fill="both", expand=True)
        ttk.Label(content, text="Редактор ROI", font=("Segoe UI", 17, "bold")).pack(anchor="w")
        ttk.Label(
            content,
            text="ЛКМ — добавить точку · Перетаскивание точки — изменить форму · "
                 "ПКМ — удалить ближайшую точку · Минимум 3 точки",
            wraplength=960,
        ).pack(anchor="w", pady=(5, 10))

        preview_frame = ttk.Frame(content)
        preview_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(preview_frame, bg="#111827", highlightthickness=0,
                                width=self.preview_width, height=self.preview_height,
                                cursor="crosshair")
        self.canvas.pack(expand=True)
        self.canvas.bind("<Button-1>", self._start_drag_or_add)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._end_drag)
        self.canvas.bind("<Button-3>", self._delete_near)

        self.position_status = tk.StringVar(value="Выбираем кадр…")
        ttk.Label(content, textvariable=self.position_status).pack(anchor="w", pady=(7, 4))

        if self.is_file:
            timeline = ttk.Frame(content)
            timeline.pack(fill="x", pady=(5, 4))
            ttk.Label(timeline, text="Кадр на секунде:").pack(side="left", padx=(0, 8))
            self.at_second = tk.DoubleVar(value=0.0)
            self.scale = ttk.Scale(timeline, orient="horizontal", from_=0, to=1,
                                   variable=self.at_second)
            self.scale.pack(side="left", fill="x", expand=True)
            ttk.Button(timeline, text="Показать кадр", command=self.load_frame).pack(
                side="left", padx=(10, 0))
            self.timeline_label = ttk.Label(timeline, text="0.0 с")
            self.timeline_label.pack(side="left", padx=(8, 0))
            self.at_second.trace_add("write", lambda *_: self.timeline_label.configure(
                text=f"{self.at_second.get():.1f} с"))

        buttons = ttk.Frame(content)
        buttons.pack(fill="x", pady=(12, 0))
        ttk.Button(buttons, text="Отменить последнюю точку", command=self.undo).pack(
            side="left", padx=(0, 6))
        ttk.Button(buttons, text="Очистить ROI", command=self.clear).pack(
            side="left", padx=(0, 6))
        if not self.is_file:
            ttk.Button(buttons, text="Обновить кадр", command=self.load_frame).pack(
                side="left", padx=(0, 6))
        ttk.Button(buttons, text="Отмена", command=self.window.destroy).pack(
            side="right", padx=(6, 0))
        ttk.Button(buttons, text="Сохранить ROI", command=self.save).pack(side="right")

        self._render()
        self.load_frame()
        self._poll_capture()

    def load_frame(self):
        if self.loading:
            return
        self.loading = True
        seconds = self.at_second.get() if self.is_file else 0.0
        self.position_status.set("Получаем кадр из источника…")
        threading.Thread(target=self._read_frame_worker, args=(seconds,), daemon=True).start()

    def _read_frame_worker(self, seconds: float):
        capture = None
        try:
            if isinstance(self.source, int) and os.name == "nt":
                capture = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if not capture.isOpened():
                    capture.release()
                    capture = cv2.VideoCapture(self.source)
            else:
                capture = cv2.VideoCapture(self.source)
            if not capture.isOpened():
                raise RuntimeError("Не получилось открыть видео или камеру. Проверь источник.")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frames / fps if self.is_file and fps > 0 and frames > 0 else 0.0
            if self.is_file and seconds > 0:
                capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            okay, frame = False, None
            for _ in range(5 if not self.is_file else 1):
                okay, frame = capture.read()
                if not okay:
                    break
            if not okay or frame is None:
                raise RuntimeError("Не удалось получить кадр. Попробуй другой момент видео.")
            self.results.put(("image", (frame, duration)))
        except Exception as error:
            self.results.put(("error", str(error)))
        finally:
            if capture is not None:
                capture.release()

    def _poll_capture(self):
        if not self.window.winfo_exists():
            return
        try:
            while True:
                kind, value = self.results.get_nowait()
                self.loading = False
                if kind == "image":
                    frame, duration = value
                    self._show_frame(frame)
                    if self.is_file and duration > 0 and self.duration <= 0:
                        self.duration = duration
                        self.scale.configure(to=duration)
                else:
                    self.position_status.set(value)
                    messagebox.showerror("Не получилось получить кадр", value, parent=self.window)
        except queue.Empty:
            pass
        self.window.after(90, self._poll_capture)

    def _show_frame(self, frame):
        self.frame = frame
        h, w = frame.shape[:2]
        scale = min(940 / w, 520 / h, 2.0)  # Небольшие кадры увеличиваем для удобства.
        self.preview_width = max(1, round(w * scale))
        self.preview_height = max(1, round(h * scale))
        if (self.preview_width, self.preview_height) != (w, h):
            frame = cv2.resize(frame, (self.preview_width, self.preview_height),
                               interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            messagebox.showerror("Ошибка изображения", "Не удалось отобразить кадр.", parent=self.window)
            return
        # Tk PhotoImage понимает PNG: не нужно устанавливать Pillow.
        png_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        self.photo = tk.PhotoImage(data=png_base64, format="png")
        self.canvas.configure(width=self.preview_width, height=self.preview_height)
        self.position_status.set(
            f"Кадр {w}×{h}. Точек ROI: {len(self.points)}. "
            "Точки сохраняются относительно размеров изображения."
        )
        self._render()

    def _point_from_event(self, event):
        x = max(0.0, min(1.0, event.x / max(1, self.preview_width)))
        y = max(0.0, min(1.0, event.y / max(1, self.preview_height)))
        return [x, y]

    def _near_vertex(self, x, y, limit=15):
        for i, point in enumerate(self.points):
            px = point[0] * self.preview_width
            py = point[1] * self.preview_height
            if math.hypot(x - px, y - py) <= limit:
                return i
        return None

    def _start_drag_or_add(self, event):
        if self.frame is None:
            return
        index = self._near_vertex(event.x, event.y)
        if index is not None:
            self.dragging_index = index
        elif len(self.points) < MAX_VERTICES:
            self.points.append(self._point_from_event(event))
            self.dragging_index = len(self.points) - 1
        else:
            messagebox.showinfo("Лимит точек", f"Не больше {MAX_VERTICES} точек.", parent=self.window)
        self._render()

    def _drag(self, event):
        if self.dragging_index is not None:
            self.points[self.dragging_index] = self._point_from_event(event)
            self._render()

    def _end_drag(self, _event):
        self.dragging_index = None

    def _delete_near(self, event):
        if self.frame is None:
            return
        index = self._near_vertex(event.x, event.y, limit=22)
        if index is not None:
            self.points.pop(index)
            self._render()

    def undo(self):
        if self.points:
            self.points.pop()
            self._render()

    def clear(self):
        self.points.clear()
        self._render()

    def _render(self):
        self.canvas.delete("all")
        if self.photo is not None:
            self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        elif self.frame is None:
            self.canvas.create_text(self.preview_width // 2, self.preview_height // 2,
                                    text="Загружаем кадр…", fill="white")

        coords = []
        for p in self.points:
            coords.extend((p[0] * self.preview_width, p[1] * self.preview_height))
        if len(self.points) >= 3:
            self.canvas.create_polygon(
                coords, fill="#ef4444", stipple="gray25",
                outline="#f43f5e", width=3,
            )
        elif len(self.points) == 2:
            self.canvas.create_line(*coords, fill="#f43f5e", width=3)
        for idx, p in enumerate(self.points, 1):
            px, py = p[0] * self.preview_width, p[1] * self.preview_height
            self.canvas.create_oval(px - 6, py - 6, px + 6, py + 6,
                                    fill="#f8fafc", outline="#be123c", width=2)
            self.canvas.create_text(px + 10, py - 11, text=str(idx), anchor="w",
                                    fill="#f8fafc", font=("Segoe UI", 11, "bold"))
        if self.frame is not None:
            self.position_status.set(
                f"Точек ROI: {len(self.points)}. ЛКМ добавить/перетащить, ПКМ удалить."
            )

    def save(self):
        try:
            if self.frame is None:
                raise ValueError("Сначала дождись изображения для настройки зоны.")
            points = validate_polygon(self.points)
        except (ValueError, TypeError) as error:
            messagebox.showerror("Некорректная зона", str(error), parent=self.window)
            return
        if self.on_save(points):
            self.window.destroy()
