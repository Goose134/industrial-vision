"""Industrial Vision — графическая настройка и запуск проекта.

Запуск из корня репозитория:
    python -m src.gui.app

Текущий этап: настройки, управление процессом и консоль событий.
Изображение пока выводится в отдельное окно OpenCV, как и при python main.py.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime

from .roi_editor import ROIPolygonEditor
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "configuration.yaml"
MAIN_PATH = PROJECT_ROOT / "main.py"

CLASSES = (
    ("person", "Человек"),
    ("car", "Легковой автомобиль"),
    ("truck", "Грузовик"),
    ("bus", "Автобус"),
    ("motorcycle", "Мотоцикл"),
    ("helmet", "Каска"),
    ("vest", "Жилет"),
    ("fire", "Огонь"),
    ("smoke", "Дым"),
    ("spill", "Разлив"),
)
MODULES = (
    ("roi_enabled", "Запретная зона (ROI)"),
    ("ppe_enabled", "Каски и жилеты (PPE)"),
    ("fire_smoke_enabled", "Огонь и дым"),
    ("spills_enabled", "Разливы"),
)
NMS_CLASSES = (
    ("person", "Люди"),
    ("helmet", "Каски"),
    ("vest", "Жилеты"),
    ("fire", "Огонь"),
    ("smoke", "Дым"),
    ("spill", "Разливы"),
)
DEFAULT_CONF = {
    "person": 0.35, "car": 0.20, "truck": 0.20,
    "bus": 0.20, "motorcycle": 0.20, "helmet": 0.25,
    "vest": 0.25, "fire": 0.15, "smoke": 0.10, "spill": 0.20,
}
DEFAULT_NMS = {
    "person": 0.55, "helmet": 0.65, "vest": 0.65,
    "fire": 0.50, "smoke": 0.60, "spill": 0.50,
}


def read_config() -> dict:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Не найден файл настроек: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("configuration.yaml должен содержать YAML-словарь.")
    return cfg


def parse_fraction(value: str, title: str, minimum=0.0, maximum=1.0) -> float:
    try:
        number = float(value.strip().replace(",", "."))
    except (TypeError, ValueError, AttributeError):
        raise ValueError(f"{title}: требуется число от {minimum} до {maximum}.") from None
    if not (minimum <= number <= maximum):
        raise ValueError(f"{title}: значение должно быть от {minimum} до {maximum}.")
    return round(number, 4)


def relative_or_absolute(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


class IndustrialVisionGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Industrial Vision — управление и настройки")
        self.root.geometry("1130x760")
        self.root.minsize(990, 650)
        self.root.configure(bg="#f3f6fb")

        self.process: subprocess.Popen | None = None
        self.process_done = False
        self.stop_requested = False
        self.outbox: queue.Queue[str] = queue.Queue()

        self.config_data = read_config()
        cfg = self.config_data
        camera = cfg.get("camera", {})
        models = cfg.get("models", {})
        modules = cfg.get("modules", {})
        tracking = cfg.get("tracking", {})
        nms = tracking.get("class_nms_iou", {}) or {}
        source = camera.get("source", "data/car.mp4")

        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            initial_mode = "Веб-камера"
        elif isinstance(source, str) and "://" in source:
            initial_mode = "IP-камера (RTSP)"
        else:
            initial_mode = "Видеофайл"

        self.mode = tk.StringVar(value=initial_mode)
        self.video_path = tk.StringVar(value=str(source) if initial_mode == "Видеофайл" else "")
        self.camera_index = tk.StringVar(value=str(source) if initial_mode == "Веб-камера" else "0")
        self.rtsp_address = tk.StringVar(value=str(source) if initial_mode == "IP-камера (RTSP)" else "")
        self.realtime_video = tk.BooleanVar(value=bool(cfg.get("playback", {}).get("realtime_video", True)))
        self.module_values = {
            key: tk.BooleanVar(value=bool(modules.get(key, True))) for key, _ in MODULES
        }
        self.roi_visible = tk.BooleanVar(value=bool(modules.get("roi_visible", True)))
        self.roi_points = copy.deepcopy(cfg.get("zones", {}).get("restricted_01", []))
        self.roi_summary = tk.StringVar(value=f"Точек: {len(self.roi_points)}")
        self.general_conf = tk.StringVar(value=f"{float(models.get('detector_confidence', 0.05)):.2f}")
        self.class_values = {
            key: tk.StringVar(value=f"{float(models.get('class_confidence', {}).get(key, DEFAULT_CONF[key])):.2f}")
            for key, _ in CLASSES
        }
        self.model_path = tk.StringVar(value=str(models.get("detector_path", "models/unified/unified_clean640.onnx")))
        self.image_size = tk.StringVar(value=str(models.get("image_size", 640)))
        self.vehicle_nms_enabled = tk.BooleanVar(value=bool(tracking.get("vehicle_nms_enabled", True)))
        self.class_nms_enabled = tk.BooleanVar(value=bool(tracking.get("class_nms_enabled", True)))
        self.stabilize_vehicle = tk.BooleanVar(value=bool(tracking.get("stabilize_vehicle_classes", True)))
        self.visual_smoothing = tk.BooleanVar(value=bool(cfg.get("visual_smoothing", {}).get("enabled", True)))
        self.vehicle_iou = tk.StringVar(value=f"{float(tracking.get('vehicle_nms_iou', 0.60)):.2f}")
        self.nms_values = {
            key: tk.StringVar(value=f"{float(nms.get(key, DEFAULT_NMS[key])):.2f}")
            for key, _ in NMS_CLASSES
        }

        self._build_styles()
        self._build_layout()
        self._update_source_fields()
        self._poll_output()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_styles(self):
        st = ttk.Style(self.root)
        if "clam" in st.theme_names():
            st.theme_use("clam")
        st.configure("TFrame", background="#f3f6fb")
        st.configure("TLabel", background="#f3f6fb", foreground="#25334a", font=("Segoe UI", 10))
        st.configure("Section.TLabel", font=("Segoe UI", 11, "bold"), foreground="#152238")
        st.configure("Hint.TLabel", font=("Segoe UI", 9), foreground="#64748b")
        st.configure("TCheckbutton", background="#f3f6fb", font=("Segoe UI", 10))
        st.configure("TLabelframe", background="#f3f6fb", bordercolor="#d7e0ea")
        st.configure("TLabelframe.Label", background="#f3f6fb", font=("Segoe UI", 10, "bold"))
        st.configure("TNotebook", background="#f3f6fb", borderwidth=0)
        st.configure("TNotebook.Tab", padding=(15, 10), font=("Segoe UI", 10))
        st.configure("Primary.TButton", padding=(12, 9), font=("Segoe UI", 10, "bold"))
        st.configure("TButton", padding=(10, 8), font=("Segoe UI", 10))

    def _build_layout(self):
        header = ttk.Frame(self.root, padding=(22, 17, 22, 8))
        header.pack(fill="x")
        ttk.Label(header, text="Industrial Vision", font=("Segoe UI", 20, "bold"),
                  foreground="#182d54").pack(anchor="w")
        ttk.Label(header, text="Одна YOLO26n · настройки, камеры, события и запуск",
                  style="Hint.TLabel").pack(anchor="w", pady=(2, 0))

        main = ttk.Frame(self.root, padding=(20, 7, 20, 18))
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=5)
        main.columnconfigure(1, weight=4)
        main.rowconfigure(0, weight=1)

        self.tabs = ttk.Notebook(main)
        self.tabs.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        source_tab = ttk.Frame(self.tabs, padding=20)
        modules_tab = ttk.Frame(self.tabs, padding=20)
        thresholds_tab = ttk.Frame(self.tabs, padding=20)
        other_tab = ttk.Frame(self.tabs, padding=20)
        self.tabs.add(source_tab, text="Источник")
        self.tabs.add(modules_tab, text="Модули")
        self.tabs.add(thresholds_tab, text="Confidence")
        self.tabs.add(other_tab, text="NMS и модель")
        self._build_source_tab(source_tab)
        self._build_modules_tab(modules_tab)
        self._build_thresholds_tab(thresholds_tab)
        self._build_other_tab(other_tab)

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(3, weight=1)
        ttk.Label(right, text="Управление", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(3, 12))

        controls = ttk.Frame(right)
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure((0, 1), weight=1)
        ttk.Button(controls, text="Сохранить настройки", command=self.save_settings).grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 9))
        self.start_button = ttk.Button(controls, text="▶ Запустить", style="Primary.TButton",
                                       command=self.start)
        self.start_button.grid(row=1, column=0, sticky="ew", padx=(0, 5))
        self.stop_button = ttk.Button(controls, text="■ Остановить", command=self.stop,
                                      state="disabled")
        self.stop_button.grid(row=1, column=1, sticky="ew", padx=(5, 0))

        self.status = tk.StringVar(value="Готово к запуску")
        ttk.Label(right, textvariable=self.status, style="Hint.TLabel",
                  wraplength=350).grid(row=2, column=0, sticky="w", pady=(14, 13))
        log_group = ttk.LabelFrame(right, text="Сообщения и события", padding=8)
        log_group.grid(row=3, column=0, sticky="nsew")
        log_group.rowconfigure(0, weight=1)
        log_group.columnconfigure(0, weight=1)
        self.log = tk.Text(log_group, wrap="word", font=("Consolas", 9),
                           bg="#132034", fg="#d7e4f6", insertbackground="white",
                           relief="flat", padx=10, pady=10, state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(log_group, orient="vertical", command=self.log.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=sb.set)
        ttk.Button(right, text="Открыть папку доказательств", command=self.open_evidence).grid(
            row=4, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(right,
                  text="Пока видео открывается в отдельном окне OpenCV. "
                       "Настройки применяются после перезапуска обработки.",
                  style="Hint.TLabel", wraplength=365).grid(row=5, column=0, sticky="w", pady=(10, 0))
        self._append_log(f"Проект: {PROJECT_ROOT}\n")
        self._append_log("Настройки загружены из configuration.yaml\n")

    def _build_source_tab(self, tab):
        tab.columnconfigure(0, weight=1)
        ttk.Label(tab, text="Откуда получать изображение", style="Section.TLabel").pack(anchor="w", pady=(0, 15))
        ttk.Label(tab, text="Тип источника").pack(anchor="w")
        combo = ttk.Combobox(tab, state="readonly", textvariable=self.mode,
                             values=("Видеофайл", "Веб-камера", "IP-камера (RTSP)"))
        combo.pack(fill="x", pady=(5, 18))
        combo.bind("<<ComboboxSelected>>", lambda _e: self._update_source_fields())
        self.video_frame = ttk.Frame(tab)
        ttk.Label(self.video_frame, text="Путь к видео").pack(anchor="w")
        video_row = ttk.Frame(self.video_frame)
        video_row.pack(fill="x", pady=(6, 4))
        ttk.Entry(video_row, textvariable=self.video_path).pack(side="left", fill="x", expand=True)
        ttk.Button(video_row, text="Обзор…", command=self.browse_video).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(self.video_frame, text="Воспроизводить с исходной скоростью",
                        variable=self.realtime_video).pack(anchor="w", pady=(12, 0))
        self.camera_frame = ttk.Frame(tab)
        ttk.Label(self.camera_frame, text="Номер веб-камеры (0, 1, 2…)").pack(anchor="w")
        ttk.Entry(self.camera_frame, textvariable=self.camera_index).pack(fill="x", pady=6)
        ttk.Label(self.camera_frame, text="0 — обычно первая встроенная или USB-камера.",
                  style="Hint.TLabel").pack(anchor="w")
        self.rtsp_frame = ttk.Frame(tab)
        ttk.Label(self.rtsp_frame, text="RTSP / HTTP-адрес камеры").pack(anchor="w")
        ttk.Entry(self.rtsp_frame, textvariable=self.rtsp_address).pack(fill="x", pady=6)
        ttk.Label(self.rtsp_frame, text="Адрес и пароль камеры (если есть) сохраняются в YAML.",
                  style="Hint.TLabel").pack(anchor="w")
        ttk.Label(tab,
                  text="После сохранения нажми «Запустить». Камера работает в режиме свежих кадров; "
                       "для видеофайла сохраняется последовательное чтение.",
                  style="Hint.TLabel", wraplength=400).pack(anchor="w", pady=(32, 0))

    def _update_source_fields(self):
        for widget in (self.video_frame, self.camera_frame, self.rtsp_frame):
            widget.pack_forget()
        chosen = {"Видеофайл": self.video_frame, "Веб-камера": self.camera_frame,
                  "IP-камера (RTSP)": self.rtsp_frame}[self.mode.get()]
        chosen.pack(fill="x")

    def _build_modules_tab(self, tab):
        ttk.Label(tab, text="Включить или отключить функции", style="Section.TLabel").pack(
            anchor="w", pady=(0, 17))
        for key, label in MODULES:
            ttk.Checkbutton(tab, text=label, variable=self.module_values[key]).pack(
                anchor="w", pady=8)

        ttk.Separator(tab, orient="horizontal").pack(fill="x", pady=(9, 13))
        ttk.Label(tab, text="Редактор запретной зоны", style="Section.TLabel").pack(anchor="w")
        ttk.Checkbutton(
            tab, text="Показывать контур ROI на видео", variable=self.roi_visible
        ).pack(anchor="w", pady=(10, 6))
        ttk.Label(
            tab,
            text="Если убрать галочку, нарушения продолжат фиксироваться, "
                 "пока включена сама запретная зона.",
            style="Hint.TLabel", wraplength=400,
        ).pack(anchor="w", pady=(0, 10))
        controls = ttk.Frame(tab)
        controls.pack(fill="x", pady=(3, 0))
        ttk.Button(
            controls, text="Настроить ROI мышкой…", command=self.open_roi_editor
        ).pack(side="left")
        ttk.Label(controls, textvariable=self.roi_summary, style="Hint.TLabel").pack(
            side="left", padx=(10, 0))
        ttk.Label(tab,
                  text="Изменения ROI сохраняются в configuration.yaml. "
                       "Отключение модуля не останавливает общий запуск YOLO.",
                  style="Hint.TLabel", wraplength=400).pack(anchor="w", pady=(18, 0))

    def _build_thresholds_tab(self, tab):
        ttk.Label(tab, text="Уверенность обнаружения", style="Section.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))
        tab.columnconfigure(0, weight=1)
        ttk.Label(tab, text="Общий минимум YOLO").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Spinbox(tab, from_=0.001, to=1.0, increment=0.01, width=9,
                    textvariable=self.general_conf).grid(row=1, column=1, sticky="e")
        ttk.Separator(tab, orient="horizontal").grid(row=2, column=0, columnspan=2,
                                                       sticky="ew", pady=13)
        for index, (key, label) in enumerate(CLASSES, start=3):
            ttk.Label(tab, text=label).grid(row=index, column=0, sticky="w", pady=5)
            ttk.Spinbox(tab, from_=0.001, to=1.0, increment=0.05, width=9,
                        textvariable=self.class_values[key]).grid(row=index, column=1,
                                                                   sticky="e", pady=5)
        ttk.Label(tab, text="Каждый порог класса должен быть не ниже общего минимума YOLO.",
                  style="Hint.TLabel", wraplength=420).grid(row=14, column=0,
                                                               columnspan=2, sticky="w", pady=(17, 0))

    def _build_other_tab(self, tab):
        tab.columnconfigure(0, weight=1)
        ttk.Label(tab, text="Модель и подавление повторных рамок", style="Section.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        ttk.Label(tab, text="ONNX-модель").grid(row=1, column=0, sticky="w")
        ttk.Entry(tab, textvariable=self.model_path).grid(row=2, column=0,
                                                          columnspan=2, sticky="ew", pady=5)
        ttk.Button(tab, text="Обзор…", command=self.browse_model).grid(row=2, column=2, padx=(5, 0))
        ttk.Label(tab, text="Размер входа модели").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(tab, textvariable=self.image_size, values=("320", "416", "640"),
                     width=10).grid(row=3, column=1, sticky="e", pady=(8, 0))
        ttk.Label(tab, text="Для ONNX с фиксированным входом размер должен совпадать с экспортом.",
                  style="Hint.TLabel", wraplength=420).grid(row=4, column=0,
                                                               columnspan=3, sticky="w", pady=(7, 8))
        ttk.Separator(tab).grid(row=5, column=0, columnspan=3, sticky="ew", pady=7)
        ttk.Checkbutton(tab, text="NMS транспорта (между car/truck/bus/motorcycle)",
                        variable=self.vehicle_nms_enabled).grid(row=6, column=0,
                                                                 columnspan=3, sticky="w", pady=4)
        ttk.Label(tab, text="IoU транспорта").grid(row=7, column=0, sticky="w", pady=4)
        ttk.Spinbox(tab, from_=0.05, to=0.99, increment=0.05, width=9,
                    textvariable=self.vehicle_iou).grid(row=7, column=1, sticky="e")
        ttk.Checkbutton(tab, text="Дополнительный NMS внутри отдельных классов",
                        variable=self.class_nms_enabled).grid(row=8, column=0,
                                                               columnspan=3, sticky="w", pady=(7, 4))
        for idx, (key, label) in enumerate(NMS_CLASSES, start=9):
            ttk.Label(tab, text=f"IoU · {label}").grid(row=idx, column=0, sticky="w", pady=3)
            ttk.Spinbox(tab, from_=0.05, to=0.99, increment=0.05, width=9,
                        textvariable=self.nms_values[key]).grid(row=idx, column=1,
                                                                 sticky="e", pady=3)
        ttk.Checkbutton(tab, text="Стабилизировать класс транспорта",
                        variable=self.stabilize_vehicle).grid(row=15, column=0,
                                                              columnspan=3, sticky="w", pady=(12, 3))
        ttk.Checkbutton(tab, text="Сглаживание рамок людей и PPE",
                        variable=self.visual_smoothing).grid(row=16, column=0,
                                                              columnspan=3, sticky="w")

    def _preview_source(self):
        """Возвращает источник без изменения конфигурации и запуска детектора."""
        mode = self.mode.get()
        if mode == "Видеофайл":
            value = self.video_path.get().strip()
            if not value:
                raise ValueError("Для редактирования ROI сначала выбери видеофайл.")
            path = Path(value)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if not path.is_file():
                raise ValueError(f"Видеофайл не найден: {path}")
            return str(path.resolve())
        if mode == "Веб-камера":
            try:
                index = int(self.camera_index.get().strip())
            except ValueError:
                raise ValueError("Номер веб-камеры должен быть целым числом.") from None
            if not 0 <= index <= 20:
                raise ValueError("Номер веб-камеры должен быть от 0 до 20.")
            return index
        url = self.rtsp_address.get().strip()
        if not url.startswith(("rtsp://", "http://", "https://")):
            raise ValueError("Укажи адрес IP-камеры для получения кадра ROI.")
        return url

    def open_roi_editor(self):
        if self.process is not None and self.process.poll() is None:
            messagebox.showinfo(
                "Сначала останови обработку",
                "Перед настройкой ROI останови обработку, чтобы не занимать камеру дважды.",
            )
            return
        if getattr(self, "roi_editor", None) is not None:
            if self.roi_editor.window.winfo_exists():
                self.roi_editor.window.lift()
                return
        try:
            source = self._preview_source()
            self.roi_editor = ROIPolygonEditor(
                parent=self.root,
                source=source,
                initial_points=self.roi_points,
                on_save=self._apply_roi_points,
            )
        except (ValueError, OSError) as error:
            messagebox.showerror("Редактор ROI", str(error))

    def _apply_roi_points(self, points: list[list[float]]) -> bool:
        previous_points = self.roi_points
        self.roi_points = copy.deepcopy(points)
        if not self.save_settings():
            self.roi_points = previous_points
            return False
        self.roi_summary.set(f"Точек: {len(points)}")
        self._append_log("[ROI] Координаты запретной зоны сохранены.\n")
        return True

    def browse_video(self):
        path = filedialog.askopenfilename(title="Выбери видео",
                                          filetypes=[("Видео", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                                                     ("Все файлы", "*.*")])
        if path:
            self.video_path.set(relative_or_absolute(Path(path)))

    def browse_model(self):
        path = filedialog.askopenfilename(title="Выбери ONNX-модель",
                                          filetypes=[("ONNX", "*.onnx"), ("Все файлы", "*.*")])
        if path:
            self.model_path.set(relative_or_absolute(Path(path)))

    def _settings_to_config(self) -> dict:
        cfg = copy.deepcopy(self.config_data)
        camera = cfg.setdefault("camera", {})
        if self.mode.get() == "Видеофайл":
            value = self.video_path.get().strip()
            if not value:
                raise ValueError("Выбери видеофайл.")
            path = Path(value)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if not path.is_file():
                raise ValueError(f"Видеофайл не найден: {path}")
            camera["source"] = relative_or_absolute(path)
        elif self.mode.get() == "Веб-камера":
            try:
                index = int(self.camera_index.get())
            except ValueError:
                raise ValueError("Номер веб-камеры должен быть целым числом.") from None
            if not 0 <= index <= 20:
                raise ValueError("Номер веб-камеры должен быть от 0 до 20.")
            camera["source"] = index
        else:
            url = self.rtsp_address.get().strip()
            if not url.startswith(("rtsp://", "http://", "https://")):
                raise ValueError("Укажи корректный RTSP или HTTP(S) адрес камеры.")
            camera["source"] = url
        camera["latest_frame"] = True
        playback = cfg.setdefault("playback", {})
        playback["realtime_video"] = bool(self.realtime_video.get())

        models = cfg.setdefault("models", {})
        model_path = self.model_path.get().strip()
        if not model_path:
            raise ValueError("Не указан путь к ONNX-модели.")
        path = Path(model_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise ValueError(f"Не найдена ONNX-модель: {path}")
        models["detector_path"] = relative_or_absolute(path)
        try:
            img_size = int(self.image_size.get().strip())
        except ValueError:
            raise ValueError("Размер изображения должен быть целым числом.") from None
        if img_size < 64 or img_size > 1280 or img_size % 32:
            raise ValueError("Размер изображения должен быть кратен 32 (например, 320 или 640).")
        models["image_size"] = img_size

        global_conf = parse_fraction(self.general_conf.get(), "Общий confidence", 0.001, 1.0)
        values = {}
        for key, label in CLASSES:
            value = parse_fraction(self.class_values[key].get(), f"Confidence: {label}", 0.001, 1.0)
            if value < global_conf:
                raise ValueError(f"Порог «{label}» ({value}) ниже общего порога ({global_conf}).")
            values[key] = value
        models["detector_confidence"] = global_conf
        models["class_confidence"] = values
        modules = cfg.setdefault("modules", {})
        for key, _ in MODULES:
            modules[key] = bool(self.module_values[key].get())
        modules["roi_visible"] = bool(self.roi_visible.get())
        cfg.setdefault("zones", {})["restricted_01"] = copy.deepcopy(self.roi_points)
        tracking = cfg.setdefault("tracking", {})
        tracking["vehicle_nms_enabled"] = bool(self.vehicle_nms_enabled.get())
        tracking["class_nms_enabled"] = bool(self.class_nms_enabled.get())
        tracking["stabilize_vehicle_classes"] = bool(self.stabilize_vehicle.get())
        tracking["vehicle_nms_iou"] = parse_fraction(self.vehicle_iou.get(), "IoU транспорта", 0.05, 0.99)
        for key, _ in NMS_CLASSES:
            tracking.setdefault("class_nms_iou", {})[key] = parse_fraction(
                self.nms_values[key].get(), f"NMS · {key}", 0.0, 0.99)
        cfg.setdefault("visual_smoothing", {})["enabled"] = bool(self.visual_smoothing.get())
        return cfg

    def save_settings(self) -> bool:
        if self.process and self.process.poll() is None:
            messagebox.showinfo("Запущена обработка",
                                "Сначала останови обработку, затем измени настройки и запусти снова.")
            return False
        try:
            new_cfg = self._settings_to_config()
            # Резервная копия YAML перед первым сохранением в этой сессии.
            if not hasattr(self, "_backed_up"):
                backup = CONFIG_PATH.with_name(
                    f"configuration.backup-{datetime.now():%Y%m%d-%H%M%S}.yaml")
                backup.write_bytes(CONFIG_PATH.read_bytes())
                self._backed_up = True
            temporary = CONFIG_PATH.with_suffix(".yaml.tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                yaml.safe_dump(new_cfg, stream, allow_unicode=True, sort_keys=False,
                               default_flow_style=False)
            temporary.replace(CONFIG_PATH)
            self.config_data = new_cfg
            self._append_log("Настройки сохранены: configs/configuration.yaml\n")
            self.status.set("Настройки сохранены")
            return True
        except (ValueError, OSError, yaml.YAMLError) as error:
            messagebox.showerror("Ошибка сохранения", str(error))
            self.status.set("Ошибка в настройках")
            return False

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return
        if not MAIN_PATH.is_file():
            messagebox.showerror("Нет main.py", f"Не найден файл: {MAIN_PATH}")
            return
        if not self.save_settings():
            return
        command = [sys.executable, "-u", str(MAIN_PATH)]
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            self.process = subprocess.Popen(
                command, cwd=PROJECT_ROOT, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=creationflags,
            )
        except OSError as error:
            messagebox.showerror("Ошибка запуска", str(error))
            return
        self.process_done = False
        self.stop_requested = False
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status.set("Обработка запущена. Для остановки нажми кнопку или Q в окне видео.")
        self._append_log("\n=== Запуск Industrial Vision ===\n")
        reader = threading.Thread(target=self._read_process, args=(self.process,), daemon=True)
        reader.start()

    def _read_process(self, process: subprocess.Popen):
        if process.stdout is None:
            return
        try:
            for line in process.stdout:
                self.outbox.put(line)
        except (OSError, ValueError):
            pass
        finally:
            self.outbox.put("[GUI] Поток сообщений завершён.\n")

    def stop(self):
        if not self.process or self.process.poll() is not None:
            return
        self.stop_requested = True
        self.stop_button.configure(state="disabled")
        self.status.set("Останавливаем обработку…")
        try:
            if os.name == "nt":
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.process.send_signal(signal.SIGINT)
        except (OSError, ValueError):
            try:
                self.process.terminate()
            except OSError:
                pass
        self.root.after(2500, self._stop_fallback)

    def _stop_fallback(self):
        if self.process and self.process.poll() is None and self.stop_requested:
            self._append_log("[GUI] Завершаем зависший процесс принудительно.\n")
            try:
                self.process.terminate()
            except OSError:
                pass

    def _poll_output(self):
        try:
            for _ in range(140):
                self._append_log(self.outbox.get_nowait())
        except queue.Empty:
            pass
        if self.process is not None and not self.process_done and self.process.poll() is not None:
            self.process_done = True
            code = self.process.returncode
            self._append_log(f"=== Обработка завершилась (код {code}) ===\n")
            self.status.set("Обработка остановлена" if self.stop_requested or code == 0
                            else f"Программа завершилась с ошибкой: {code}")
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            if self.process.stdout:
                self.process.stdout.close()
        self.root.after(100, self._poll_output)

    def _append_log(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        # Чтобы окно не расходовало память при очень длинной работе.
        if int(self.log.index("end-1c").split(".")[0]) > 2200:
            self.log.delete("1.0", "300.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def open_evidence(self):
        for folder in (PROJECT_ROOT / "evidence", PROJECT_ROOT / "events", PROJECT_ROOT / "output"):
            if folder.is_dir():
                try:
                    if os.name == "nt":
                        os.startfile(str(folder))
                    elif sys.platform == "darwin":
                        subprocess.Popen(["open", str(folder)])
                    else:
                        subprocess.Popen(["xdg-open", str(folder)])
                except OSError as error:
                    messagebox.showerror("Ошибка", str(error))
                return
        messagebox.showinfo("Нет папки", "Папка evidence/events ещё не появилась. Сначала запусти обнаружение событий.")

    def _on_close(self):
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno("Завершение", "Обработка ещё работает. Остановить и закрыть окно?"):
                return
            self.stop()
            self.root.after(2800, self.root.destroy)
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    try:
        IndustrialVisionGUI(root)
    except Exception as error:
        messagebox.showerror("Ошибка запуска GUI", str(error))
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
