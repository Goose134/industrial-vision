
import sys
import cv2
import numpy
import yaml
import fastapi
import uvicorn
import psutil
import pytest

print("=== Проверка окружения ===")

print("Python:", sys.version.split()[0])
print("OpenCV:", cv2.__version__)
print("NumPy:", numpy.__version__)
print("PyYAML:", yaml.__version__)
print("FastAPI:", fastapi.__version__)
print("Uvicorn:", uvicorn.__version__)
print("psutil:", psutil.__version__)
print("pytest:", pytest.__version__)

print("\nВсе базовые библиотеки успешно импортированы!")
