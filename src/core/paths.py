"""Rutas del proyecto ancladas a la raíz del repo.

Centraliza dónde viven los datos para que los módulos no dependan del directorio
desde el que se ejecutan (CWD). Todo cuelga de la raíz del repositorio, calculada
a partir de la ubicación de este archivo: src/core/paths.py -> parents[2] = raíz.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGES_DIR = RAW_DIR / "images"
PROCESSED_DIR = DATA_DIR / "processed"
