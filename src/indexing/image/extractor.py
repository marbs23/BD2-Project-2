"""
Módulo Extractor - SIFT

Extrae descriptores SIFT de cada patch. SIFT detecta keypoints y describe el
entorno de cada uno con un vector de 128 dimensiones; un patch produce un número
variable de descriptores (0 si es una zona plana sin textura).

Recorre los documentos con imagen descargada, aplica el split y guarda en disco:
  - descriptors.npy : todos los descriptores apilados (N, 128), entrada del K-Means
  - chunk_index.npy : (doc_id, patch_index, count) por patch, en el mismo orden,
                      para reconstruir a qué patch pertenece cada bloque de filas

Es el equivalente al cache corpus_stats.pkl del pipeline de texto: trabajo pesado
hecho una vez para que el codebook y el índice no recalculen.
"""
import os

import cv2
import numpy as np

from src.core.db import get_connection
from src.core.paths import IMAGES_DIR, PROCESSED_DIR
from src.indexing.image.split import load_gray, split_patches

DESCRIPTORS_PATH = PROCESSED_DIR / "descriptors.npy"
CHUNK_INDEX_PATH = PROCESSED_DIR / "chunk_index.npy"

# Límite de imágenes a indexar. SIFT sobre las decenas de miles de imágenes del
# scrape no cabe en RAM (todos los descriptores se apilan en un único array); para
# la evaluación (Fase 4) basta un subconjunto. Se toma de la variable de entorno
# IMAGE_LIMIT; 0 o ausente = todas las imágenes disponibles.
IMAGE_LIMIT = int(os.getenv("IMAGE_LIMIT", "0"))


def extract_patch(patch: np.ndarray) -> np.ndarray | None:
    """Descriptores SIFT de un patch: array (N, 128) o None si no hay keypoints."""
    # Instancia local: el detector SIFT de OpenCV no es seguro de compartir entre
    # hilos, y el backend puede atender consultas concurrentes. El costo de crearlo
    # es despreciable frente al de detectAndCompute.
    sift = cv2.SIFT_create()
    _, des = sift.detectAndCompute(patch, None)
    return des


def documents_with_image(conn) -> list[int]:
    """doc_id de documentos cuya imagen está descargada en disco."""
    cur = conn.cursor()
    cur.execute("SELECT doc_id FROM documents WHERE image_url IS NOT NULL AND image_url <> '' ORDER BY doc_id")
    ids = [r[0] for r in cur.fetchall()]
    cur.close()
    return [d for d in ids if (IMAGES_DIR / f"{d}.jpg").exists()]


def build(conn=None, limit: int | None = None) -> dict:
    """
    Extrae descriptores de los documentos con imagen y los persiste.

    Los patches sin keypoints se omiten (no generan chunk). Devuelve un resumen
    con el total de descriptores, patches y documentos procesados.

    `limit` acota cuántas imágenes se indexan (por defecto, la variable de entorno
    IMAGE_LIMIT). Se toman los primeros doc_id con imagen en disco, para que el
    build sea acotado en memoria/tiempo sin cambiar el algoritmo.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    doc_ids = documents_with_image(conn)
    if own_conn:
        conn.close()

    limit = IMAGE_LIMIT if limit is None else limit
    disponibles = len(doc_ids)
    if limit and limit > 0:
        doc_ids = doc_ids[:limit]

    all_descriptors: list[np.ndarray] = []
    chunk_index: list[tuple[int, int, int]] = []

    for doc_id in doc_ids:
        img = load_gray(IMAGES_DIR / f"{doc_id}.jpg")
        if img is None:
            continue
        for patch_index, patch in enumerate(split_patches(img)):
            des = extract_patch(patch)
            if des is None or len(des) == 0:
                continue
            all_descriptors.append(des)
            chunk_index.append((doc_id, patch_index, len(des)))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    descriptors = (np.vstack(all_descriptors).astype(np.float32)
                   if all_descriptors else np.empty((0, 128), np.float32))
    np.save(DESCRIPTORS_PATH, descriptors)
    np.save(CHUNK_INDEX_PATH, np.array(chunk_index, dtype=np.int64))

    return {
        "documentos": len(doc_ids),
        "documentos_disponibles": disponibles,
        "patches": len(chunk_index),
        "descriptores": len(descriptors),
    }


def load_descriptors() -> tuple[np.ndarray, np.ndarray]:
    """Carga (descriptors, chunk_index) cacheados."""
    return np.load(DESCRIPTORS_PATH), np.load(CHUNK_INDEX_PATH)


if __name__ == "__main__":
    stats = build()
    print("Extracción SIFT completada")
    limite = f" (tope IMAGE_LIMIT={IMAGE_LIMIT})" if IMAGE_LIMIT > 0 else ""
    print(f"  documentos:    {stats['documentos']} de {stats['documentos_disponibles']}{limite}")
    print(f"  patches (chunks): {stats['patches']}")
    print(f"  descriptores:  {stats['descriptores']}")
    if stats["patches"]:
        print(f"  promedio descriptores/patch: {stats['descriptores'] / stats['patches']:.1f}")
