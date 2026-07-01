"""
Descarga la imagen principal (hero) de cada documento (maintainer).

Cada documento de BBC tiene una sola imagen representativa (og:image), que el
scraper de artículos ya guardó en documents.image_url. Aquí se descargan esos
archivos a disco para que el pipeline de visión (split + SIFT) los procese.
Relación 1:1 documento-imagen: el nombre del archivo es el doc_id.
"""
import os

import cv2
import numpy as np
import requests

from src.core.db import get_connection
from src.core.paths import IMAGES_DIR

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
TIMEOUT = 10


def _path_for(doc_id: int):
    return IMAGES_DIR / f"{doc_id}.jpg"


def pending_documents(conn) -> list[tuple[int, str]]:
    """Documentos con image_url que aún no tienen su archivo descargado."""
    cur = conn.cursor()
    cur.execute(
        "SELECT doc_id, image_url FROM documents "
        "WHERE image_url IS NOT NULL AND image_url <> '' ORDER BY doc_id"
    )
    rows = cur.fetchall()
    cur.close()
    return [(doc_id, url) for doc_id, url in rows if not _path_for(doc_id).exists()]


def download(doc_id: int, url: str) -> bool:
    """Descarga una imagen. Devuelve True si quedó guardada."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    except requests.RequestException:
        return False
    if r.status_code != 200 or "image" not in r.headers.get("Content-Type", ""):
        return False
    # El Content-Type no garantiza una imagen real; confirmamos que OpenCV pueda
    # decodificarla antes de tocar el disco.
    if cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_GRAYSCALE) is None:
        return False
    # Escritura atómica: si el proceso muere, no queda un .jpg truncado que luego
    # se confunda con una descarga completa y nunca se reintente.
    dest = _path_for(doc_id)
    tmp = dest.with_suffix(".jpg.tmp")
    tmp.write_bytes(r.content)
    os.replace(tmp, dest)
    return True


def run() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_connection()
    pendientes = pending_documents(conn)
    conn.close()

    print(f"Imágenes por descargar: {len(pendientes)}")
    ok = 0
    for i, (doc_id, url) in enumerate(pendientes, 1):
        if download(doc_id, url):
            ok += 1
        if i % 50 == 0:
            print(f"  [{i}/{len(pendientes)}] descargadas={ok}")

    print(f"Listo: {ok}/{len(pendientes)} imágenes en {IMAGES_DIR}")


if __name__ == "__main__":
    run()
