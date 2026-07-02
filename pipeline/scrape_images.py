"""
Descarga la imagen principal (hero) de cada documento (maintainer).

Cada documento de BBC tiene una sola imagen representativa (og:image), que el
scraper de artículos ya guardó en documents.image_url. Aquí se descargan esos
archivos a disco para que el pipeline de visión (split + SIFT) los procese.
Relación 1:1 documento-imagen: el nombre del archivo es el doc_id.

Corre en paralelo (ThreadPoolExecutor) con un pequeño sleep por worker y backoff
simple ante 429/503, para no saturar el origen de las imágenes ni terminar
bloqueado a mitad de la descarga.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import requests

from src.core.db import get_connection
from src.core.paths import IMAGES_DIR

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
TIMEOUT = 10

MAX_WORKERS = 8            # las imágenes suelen estar repartidas en varios hosts/CDNs de BBC
SLEEP_POR_REQUEST = 0.1    # pequeño respiro por hilo (no bloquea a los demás workers)
BACKOFF_SEGUNDOS = 5       # espera si el servidor responde 429/503


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


def _get(url):
    """GET con un reintento simple si el servidor responde 429/503 (rate limiting)."""
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    if r.status_code in (429, 503):
        time.sleep(BACKOFF_SEGUNDOS)
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    return r


def download(doc_id: int, url: str) -> bool:
    """Descarga una imagen. Devuelve True si quedó guardada."""
    time.sleep(SLEEP_POR_REQUEST)  # respiro por request, no serializa a los demás workers
    try:
        r = _get(url)
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

    if not pendientes:
        print("Listo: nada por descargar")
        return

    ok = 0
    fail = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download, doc_id, url): doc_id for doc_id, url in pendientes}

        for i, fut in enumerate(as_completed(futures), 1):
            doc_id = futures[fut]
            try:
                exito = fut.result()
            except Exception:
                exito = False

            if exito:
                ok += 1
            else:
                fail += 1

            if i % 50 == 0 or i == len(pendientes):
                print(f"  [{i}/{len(pendientes)}] descargadas={ok} fallidas={fail}")

    print(f"\nListo: {ok}/{len(pendientes)} imágenes en {IMAGES_DIR} (fallidas: {fail})")


if __name__ == "__main__":
    run()