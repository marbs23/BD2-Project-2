"""
Módulo Codebook — K-Means (issue #9)

Construye el diccionario visual: agrupa todos los descriptores SIFT en k clusters
con K-Means y se queda con los centroides. Cada centroide es una "palabra visual",
el análogo a una palabra del codebook textual. Un descriptor cualquiera se cuantiza
luego a la palabra visual (centroide) más cercana.

Se usa MiniBatchKMeans porque el K-Means clásico no escala a millones de
descriptores. Los centroides se persisten en codebook_image y se cachean en disco
para que el índice y la búsqueda cuantizen sin reentrenar.
"""
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psycopg2.extras import execute_values

from db_module.connection import get_connection
from image_module.extractor import load_descriptors
from image_module.pgvec import to_pgvector

K = 512  # debe coincidir con vector(512) de image_chunks.histogram (db/init.sql)
BATCH_SIZE = 10000
SEED = 42

CODEBOOK_PATH = Path("data/processed/codebook.npy")


def train(descriptors: np.ndarray, k: int = K) -> np.ndarray:
    """Entrena MiniBatchKMeans y devuelve los k centroides (k, 128)."""
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=min(BATCH_SIZE, len(descriptors)),
        random_state=SEED,
        n_init=3,
        max_no_improvement=10,
    )
    kmeans.fit(descriptors)
    return kmeans.cluster_centers_.astype(np.float32)


def assign(descriptors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Cuantiza cada descriptor a la palabra visual más cercana. Devuelve labels (N,)."""
    return pairwise_distances_argmin(descriptors, centroids)


def persist(centroids: np.ndarray, conn) -> None:
    """Guarda los centroides en codebook_image (word_id 1..k) y en cache de disco."""
    cur = conn.cursor()
    cur.execute("TRUNCATE codebook_image RESTART IDENTITY CASCADE")
    execute_values(
        cur,
        "INSERT INTO codebook_image (centroid) VALUES %s",
        [(to_pgvector(c),) for c in centroids],
    )
    conn.commit()
    cur.close()
    CODEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(CODEBOOK_PATH, centroids)


def load_codebook() -> np.ndarray:
    """Carga los centroides cacheados (k, 128)."""
    return np.load(CODEBOOK_PATH)


def build(k: int = K, conn=None) -> dict:
    """Entrena el codebook sobre los descriptores cacheados y lo persiste."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    descriptors, _ = load_descriptors()
    if len(descriptors) < k:
        raise ValueError(
            f"Solo hay {len(descriptors)} descriptores para k={k}; "
            "extrae más imágenes o baja k."
        )

    centroids = train(descriptors, k)
    persist(centroids, conn)

    if own_conn:
        conn.close()

    return {"k": k, "descriptores": len(descriptors)}


if __name__ == "__main__":
    stats = build()
    print("Codebook visual construido y persistido en codebook_image")
    print(f"  palabras visuales (k): {stats['k']}")
    print(f"  descriptores agrupados: {stats['descriptores']}")
