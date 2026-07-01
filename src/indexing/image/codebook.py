"""
Módulo Codebook - K-Means

Construye el diccionario visual: agrupa todos los descriptores SIFT en k clusters
con K-Means y se queda con los centroides. Cada centroide es una "palabra visual",
el análogo a una palabra del codebook textual. Un descriptor cualquiera se cuantiza
luego a la palabra visual (centroide) más cercana.

Se usa MiniBatchKMeans porque el K-Means clásico no escala a millones de
descriptores. Los centroides se persisten en codebook_image y se cachean en disco
para que el índice y la búsqueda cuantizen sin reentrenar.
"""
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin

from psycopg2.extras import execute_values

from src.core.db import get_connection
from src.core.paths import PROCESSED_DIR
from src.indexing.image.extractor import load_descriptors
from src.indexing.image.pgvec import to_pgvector

K = 512  # debe coincidir con vector(512) de image_chunks.histogram (db/schema.sql)
BATCH_SIZE = 10000
SEED = 42

CODEBOOK_PATH = PROCESSED_DIR / "codebook.npy"


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


def load_codebook(conn=None) -> np.ndarray:
    """
    Carga los centroides (k, 128) desde codebook_image (word_id 1..k).

    Se lee de la BD (no del .npy) para que el backend sea autosuficiente a partir
    del dump: el cache en disco no viaja al contenedor. Si la tabla está vacía cae
    al cache local, útil solo en la máquina del maintainer durante el build.
    """
    own = conn is None
    if own:
        conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT centroid FROM codebook_image ORDER BY word_id")
    rows = [r[0] for r in cur.fetchall()]
    cur.close()
    if own:
        conn.close()
    if not rows:
        return np.load(CODEBOOK_PATH)
    return np.array(
        [np.fromstring(str(r).strip("[]"), sep=",") for r in rows],
        dtype=np.float32,
    )


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
