"""
Módulo Índice Invertido - histogramas

Convierte cada patch en un histograma de palabras visuales y construye el índice
invertido con el mismo modelo y la misma ponderación TF-IDF que el texto. No usa
SPIMI por bloques: el vocabulario visual es fijo (k del codebook) y el histograma
es denso, así que la mezcla externa no aporta (el extractor ya dejó todos los
descriptores en memoria).

  1. cuantización : cada descriptor SIFT del patch -> su palabra visual (centroide).
  2. histograma   : conteo de palabras visuales del patch (term frequency).
  3. TF-IDF       : se pondera igual que en texto -> (1 + log10(tf)) * log10(N/df).
  4. persistencia : image_chunks (histograma denso + norma) e inverted_index_image
                    (posting lists palabra_visual -> patch).

El histograma denso se guarda en image_chunks.histogram para la comparativa con
pgvector; el índice invertido es el método propio.
"""
import numpy as np

from psycopg2.extras import execute_values

from src.core.db import get_connection
from src.indexing.image.extractor import load_descriptors
from src.indexing.image.codebook import load_codebook, assign
from src.indexing.image.pgvec import to_pgvector

INSERT_BATCH = 5000


def _histogramas_por_patch(labels: np.ndarray, counts: np.ndarray, k: int) -> list[np.ndarray]:
    """Histograma de conteo de palabras visuales por patch, a partir de los labels apilados."""
    offsets = np.concatenate([[0], np.cumsum(counts)])
    histogramas = []
    for i in range(len(counts)):
        words = labels[offsets[i]:offsets[i + 1]]
        histogramas.append(np.bincount(words, minlength=k).astype(np.float64))
    return histogramas


def build_index(conn=None) -> dict:
    """
    Pipeline completo: cuantización -> histograma -> TF-IDF -> persistencia.

    Por cada palabra visual se calcula, con las fórmulas vistas en clase:
        df  = nº de patches que la contienen
        idf = log10(N / df)
        peso(patch) = (1 + log10(tf)) * idf

    El histograma TF-IDF denso y su norma ||d|| = sqrt(sum(tf_idf^2)) se guardan en
    image_chunks (la norma, para que la búsqueda por coseno no la recalcule en cada
    query) y cada posting palabra_visual -> patch va a inverted_index_image.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    descriptors, chunk_index = load_descriptors()
    centroids = load_codebook()
    if len(descriptors) == 0:
        raise ValueError("No hay descriptores; corre primero el extractor SIFT.")

    # k sale del codebook cargado, no de una constante: evita desalinear el
    # histograma con la columna vector(k) y los word_id si se reentrena con otro k.
    k = centroids.shape[0]

    # Cuantización de todos los descriptores y reconstrucción por patch.
    labels = assign(descriptors, centroids)
    counts = chunk_index[:, 2]
    histogramas = _histogramas_por_patch(labels, counts, k)
    n_patches = len(histogramas)

    # df por palabra visual = nº de patches que la contienen; de ahí el idf.
    df = np.zeros(k, dtype=np.int64)
    for h in histogramas:
        df[h > 0] += 1
    idf = np.where(df > 0, np.log10(n_patches / np.maximum(df, 1)), 0.0)

    cur = conn.cursor()
    cur.execute("TRUNCATE image_chunks RESTART IDENTITY CASCADE")
    cur.execute("TRUNCATE inverted_index_image")

    def _tfidf(h: np.ndarray) -> np.ndarray:
        """Histograma TF-IDF denso del patch: (1 + log10(tf)) * idf en las palabras presentes."""
        present = h > 0
        tfidf = np.zeros(k, dtype=np.float64)
        tfidf[present] = (1 + np.log10(h[present])) * idf[present]
        return tfidf

    # Se procesa por lotes: por cada lote se insertan sus image_chunks (histograma
    # denso + norma), se recuperan sus chunk_id e inmediatamente se insertan sus
    # postings. Así no se mantienen en memoria los histogramas TF-IDF densos de toda
    # la colección a la vez; el chunk_id (SERIAL) queda alineado con el orden de
    # inserción dentro del lote, así que las postings referencian el patch correcto.
    n_postings = 0
    for inicio in range(0, n_patches, INSERT_BATCH):
        tfidf_lote = [_tfidf(h) for h in histogramas[inicio:inicio + INSERT_BATCH]]

        valores_chunks = [
            (int(chunk_index[inicio + j, 0]), int(chunk_index[inicio + j, 1]),
             float(np.sqrt(np.sum(t * t))), to_pgvector(t))
            for j, t in enumerate(tfidf_lote)
        ]
        chunk_ids = execute_values(
            cur,
            "INSERT INTO image_chunks (doc_id, patch_index, norm, histogram) VALUES %s "
            "RETURNING chunk_id",
            valores_chunks,
            fetch=True,
        )

        # Una posting por palabra visual presente. word_id en BD es 1..k; el label
        # del cluster es 0..k-1.
        postings = [
            (int(word_idx) + 1, chunk_id, float(t[word_idx]))
            for (chunk_id,), t in zip(chunk_ids, tfidf_lote)
            for word_idx in np.nonzero(t)[0]
        ]
        if postings:
            execute_values(
                cur,
                "INSERT INTO inverted_index_image (word_id, chunk_id, tf_idf) VALUES %s",
                postings,
            )
            n_postings += len(postings)

    conn.commit()
    cur.close()
    if own_conn:
        conn.close()

    return {"patches": n_patches, "postings": n_postings, "palabras_visuales": k}


if __name__ == "__main__":
    stats = build_index()
    print("Índice invertido de imagen construido")
    print(f"  patches indexados: {stats['patches']}")
    print(f"  postings:          {stats['postings']}")
    print(f"  palabras visuales: {stats['palabras_visuales']}")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM image_chunks")
    print(f"  image_chunks filas:        {cur.fetchone()[0]}")
    cur.execute("SELECT count(*) FROM inverted_index_image")
    print(f"  inverted_index_image filas: {cur.fetchone()[0]}")
    cur.close()
    conn.close()
