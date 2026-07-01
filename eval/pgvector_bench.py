"""
Comparativa pgvector - HNSW / IVFFlat

Baseline nativo contra el índice invertido propio. Los mismos histogramas TF-IDF
guardados en image_chunks.histogram se buscan con los índices vectoriales de
PostgreSQL en lugar de las posting lists.

  - HNSW    : grafo navegable; consultas rápidas, índice más caro de construir.
  - IVFFlat : listas invertidas por centroides; índice barato, algo menos preciso.

Ambos con distancia de coseno (<=>), la misma medida del método propio, para que
la comparación de latencia y resultados sea justa.
"""
import sys
import math

import numpy as np

from src.core.db import get_connection
from src.indexing.image.codebook import load_codebook, K
from src.indexing.image.search import query_visual_words
from src.indexing.image.split import GRID
from src.indexing.image.pgvec import to_pgvector

METODOS = ("hnsw", "ivfflat")


def _ivf_lists(n_rows: int) -> int:
    """Listas de IVFFlat según la recomendación de pgvector (~filas/1000, sqrt sobre 1M)."""
    if n_rows <= 1_000_000:
        return max(1, n_rows // 1000)
    return int(math.sqrt(n_rows))


def drop_indexes(conn) -> None:
    """Elimina ambos índices vectoriales de image_chunks.histogram."""
    cur = conn.cursor()
    cur.execute("DROP INDEX IF EXISTS idx_img_hnsw")
    cur.execute("DROP INDEX IF EXISTS idx_img_ivf")
    conn.commit()
    cur.close()


def create_index(method: str, conn=None) -> None:
    """
    Crea el índice vectorial elegido dejándolo como el único sobre histogram.

    Se borran ambos índices antes de crear: si HNSW e IVFFlat coexisten, el planner
    elegiría uno por costo y el benchmark mediría el índice equivocado.
    """
    if method not in METODOS:
        raise ValueError(f"método desconocido: {method}")
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    cur = conn.cursor()
    drop_indexes(conn)
    if method == "hnsw":
        cur.execute(
            "CREATE INDEX idx_img_hnsw ON image_chunks "
            "USING hnsw (histogram vector_cosine_ops) WITH (m=16, ef_construction=64)"
        )
    else:
        cur.execute("SELECT count(*) FROM image_chunks")
        lists = _ivf_lists(cur.fetchone()[0])
        cur.execute(
            "CREATE INDEX idx_img_ivf ON image_chunks "
            f"USING ivfflat (histogram vector_cosine_ops) WITH (lists={lists})"
        )
    conn.commit()
    cur.close()
    if own_conn:
        conn.close()


def _df_por_word(conn, word_ids: list[int], max_chunks: int | None = None) -> dict[int, int]:
    """df = nº de patches que contienen cada palabra visual (largo de su posting list)."""
    if not word_ids:
        return {}
    sql = ("SELECT word_id, count(*) FROM inverted_index_image "
           "WHERE word_id = ANY(%s)")
    params: list = [word_ids]
    if max_chunks is not None:
        sql += " AND chunk_id <= %s"
        params.append(max_chunks)
    sql += " GROUP BY word_id"
    cur = conn.cursor()
    cur.execute(sql, params)
    df = dict(cur.fetchall())
    cur.close()
    return df


def query_vector(conn, image_path: str, centroids: np.ndarray,
                 k: int = K, max_chunks: int | None = None) -> np.ndarray | None:
    """Histograma TF-IDF denso de la imagen de consulta, en el mismo espacio que el índice."""
    qtf_by_id = query_visual_words(image_path, centroids)
    if not qtf_by_id:
        return None
    if max_chunks is not None:
        n_chunks = max_chunks
    else:
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM image_chunks")
        n_chunks = cur.fetchone()[0]
        cur.close()
    df = _df_por_word(conn, list(qtf_by_id), max_chunks)

    vec = np.zeros(k, dtype=np.float64)
    for word_id, qtf in qtf_by_id.items():
        d = df.get(word_id, 0)
        if d:
            vec[word_id - 1] = (1 + math.log10(qtf)) * math.log10(n_chunks / d)
    return vec


def buscar(image_path: str, top_n: int = 10, group_by_doc: bool = True,
           conn=None, max_chunks: int | None = None) -> list[dict]:
    """Top-N documentos por distancia de coseno usando el índice pgvector."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    centroids = load_codebook()
    vec = query_vector(conn, image_path, centroids, max_chunks=max_chunks)
    if vec is None or not vec.any():
        if own_conn:
            conn.close()
        return []

    cur = conn.cursor()
    # Margen sobre top_n: el dedup por documento puede colapsar hasta GRID*GRID
    # patches de un mismo artículo. Se excluye el histograma todo-cero (norm=0),
    # cuya distancia de coseno es indefinida, igual que hace el método propio.
    limite = top_n * (GRID * GRID) if group_by_doc else top_n
    where = "WHERE ic.norm > 0"
    params: list = [to_pgvector(vec)]
    if max_chunks is not None:
        where += " AND ic.chunk_id <= %s"
        params.append(max_chunks)
    params += [to_pgvector(vec), limite]
    cur.execute(
        f"""
        SELECT ic.doc_id, d.title, d.url, d.category, d.image_url,
               ic.histogram <=> %s AS dist
        FROM image_chunks ic
        JOIN documents d ON d.doc_id = ic.doc_id
        {where}
        ORDER BY ic.histogram <=> %s
        LIMIT %s
        """,
        params,
    )
    filas = cur.fetchall()
    cur.close()
    if own_conn:
        conn.close()

    results = []
    seen: set[int] = set()
    for doc_id, title, url, category, image_url, dist in filas:
        if group_by_doc and doc_id in seen:
            continue
        seen.add(doc_id)
        results.append({
            "doc_id": doc_id, "title": title, "url": url, "category": category,
            "image_url": image_url, "score": 1 - float(dist),  # coseno = 1 - distancia
        })
        if len(results) >= top_n:
            break
    return results


if __name__ == "__main__":
    from src.core.paths import IMAGES_DIR

    method = sys.argv[1] if len(sys.argv) > 1 else "hnsw"
    print(f"Creando índice {method}...")
    create_index(method)
    print("  ok")

    sample = next((str(p) for p in IMAGES_DIR.glob("*.jpg")), None)
    if sample:
        print(f"\nConsulta ({method}): {sample}\n")
        for i, r in enumerate(buscar(sample, top_n=10), 1):
            print(f"  {i:>2}. [{r['score']:.4f}] {r['title']}  ({r['category']})")
