"""
Módulo Búsqueda por imagen - similitud de histograma

Dada una imagen de consulta, recupera los documentos con patches visualmente más
parecidos. Es el mismo modelo de espacio vectorial de la búsqueda de texto, con
palabras visuales en lugar de términos:

  1. la imagen -> descriptores SIFT -> palabras visuales -> histograma TF-IDF.
  2. se cruzan las posting lists del índice invertido (inverted_index_image).
  3. similitud de coseno entre el histograma de la consulta y el de cada patch.
  4. se deduplica por documento: el mejor patch representa al artículo.
"""
import sys
import math
from collections import defaultdict, Counter

import numpy as np

from src.core.db import get_connection
from src.indexing.image.split import load_gray, split_patches
from src.indexing.image.extractor import extract_patch
from src.indexing.image.codebook import load_codebook, assign


def query_visual_words(image_path: str, centroids: np.ndarray) -> dict[int, int]:
    """
    Convierte la imagen de consulta en {word_id: qtf}, donde word_id es 1..k.
    Junta los descriptores de todos sus patches y los cuantiza contra el codebook.
    """
    img = load_gray(image_path)
    if img is None:
        return {}
    descriptores = [d for d in (extract_patch(p) for p in split_patches(img)) if d is not None]
    if not descriptores:
        return {}
    labels = assign(np.vstack(descriptores), centroids)
    # word_id en BD es label + 1.
    return {int(word) + 1: int(c) for word, c in Counter(labels).items()}


def _count_chunks(conn, max_chunks: int | None = None) -> int:
    """N = número de chunks de la colección (para el IDF). Acotado por max_chunks si se da."""
    if max_chunks is not None:
        return max_chunks
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM image_chunks")
    n = cur.fetchone()[0]
    cur.close()
    return n


def _fetch_postings(conn, word_ids: list[int],
                    max_chunks: int | None = None) -> dict[int, list[tuple[int, float]]]:
    """Posting lists de los word_ids: {word_id: [(chunk_id, tf_idf), ...]}, hasta max_chunks."""
    if not word_ids:
        return {}
    sql = "SELECT word_id, chunk_id, tf_idf FROM inverted_index_image WHERE word_id = ANY(%s)"
    params: list = [word_ids]
    if max_chunks is not None:
        sql += " AND chunk_id <= %s"
        params.append(max_chunks)
    cur = conn.cursor()
    cur.execute(sql, params)
    postings: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for word_id, chunk_id, tf_idf in cur.fetchall():
        postings[word_id].append((chunk_id, tf_idf))
    cur.close()
    return postings


def _chunk_norms(conn, chunk_ids: list[int]) -> dict[int, float]:
    """Norma ||d|| de cada patch candidato, leída de image_chunks.norm (precalculada al indexar)."""
    cur = conn.cursor()
    cur.execute(
        "SELECT chunk_id, norm FROM image_chunks WHERE chunk_id = ANY(%s)",
        (chunk_ids,),
    )
    norms = {cid: norm for cid, norm in cur.fetchall() if norm is not None}
    cur.close()
    return norms


def _fetch_doc_meta(conn, chunk_ids: list[int]) -> dict[int, dict]:
    """Metadata del documento de cada patch candidato: doc_id, título, url, categoría e imagen."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ic.chunk_id, ic.doc_id, d.title, d.url, d.category, d.image_url
        FROM image_chunks ic
        JOIN documents d ON d.doc_id = ic.doc_id
        WHERE ic.chunk_id = ANY(%s)
        """,
        (chunk_ids,),
    )
    meta = {
        cid: {"doc_id": doc_id, "title": title, "url": url,
              "category": category, "image_url": image_url}
        for cid, doc_id, title, url, category, image_url in cur.fetchall()
    }
    cur.close()
    return meta


def buscar(image_path: str, top_n: int = 10, group_by_doc: bool = True,
           conn=None, max_chunks: int | None = None) -> list[dict]:
    """
    Top-N documentos con patches más similares a la imagen de consulta.

    max_chunks restringe la búsqueda a los primeros N chunks (carga del benchmark).
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    centroids = load_codebook()
    qtf_by_id = query_visual_words(image_path, centroids)
    postings = _fetch_postings(conn, list(qtf_by_id), max_chunks)
    n_chunks = _count_chunks(conn, max_chunks)

    dot: dict[int, float] = defaultdict(float)
    query_norm_sq = 0.0
    for word_id, qtf in qtf_by_id.items():
        plist = postings.get(word_id, [])
        df = len(plist)
        if df == 0:
            continue
        idf = math.log10(n_chunks / df)
        q_weight = (1 + math.log10(qtf)) * idf
        query_norm_sq += q_weight * q_weight
        for chunk_id, d_weight in plist:
            dot[chunk_id] += q_weight * d_weight

    if not dot or query_norm_sq == 0:
        if own_conn:
            conn.close()
        return []

    query_norm = math.sqrt(query_norm_sq)
    norms = _chunk_norms(conn, list(dot))
    scores: dict[int, float] = {}
    for cid, dot_v in dot.items():
        d_norm = norms.get(cid, 0.0)
        if d_norm > 0:
            scores[cid] = dot_v / (query_norm * d_norm)

    meta = _fetch_doc_meta(conn, list(scores))
    if own_conn:
        conn.close()

    results = []
    for chunk_id, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        m = meta[chunk_id]
        results.append({
            "chunk_id": chunk_id, "doc_id": m["doc_id"], "title": m["title"],
            "url": m["url"], "category": m["category"], "image_url": m["image_url"],
            "score": score,
        })

    if group_by_doc:
        seen: set[int] = set()
        deduped = []
        for r in results:
            if r["doc_id"] not in seen:
                seen.add(r["doc_id"])
                deduped.append(r)
        results = deduped

    return results[:top_n]


if __name__ == "__main__":
    from src.core.paths import IMAGES_DIR

    sample = sys.argv[1] if len(sys.argv) > 1 else next(
        (str(p) for p in IMAGES_DIR.glob("*.jpg")), None
    )
    if not sample:
        print("Pasa la ruta de una imagen de consulta")
        sys.exit(1)

    print(f"Consulta: {sample}\n")
    for i, r in enumerate(buscar(sample, top_n=10), 1):
        print(f"  {i:>2}. [{r['score']:.4f}] {r['title']}  ({r['category']})")
