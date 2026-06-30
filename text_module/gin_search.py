"""
Búsqueda full-text nativa de PostgreSQL (issue #23, Fase 3)

Usa el índice GIN sobre `text_chunks.tsv` para resolver consultas con el motor
full-text de Postgres (`tsv @@ query`, ranking con `ts_rank`). Es la contraparte
nativa del índice invertido propio (search.py): devuelve resultados con la misma
estructura para poder compararlos de forma justa.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_module.connection import get_connection
from text_module.search import _fetch_chunk_meta, _snippet


def buscar_fulltext(query: str, top_n: int = 10, conn=None) -> list[dict]:
    """
    Búsqueda full-text con índice GIN. Recupera los chunks que matchean la query,
    los rankea con ts_rank, deduplica por documento (mejor chunk por artículo) y
    devuelve el top-N con la misma forma que search.buscar (score = ts_rank).
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    cur = conn.cursor()
    # plainto_tsquery: convierte la query en términos AND, normalizados al idioma.
    cur.execute(
        """
        SELECT tc.chunk_id, ts_rank(tc.tsv, q) AS rank
        FROM text_chunks tc, plainto_tsquery('english', %s) AS q
        WHERE tc.tsv @@ q
        ORDER BY rank DESC
        LIMIT %s
        """,
        (query, top_n * 10),  # pool de candidatos amplio para deduplicar a top_n docs
    )
    ranked = cur.fetchall()
    cur.close()

    if not ranked:
        if own_conn:
            conn.close()
        return []

    meta = _fetch_chunk_meta(conn, [chunk_id for chunk_id, _ in ranked])
    if own_conn:
        conn.close()

    # Deduplicar por documento conservando el mejor chunk (ya vienen por rank desc).
    seen: set[int] = set()
    results = []
    for chunk_id, rank in ranked:
        m = meta[chunk_id]
        if m["doc_id"] in seen:
            continue
        seen.add(m["doc_id"])
        results.append({
            "chunk_id": chunk_id,
            "doc_id": m["doc_id"],
            "title": m["title"],
            "url": m["url"],
            "score": float(rank),
            "snippet": _snippet(m["content"]),
        })
        if len(results) >= top_n:
            break

    return results


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "russia ukraine war"
    print(f"Query (full-text GIN): {query!r}\n")
    resultados = buscar_fulltext(query, top_n=10)
    if not resultados:
        print("  (sin resultados)")
    for i, r in enumerate(resultados, start=1):
        print(f"  {i:>2}. [{r['score']:.4f}] {r['title']}")
        print(f"      doc {r['doc_id']} · chunk {r['chunk_id']}")
        print(f"      {r['snippet']}\n")
