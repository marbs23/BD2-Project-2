"""
Módulo Búsqueda por texto — ranking coseno (issue #16)

Dada una consulta de texto, recupera los top-N resultados usando el índice
invertido (inverted_index_text) y el modelo de espacio vectorial visto en clase:

    Pipeline (Ranked Retrieval):
      1. Representación : query -> vector TF-IDF.
      2. Filtrado      : se cruzan las posting lists del índice invertido para
                         obtener solo los chunks candidatos (los que comparten
                         algún término con la query).
      3. Score         : similitud de coseno query-chunk.
      4. Ranking       : se devuelven los top-N por score descendente.

Pesos (mismas fórmulas que el extractor / la clase):
    idf(t)        = log10(N / df(t))      df = largo de la posting list de t
    peso_doc      = tf_idf almacenado en inverted_index_text
    peso_query(t) = (1 + log10(qtf)) * idf(t)
"""

import os
import sys
import math
from collections import defaultdict, Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_module.connection import get_connection
from text_module.extractor import preprocess


def _count_chunks(conn) -> int:
    """N = número total de chunks de la colección (para el IDF)."""
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM text_chunks")
    n = cur.fetchone()[0]
    cur.close()
    return n


def _query_term_ids(conn, tokens: list[str]) -> dict[int, int]:
    """
    Mapea los términos de la query (ya preprocesados) a su word_id del codebook,
    quedándose solo con los que están en el diccionario. Los términos fuera del
    codebook se ignoran (igual que la cuantización descarta lo no representado).

    Devuelve {word_id: qtf} donde qtf es la frecuencia del término en la query.
    """
    qtf = Counter(tokens)
    if not qtf:
        return {}
    cur = conn.cursor()
    cur.execute(
        "SELECT word, word_id FROM codebook_text WHERE word = ANY(%s)",
        (list(qtf),),
    )
    word_to_id = dict(cur.fetchall())
    cur.close()
    return {word_to_id[w]: c for w, c in qtf.items() if w in word_to_id}


def _fetch_postings(conn, word_ids: list[int]) -> dict[int, list[tuple[int, float]]]:
    """Trae las posting lists de los word_ids dados: {word_id: [(chunk_id, tf_idf), ...]}."""
    if not word_ids:
        return {}
    cur = conn.cursor()
    cur.execute(
        "SELECT word_id, chunk_id, tf_idf FROM inverted_index_text WHERE word_id = ANY(%s)",
        (word_ids,),
    )
    postings: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for word_id, chunk_id, tf_idf in cur.fetchall():
        postings[word_id].append((chunk_id, tf_idf))
    cur.close()
    return postings


def _chunk_norms(conn, chunk_ids: list[int]) -> dict[int, float]:
    """
    Norma euclídea ||d|| de cada chunk candidato, leída desde text_chunks.norm
    (precalculada por SPIMI durante la indexación). Es un lookup por PK, mucho más
    rápido que reagregar sum(tf_idf^2) sobre el índice en cada query.
    """
    if not chunk_ids:
        return {}
    cur = conn.cursor()
    cur.execute(
        "SELECT chunk_id, norm FROM text_chunks WHERE chunk_id = ANY(%s)",
        (chunk_ids,),
    )
    norms = {chunk_id: norm for chunk_id, norm in cur.fetchall() if norm is not None}
    cur.close()
    return norms


def _fetch_chunk_meta(conn, chunk_ids: list[int]) -> dict[int, dict]:
    """Trae metadata de cada chunk candidato: doc_id, contenido, título y url."""
    if not chunk_ids:
        return {}
    cur = conn.cursor()
    cur.execute(
        """
        SELECT tc.chunk_id, tc.doc_id, tc.content, d.title, d.url
        FROM text_chunks tc
        JOIN documents d ON d.doc_id = tc.doc_id
        WHERE tc.chunk_id = ANY(%s)
        """,
        (chunk_ids,),
    )
    meta = {
        chunk_id: {"doc_id": doc_id, "content": content, "title": title, "url": url}
        for chunk_id, doc_id, content, title, url in cur.fetchall()
    }
    cur.close()
    return meta


def _snippet(content: str, length: int = 160) -> str:
    """Recorte del contenido del chunk para mostrar en los resultados."""
    content = " ".join(content.split())
    return content[:length] + ("…" if len(content) > length else "")


def buscar(query: str, top_n: int = 10, group_by_doc: bool = True, conn=None) -> list[dict]:
    """
    Recupera los top-N resultados para la query rankeados por similitud de coseno:

        cos(q, d) = (q · d) / (||q|| * ||d||)

    Cada resultado incluye chunk_id, doc_id, title, url, score y snippet. Si
    group_by_doc es True (por defecto) se conserva el mejor chunk por documento,
    de modo que el top-N son artículos distintos.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    tokens = preprocess(query)
    qtf_by_id = _query_term_ids(conn, tokens)
    postings = _fetch_postings(conn, list(qtf_by_id))
    n_chunks = _count_chunks(conn)

    # Acumulación término a término del producto punto + norma de la query.
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
    chunk_norm = _chunk_norms(conn, list(dot))

    # Normalización coseno.
    scores: dict[int, float] = {}
    for chunk_id, dot_value in dot.items():
        d_norm = chunk_norm.get(chunk_id, 0.0)
        if d_norm > 0:
            scores[chunk_id] = dot_value / (query_norm * d_norm)

    # Enriquecer con metadata del documento.
    meta = _fetch_chunk_meta(conn, list(scores))
    if own_conn:
        conn.close()

    results = []
    for chunk_id, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        m = meta[chunk_id]
        results.append({
            "chunk_id": chunk_id,
            "doc_id": m["doc_id"],
            "title": m["title"],
            "url": m["url"],
            "score": score,
            "snippet": _snippet(m["content"]),
        })

    # Deduplicar por documento: el mejor chunk representa al artículo.
    if group_by_doc:
        seen: set[int] = set()
        deduped = []
        for r in results:  # ya vienen ordenados por score descendente
            if r["doc_id"] not in seen:
                seen.add(r["doc_id"])
                deduped.append(r)
        results = deduped

    return results[:top_n]


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "russia ukraine war"
    print(f"Query: {query!r}\n")
    resultados = buscar(query, top_n=10)
    if not resultados:
        print("  (sin resultados — ningún término de la query está en el codebook)")
    for i, r in enumerate(resultados, start=1):
        print(f"  {i:>2}. [{r['score']:.4f}] {r['title']}")
        print(f"      doc {r['doc_id']} · chunk {r['chunk_id']}")
        print(f"      {r['snippet']}\n")
