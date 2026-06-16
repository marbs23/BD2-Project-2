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
    Norma euclídea ||d|| de cada chunk candidato = sqrt(sum(tf_idf^2)) sobre TODOS
    los términos del chunk (no solo los de la query). Se calcula al vuelo para los
    candidatos con un agregado en SQL.
    """
    if not chunk_ids:
        return {}
    cur = conn.cursor()
    cur.execute(
        """
        SELECT chunk_id, sum(tf_idf * tf_idf)
        FROM inverted_index_text
        WHERE chunk_id = ANY(%s)
        GROUP BY chunk_id
        """,
        (chunk_ids,),
    )
    norms = {chunk_id: math.sqrt(sq) for chunk_id, sq in cur.fetchall()}
    cur.close()
    return norms


def buscar(query: str, top_n: int = 10, conn=None) -> list[dict]:
    """
    Recupera los top-N chunks para la query rankeados por similitud de coseno:

        cos(q, d) = (q · d) / (||q|| * ||d||)
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

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    if own_conn:
        conn.close()

    return [{"chunk_id": cid, "score": score} for cid, score in ranked]


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "russia ukraine war"
    print(f"Query: {query!r}\n")
    for i, r in enumerate(buscar(query, top_n=10), start=1):
        print(f"  {i:>2}. chunk {r['chunk_id']:<6} score={r['score']:.4f}")
