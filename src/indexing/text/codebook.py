"""
Módulo Codebook — top-k palabras

Construye el diccionario textual (codebook) seleccionando las K palabras más
frecuentes de toda la colección, tal como indica el enunciado:

    "tomar las k palabras más frecuentes en toda la colección. Estas k palabras
     conforman el diccionario (codebook) textual."

La frecuencia usada es la *frecuencia de colección* (cf): el total de apariciones
del término sumado sobre todos los chunks. El resultado se persiste en la tabla
codebook_text (word_id, word, frequency), que luego usa el índice invertido
(SPIMI) para restringir el vocabulario a indexar a estas K palabras.

Consume el cache generado por extractor.py (data/processed/corpus_stats.pkl); si
no existe, lo reconstruye en el momento.
"""
import os
from collections import defaultdict

from psycopg2.extras import execute_values

from src.core.db import get_connection
from src.indexing.text import extractor

# Tamaño del diccionario. Ajustable según el tamaño del corpus.
K = 5000


def load_stats() -> dict:
    """Carga el cache del extractor; si no está, lo reconstruye desde la BD."""
    if os.path.exists(extractor.CACHE_PATH):
        return extractor.load_cache()
    stats = extractor.build_corpus_stats()
    extractor.save_cache(stats)
    return stats


def compute_collection_frequencies(stats: dict) -> dict[str, int]:
    """Frecuencia de colección por término = suma de tf sobre todos los chunks."""
    cf: dict[str, int] = defaultdict(int)
    for tf_map in stats["term_freq_per_chunk"].values():
        for term, tf in tf_map.items():
            cf[term] += tf
    return cf


def select_top_k(cf: dict[str, int], k: int = K) -> list[tuple[str, int]]:
    """Devuelve las k palabras más frecuentes como [(palabra, frecuencia), ...]."""
    return sorted(cf.items(), key=lambda item: (-item[1], item[0]))[:k]


def build_codebook(k: int = K, conn=None) -> list[tuple[str, int]]:
    """
    Construye el codebook y lo persiste en codebook_text.

    Reinicia la tabla (TRUNCATE ... RESTART IDENTITY CASCADE) para que cada
    construcción sea determinista: word_id queda asignado 1..k en orden de
    frecuencia descendente. El CASCADE limpia inverted_index_text, que depende
    de codebook_text vía FK (debe reindexarse con SPIMI tras reconstruir).
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    stats = load_stats()
    cf = compute_collection_frequencies(stats)
    top = select_top_k(cf, k)

    cur = conn.cursor()
    cur.execute("TRUNCATE codebook_text RESTART IDENTITY CASCADE")
    execute_values(
        cur,
        "INSERT INTO codebook_text (word, frequency) VALUES %s",
        top,
    )
    conn.commit()
    cur.close()

    if own_conn:
        conn.close()

    return top


if __name__ == "__main__":
    stats = load_stats()
    cf = compute_collection_frequencies(stats)
    vocab_size = len(cf)

    top = build_codebook(K)

    print(f"✅ Codebook construido y persistido en codebook_text")
    print(f"   Vocabulario total:     {vocab_size}")
    print(f"   K seleccionado:        {len(top)}")
    cobertura = sum(f for _, f in top) / sum(cf.values()) * 100
    print(f"   Cobertura de tokens:   {cobertura:.1f}%  (los K términos cubren este % del corpus)")

    print(f"\n   Top 10 palabras del codebook (word_id : palabra : frecuencia):")
    for word_id, (word, freq) in enumerate(top[:10], start=1):
        print(f"     {word_id:>3} : {word:<15} : {freq}")

    print(f"\n   Palabra menos frecuente del codebook (word_id {len(top)}): "
          f"{top[-1][0]!r} (freq {top[-1][1]})")
