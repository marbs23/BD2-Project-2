"""
Módulo Extractor — TF-IDF

Pipeline lingüístico sobre los chunks de texto:
    tokenización -> normalización -> eliminación de stopwords -> stemming
y cálculo del peso TF-IDF por chunk usando las fórmulas vistas en clase:

    tf ponderado:  wf(t,d) = 1 + log10(tf)        si tf > 0
    idf:           idf(t)  = log10(N / df(t))
    peso:          w(t,d)  = wf(t,d) * idf(t)

Donde cada *chunk* se trata como un documento de la colección (N = nº de chunks,
df(t) = nº de chunks que contienen el término t). Esto es coherente con el índice
invertido del proyecto, que mapea términos -> chunk_id.

El resultado se cachea en data/processed/corpus_stats.pkl para que los módulos
posteriores (codebook, SPIMI) no tengan que re-tokenizar toda la colección.
"""
import os
import re
import math
import pickle
from collections import defaultdict

from nltk.stem import PorterStemmer

from src.core.db import get_connection
from src.core.paths import PROCESSED_DIR

CACHE_PATH = str(PROCESSED_DIR / "corpus_stats.pkl")

# Regex de tokenización: secuencias de letras (descarta números y puntuación).
_TOKEN_RE = re.compile(r"[a-z]+")


def _load_stopwords() -> set:
    """Carga las stopwords en inglés de nltk; las descarga la primera vez."""
    import nltk
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))


_STEMMER = PorterStemmer()
_STOPWORDS = _load_stopwords()


def tokenize(text: str) -> list[str]:
    """Tokeniza y normaliza: pasa a minúsculas y extrae solo secuencias de letras."""
    return _TOKEN_RE.findall(text.lower())


def preprocess(text: str) -> list[str]:
    """
    Pipeline completo: tokenizar -> quitar stopwords -> stemming.
    Se eliminan stopwords ANTES del stemming (orden del enunciado del proyecto).
    """
    return [
        _STEMMER.stem(tok)
        for tok in tokenize(text)
        if tok not in _STOPWORDS
    ]


def fetch_chunks(conn) -> list[tuple[int, int, str]]:
    """Trae (chunk_id, doc_id, content) de todos los chunks, ordenados por chunk_id."""
    cur = conn.cursor()
    cur.execute(
        "SELECT chunk_id, doc_id, content FROM text_chunks ORDER BY chunk_id"
    )
    rows = cur.fetchall()
    cur.close()
    return rows


def build_corpus_stats(conn=None) -> dict:
    """
    Recorre toda la colección y construye las estadísticas base para TF-IDF.

    Devuelve un dict con:
        term_freq_per_chunk: {chunk_id: {término: tf}}
        doc_freq:            {término: nº de chunks que lo contienen}
        chunk_to_doc:        {chunk_id: doc_id}
        n_chunks:            nº total de chunks (N)
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    chunks = fetch_chunks(conn)

    term_freq_per_chunk: dict[int, dict[str, int]] = {}
    doc_freq: dict[str, int] = defaultdict(int)
    chunk_to_doc: dict[int, int] = {}

    for chunk_id, doc_id, content in chunks:
        tf: dict[str, int] = defaultdict(int)
        for term in preprocess(content):
            tf[term] += 1

        term_freq_per_chunk[chunk_id] = dict(tf)
        chunk_to_doc[chunk_id] = doc_id
        for term in tf:  # cada término cuenta una sola vez por chunk para df
            doc_freq[term] += 1

    if own_conn:
        conn.close()

    return {
        "term_freq_per_chunk": term_freq_per_chunk,
        "doc_freq": dict(doc_freq),
        "chunk_to_doc": chunk_to_doc,
        "n_chunks": len(chunks),
    }


def compute_idf(doc_freq: dict[str, int], n_chunks: int) -> dict[str, float]:
    """idf(t) = log10(N / df(t))."""
    return {
        term: math.log10(n_chunks / df)
        for term, df in doc_freq.items()
    }


def compute_tfidf(stats: dict) -> dict[int, dict[str, float]]:
    """
    Calcula el peso TF-IDF por chunk:
        w(t,d) = (1 + log10(tf)) * idf(t)

    Devuelve {chunk_id: {término: peso_tfidf}}.
    """
    idf = compute_idf(stats["doc_freq"], stats["n_chunks"])
    tfidf: dict[int, dict[str, float]] = {}

    for chunk_id, tf_map in stats["term_freq_per_chunk"].items():
        tfidf[chunk_id] = {
            term: (1 + math.log10(tf)) * idf[term]
            for term, tf in tf_map.items()
        }

    return tfidf


def save_cache(stats: dict, path: str = CACHE_PATH) -> None:
    """Guarda las estadísticas del corpus en disco (pickle)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(stats, f)


def load_cache(path: str = CACHE_PATH) -> dict:
    """Carga las estadísticas del corpus cacheadas."""
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    stats = build_corpus_stats()
    save_cache(stats)

    vocab_size = len(stats["doc_freq"])
    n_chunks = stats["n_chunks"]
    total_terms = sum(
        sum(tf.values()) for tf in stats["term_freq_per_chunk"].values()
    )

    print(f"✅ Estadísticas del corpus calculadas y cacheadas en {CACHE_PATH}")
    print(f"   Chunks (N):            {n_chunks}")
    print(f"   Vocabulario único:     {vocab_size}")
    print(f"   Tokens totales:        {total_terms}")
    print(f"   Promedio términos/chunk: {total_terms / n_chunks:.1f}")

    # Muestra de TF-IDF del primer chunk (top 5 términos por peso)
    tfidf = compute_tfidf(stats)
    primer_chunk = min(tfidf)
    top = sorted(tfidf[primer_chunk].items(), key=lambda x: -x[1])[:5]
    print(f"\n   Ejemplo TF-IDF (chunk {primer_chunk}, top 5 términos):")
    for term, peso in top:
        print(f"     {term:<20} {peso:.4f}")
