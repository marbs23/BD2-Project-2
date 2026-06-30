"""
Módulo Índice Invertido — SPIMI

Implementa SPIMI (Single-Pass In-Memory Indexing) para construir el índice
invertido de texto. El algoritmo procesa los chunks en una sola pasada por
streaming desde la BD (nunca carga toda la colección en RAM):

    1. BLOQUES  : acumula postings (término -> [(chunk_id, tf), ...]) en un
                  diccionario en memoria; cuando se llena el buffer, ordena por
                  término y vuelca el bloque a disco.  <-- ESTE ARCHIVO (fase 1)
    2. MERGE    : mezcla los bloques ordenados en posting lists completas.
    3. ÍNDICE   : calcula TF-IDF y persiste en inverted_index_text.

El vocabulario a indexar se restringe a las palabras del codebook (top-k),
coherente con la idea de "inversión lingüística" del proyecto: solo las palabras
del diccionario mapean a chunks.
"""
import os
import glob
import math
import heapq
from collections import defaultdict

from psycopg2.extras import execute_values

from src.core.db import get_connection
from src.core.paths import PROCESSED_DIR
from src.indexing.text.extractor import preprocess

# Tamaño de lote para los INSERT en inverted_index_text.
INSERT_BATCH = 5000

# Directorio donde se vuelcan los bloques intermedios (gitignored).
BLOCK_DIR = str(PROCESSED_DIR / "spimi_blocks")

# Límite de postings en memoria antes de volcar un bloque a disco. Deliberadamente
# modesto para que el spill a disco se ejerza incluso con la muestra actual; subir
# para corpus grandes (Fase 4).
BLOCK_SIZE_POSTINGS = 10000


def load_codebook(conn) -> dict[str, int]:
    """Carga el codebook como {palabra: word_id}. Es el vocabulario a indexar."""
    cur = conn.cursor()
    cur.execute("SELECT word, word_id FROM codebook_text")
    mapping = {word: word_id for word, word_id in cur.fetchall()}
    cur.close()
    return mapping


def count_chunks(conn) -> int:
    """Número total de chunks (N), usado luego para el IDF."""
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM text_chunks")
    n = cur.fetchone()[0]
    cur.close()
    return n


def stream_chunks(batch_size: int = 500):
    """
    Itera (chunk_id, content) desde la BD usando un cursor del lado servidor,
    de modo que no se cargan todos los chunks en memoria a la vez (streaming real).
    Ordenado por chunk_id para que las posting lists queden ordenadas por chunk_id.
    """
    conn = get_connection()
    cur = conn.cursor(name="spimi_stream")  # cursor con nombre => server-side
    cur.itersize = batch_size
    cur.execute("SELECT chunk_id, content FROM text_chunks ORDER BY chunk_id")
    try:
        for row in cur:
            yield row
    finally:
        cur.close()
        conn.close()


def write_block(block: dict[str, list[tuple[int, int]]], block_num: int) -> str:
    """
    Vuelca un bloque a disco, ordenado por término. Formato por línea:
        término\\tchunk_id:tf chunk_id:tf ...
    Dentro de cada término las postings ya vienen en orden de chunk_id.
    """
    path = os.path.join(BLOCK_DIR, f"block_{block_num:04d}.tsv")
    with open(path, "w", encoding="utf-8") as f:
        for term in sorted(block):
            postings = block[term]
            postings_str = " ".join(f"{cid}:{tf}" for cid, tf in postings)
            f.write(f"{term}\t{postings_str}\n")
    return path


def build_blocks(vocab: set[str]) -> list[str]:
    """
    Fase 1 de SPIMI: pasada única por streaming generando bloques en disco.

    Por cada chunk se cuentan las frecuencias de los términos que están en el
    vocabulario (codebook) y se agregan al diccionario en memoria. Al superar
    BLOCK_SIZE_POSTINGS postings acumulados, se vuelca un bloque ordenado y se
    vacía el diccionario.

    Devuelve la lista de rutas de los bloques generados.
    """
    os.makedirs(BLOCK_DIR, exist_ok=True)
    # Limpia bloques de corridas anteriores.
    for old in glob.glob(os.path.join(BLOCK_DIR, "block_*.tsv")):
        os.remove(old)

    block: dict[str, list[tuple[int, int]]] = defaultdict(list)
    n_postings = 0
    block_num = 0
    block_files: list[str] = []

    for chunk_id, content in stream_chunks():
        tf: dict[str, int] = defaultdict(int)
        for term in preprocess(content):
            if term in vocab:
                tf[term] += 1

        for term, freq in tf.items():
            block[term].append((chunk_id, freq))
            n_postings += 1

        if n_postings >= BLOCK_SIZE_POSTINGS:
            block_files.append(write_block(block, block_num))
            block_num += 1
            block = defaultdict(list)
            n_postings = 0

    # Último bloque parcial.
    if block:
        block_files.append(write_block(block, block_num))

    return block_files


def _parse_postings(postings_str: str) -> list[tuple[int, int]]:
    """Convierte 'chunk_id:tf chunk_id:tf ...' en [(chunk_id, tf), ...]."""
    postings = []
    for token in postings_str.split():
        cid, tf = token.split(":")
        postings.append((int(cid), int(tf)))
    return postings


def _iter_block(path: str):
    """Itera (término, postings) de un bloque en disco (ya ordenado por término)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            term, postings_str = line.rstrip("\n").split("\t")
            yield term, _parse_postings(postings_str)


def merge_blocks(block_files: list[str]):
    """
    Fase 2 de SPIMI: merge k-way de los bloques ordenados (mezcla externa).

    Usa un heap con la cabeza de cada bloque. Para cada término se combinan las
    posting lists de todos los bloques que lo contienen, concatenándolas en orden
    de bloque (los bloques se generaron en orden de chunk_id creciente, así que la
    posting list resultante queda ordenada por chunk_id).

    Genera (término, [(chunk_id, tf), ...]) en orden alfabético de término.
    """
    iters = [_iter_block(bf) for bf in block_files]
    heap: list[tuple[str, int, list[tuple[int, int]]]] = []

    # Cabeza inicial de cada bloque.
    for i, it in enumerate(iters):
        term, postings = next(it, (None, None))
        if term is not None:
            heapq.heappush(heap, (term, i, postings))

    while heap:
        current_term = heap[0][0]
        # Junta todas las entradas del término mínimo (puede estar en varios bloques).
        same_term: list[tuple[int, list[tuple[int, int]]]] = []
        while heap and heap[0][0] == current_term:
            term, i, postings = heapq.heappop(heap)
            same_term.append((i, postings))
            nxt_term, nxt_postings = next(iters[i], (None, None))
            if nxt_term is not None:
                heapq.heappush(heap, (nxt_term, i, nxt_postings))

        # Concatena en orden de bloque (i ascendente) => preserva orden de chunk_id.
        same_term.sort(key=lambda x: x[0])
        merged: list[tuple[int, int]] = []
        for _, postings in same_term:
            merged.extend(postings)

        yield current_term, merged


def build_index(conn=None) -> dict:
    """
    Pipeline SPIMI completo: bloques -> merge -> TF-IDF -> inverted_index_text.

    Para cada término se calcula, con las fórmulas vistas en clase:
        df  = nº de chunks que contienen el término (largo de la posting list)
        idf = log10(N / df)
        peso(chunk) = (1 + log10(tf)) * idf

    y se persiste cada posting en inverted_index_text (word_id, chunk_id, tf_idf).
    Además acumula y guarda la norma ||d|| = sqrt(sum(tf_idf^2)) de cada chunk en
    text_chunks.norm, para que la búsqueda por coseno no la recalcule en cada query.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    vocab_map = load_codebook(conn)
    n_chunks = count_chunks(conn)

    # Fase 1 + 2: bloques en disco y merge.
    block_files = build_blocks(set(vocab_map))

    cur = conn.cursor()
    cur.execute("TRUNCATE inverted_index_text")

    batch: list[tuple[int, int, float]] = []
    norm_sq: dict[int, float] = defaultdict(float)  # acumulador de sum(tf_idf^2) por chunk
    n_terms = 0
    n_postings = 0

    def flush():
        if batch:
            execute_values(
                cur,
                "INSERT INTO inverted_index_text (word_id, chunk_id, tf_idf) VALUES %s",
                batch,
            )
            batch.clear()

    for term, postings in merge_blocks(block_files):
        n_terms += 1
        df = len(postings)
        idf = math.log10(n_chunks / df)
        word_id = vocab_map[term]
        for chunk_id, tf in postings:
            tf_idf = (1 + math.log10(tf)) * idf
            batch.append((word_id, chunk_id, tf_idf))
            norm_sq[chunk_id] += tf_idf * tf_idf
            n_postings += 1
            if len(batch) >= INSERT_BATCH:
                flush()

    flush()

    # Persistir la norma de cada chunk (||d||) en text_chunks.norm.
    cur.execute("UPDATE text_chunks SET norm = NULL")
    norms = [(chunk_id, math.sqrt(sq)) for chunk_id, sq in norm_sq.items()]
    execute_values(
        cur,
        """
        UPDATE text_chunks AS t SET norm = v.norm
        FROM (VALUES %s) AS v(chunk_id, norm)
        WHERE t.chunk_id = v.chunk_id
        """,
        norms,
    )

    conn.commit()
    cur.close()

    if own_conn:
        conn.close()

    return {
        "n_blocks": len(block_files),
        "n_terms": n_terms,
        "n_postings": n_postings,
        "n_chunks": n_chunks,
        "n_norms": len(norms),
    }


if __name__ == "__main__":
    print(f"Construyendo índice invertido con SPIMI (BLOCK_SIZE={BLOCK_SIZE_POSTINGS})...")
    stats = build_index()

    print(f"\n✅ Índice invertido construido y persistido en inverted_index_text")
    print(f"   Bloques en disco (fase 1): {stats['n_blocks']}")
    print(f"   Términos indexados:        {stats['n_terms']}")
    print(f"   Postings insertados:       {stats['n_postings']}")
    print(f"   Normas de chunk guardadas: {stats['n_norms']}")
    print(f"   Chunks (N):                {stats['n_chunks']}")

    # Verificación directa contra la BD.
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM inverted_index_text")
    print(f"\n   inverted_index_text filas: {cur.fetchone()[0]}")

    cur.execute(
        """
        SELECT cb.word, ii.chunk_id, ii.tf_idf
        FROM inverted_index_text ii
        JOIN codebook_text cb ON cb.word_id = ii.word_id
        ORDER BY ii.tf_idf DESC
        LIMIT 5
        """
    )
    print(f"   Top 5 postings por TF-IDF (palabra, chunk_id, tf_idf):")
    for word, chunk_id, tf_idf in cur.fetchall():
        print(f"     {word:<15} chunk {chunk_id:<6} {tf_idf:.4f}")

    cur.close()
    conn.close()
