"""
Módulo Índice Invertido — SPIMI (issue #6)

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
import sys
import glob
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_module.connection import get_connection
from text_module.extractor import preprocess

# Directorio donde se vuelcan los bloques intermedios (gitignored).
BLOCK_DIR = "data/processed/spimi_blocks"

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


if __name__ == "__main__":
    conn = get_connection()
    vocab_map = load_codebook(conn)
    n_chunks = count_chunks(conn)
    conn.close()

    print(f"Codebook (vocabulario a indexar): {len(vocab_map)} palabras")
    print(f"Chunks a procesar (N):            {n_chunks}")
    print(f"BLOCK_SIZE_POSTINGS:              {BLOCK_SIZE_POSTINGS}")

    block_files = build_blocks(set(vocab_map))

    total_postings = 0
    for bf in block_files:
        with open(bf, encoding="utf-8") as f:
            for line in f:
                total_postings += len(line.split("\t")[1].split())

    print(f"\n✅ Fase 1 (bloques) completada")
    print(f"   Bloques escritos en disco: {len(block_files)}  ({BLOCK_DIR}/)")
    print(f"   Postings totales:          {total_postings}")
