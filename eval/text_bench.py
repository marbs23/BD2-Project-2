"""
Comparativa full-text nativo de PostgreSQL - GIN / GiST

Baselines contra el índice invertido propio (SPIMI). En lugar de cruzar las posting
lists de inverted_index_text, se indexa el texto de cada chunk con los índices de
texto completo nativos de Postgres y se rankea con ts_rank:

  - GIN  : índice invertido interno de Postgres; consultas rápidas sobre tsvector,
           índice más grande y algo más caro de construir.
  - GiST : árbol con firmas (lossy); índice más pequeño y barato, consultas más
           lentas y con posibles falsos positivos que el motor re-verifica.

Ambos operan sobre la misma columna tsvector y las mismas consultas que el método
propio, para que la comparación de latencia, memoria, I/O y resultados sea justa.
La relevancia se mide con ts_rank (equivalente nativo del score TF-IDF por coseno).
"""
import sys

from src.core.db import get_connection

METODOS = ("gin", "gist")
TS_CONFIG = "english"  # misma lengua que el preprocesamiento del extractor propio


def ensure_tsv(conn) -> None:
    """
    Garantiza la columna tsvector generada sobre text_chunks.content.

    Se crea desde el benchmark (no en db/schema.sql) para no acoplar el esquema de
    la app con la evaluación. Es idempotente: si la columna ya existe, no hace nada.
    STORED => se materializa una vez y GIN/GiST indexan sobre ella.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'text_chunks' AND column_name = 'content_tsv'
        """
    )
    if cur.fetchone() is None:
        cur.execute(
            f"""
            ALTER TABLE text_chunks
            ADD COLUMN content_tsv tsvector
            GENERATED ALWAYS AS (to_tsvector('{TS_CONFIG}', coalesce(content, ''))) STORED
            """
        )
        conn.commit()
    cur.close()


def index_name(method: str) -> str:
    """Nombre del índice de texto completo según el método."""
    return "idx_text_gin" if method == "gin" else "idx_text_gist"


def drop_indexes(conn) -> None:
    """Elimina ambos índices de texto completo sobre text_chunks.content_tsv."""
    cur = conn.cursor()
    cur.execute("DROP INDEX IF EXISTS idx_text_gin")
    cur.execute("DROP INDEX IF EXISTS idx_text_gist")
    conn.commit()
    cur.close()


def create_index(method: str, conn=None) -> None:
    """
    Crea el índice de texto elegido dejándolo como el único sobre content_tsv.

    Se borran ambos antes de crear: si GIN y GiST coexisten, el planner elegiría uno
    por costo y el benchmark mediría el índice equivocado (mismo criterio que
    pgvector_bench.create_index con HNSW/IVFFlat).
    """
    if method not in METODOS:
        raise ValueError(f"método desconocido: {method}")
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    ensure_tsv(conn)
    drop_indexes(conn)
    cur = conn.cursor()
    using = "gin" if method == "gin" else "gist"
    cur.execute(
        f"CREATE INDEX {index_name(method)} ON text_chunks USING {using} (content_tsv)"
    )
    conn.commit()
    cur.close()
    if own_conn:
        conn.close()


def search_sql(max_chunks: int | None) -> tuple[str, str]:
    """
    SQL de búsqueda full-text y su cláusula WHERE, reutilizable por buscar() y por
    metrics.explain_buffers (para medir I/O sobre exactamente la misma consulta).

    Devuelve (sql, where) donde la SQL usa placeholders posicionales:
      %s -> query (texto de la consulta)  [aparece dos veces: filtro y ranking]
      %s -> max_chunks (solo si aplica)
      %s -> límite
    """
    where = f"WHERE tc.content_tsv @@ websearch_to_tsquery('{TS_CONFIG}', %s)"
    if max_chunks is not None:
        where += " AND tc.chunk_id <= %s"
    sql = f"""
        SELECT tc.chunk_id, tc.doc_id, d.title, d.url, d.category, d.image_url,
               ts_rank(tc.content_tsv, websearch_to_tsquery('{TS_CONFIG}', %s)) AS rank
        FROM text_chunks tc
        JOIN documents d ON d.doc_id = tc.doc_id
        {where}
        ORDER BY rank DESC
        LIMIT %s
    """
    return sql, where


def _params(query: str, max_chunks: int | None, limite: int) -> list:
    """Arma la lista de parámetros en el orden en que aparecen en search_sql."""
    params: list = [query]  # filtro @@
    if max_chunks is not None:
        params.append(max_chunks)
    params += [query, limite]  # ranking ts_rank + LIMIT
    return params


def buscar(query: str, top_n: int = 10, group_by_doc: bool = True,
           conn=None, max_chunks: int | None = None) -> list[dict]:
    """
    Top-N documentos por relevancia full-text (ts_rank) usando el índice GIN/GiST.

    max_chunks restringe la búsqueda a los primeros N chunks (carga del benchmark).
    Devuelve el mismo formato de dict que src.indexing.text.search.buscar para que
    precisión y ranking se comparen directamente.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    # Margen sobre top_n: el dedup por documento puede colapsar varios chunks del
    # mismo artículo en una sola fila de resultado.
    limite = top_n * 16 if group_by_doc else top_n
    sql, _ = search_sql(max_chunks)
    cur = conn.cursor()
    cur.execute(sql, _params(query, max_chunks, limite))
    filas = cur.fetchall()
    cur.close()
    if own_conn:
        conn.close()

    results = []
    seen: set[int] = set()
    for chunk_id, doc_id, title, url, category, image_url, rank in filas:
        if group_by_doc and doc_id in seen:
            continue
        seen.add(doc_id)
        results.append({
            "chunk_id": chunk_id, "doc_id": doc_id, "title": title, "url": url,
            "category": category, "image_url": image_url, "score": float(rank),
        })
        if len(results) >= top_n:
            break
    return results


if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else "gin"
    query = " ".join(sys.argv[2:]) or "russia ukraine war"
    print(f"Creando índice {method}...")
    create_index(method)
    print("  ok")

    print(f"\nConsulta ({method}): {query!r}\n")
    for i, r in enumerate(buscar(query, top_n=10), 1):
        print(f"  {i:>2}. [{r['score']:.4f}] {r['title']}  ({r['category']})")
