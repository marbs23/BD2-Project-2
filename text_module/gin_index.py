"""
Índice GIN para texto (issue #23, Fase 3 — comparación con PostgreSQL)

Crea la infraestructura de full-text nativo de PostgreSQL sobre los chunks:
  - una columna generada `text_chunks.tsv` (tsvector de `content`), que Postgres
    mantiene automáticamente, y
  - un índice GIN sobre esa columna para acelerar las consultas `tsv @@ query`.

Es idempotente: se puede correr varias veces sin error. La columna generada
auto-pobla las filas existentes al crearse.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_module.connection import get_connection

INDEX_NAME = "idx_text_chunks_tsv"


def setup_gin(conn=None) -> None:
    """Agrega la columna tsvector generada y el índice GIN si no existen."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    cur = conn.cursor()
    cur.execute("""
        ALTER TABLE text_chunks
        ADD COLUMN IF NOT EXISTS tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
    """)
    cur.execute(f"CREATE INDEX IF NOT EXISTS {INDEX_NAME} ON text_chunks USING GIN (tsv)")
    conn.commit()
    cur.close()

    if own_conn:
        conn.close()


if __name__ == "__main__":
    setup_gin()

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM text_chunks WHERE tsv IS NOT NULL")
    poblados = cur.fetchone()[0]
    cur.execute("SELECT pg_size_pretty(pg_relation_size(%s))", (INDEX_NAME,))
    tam = cur.fetchone()[0]
    cur.close()
    conn.close()

    print("✅ Columna tsvector e índice GIN listos")
    print(f"   Chunks con tsv: {poblados}")
    print(f"   Tamaño índice GIN ({INDEX_NAME}): {tam}")
