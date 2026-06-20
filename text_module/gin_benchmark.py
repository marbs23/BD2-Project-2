"""
Benchmark de latencia (issue #23, Fase 3)

Compara la latencia de la búsqueda full-text nativa (índice GIN) contra el índice
invertido propio (search.buscar), sobre las mismas consultas. Además ejecuta
EXPLAIN (ANALYZE, BUFFERS) para demostrar que la consulta full-text usa el índice
GIN y reportar accesos a buffers (I/O).

Insumo para el análisis comparativo de la Fase 4.
"""

import os
import sys
import time
import statistics

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_module.connection import get_connection
from text_module.search import buscar
from text_module.gin_search import buscar_fulltext

QUERIES = [
    "russia ukraine war",
    "covid vaccine health",
    "football premier league",
    "climate change energy",
    "boris johnson government",
]
REPEATS = 50


def _avg_ms(fn, conn, query: str, repeats: int) -> float:
    """Latencia promedio (ms) de fn sobre una query, con un warmup previo."""
    fn(query, top_n=10, conn=conn)  # warmup (calienta caché del plan/datos)
    tiempos = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(query, top_n=10, conn=conn)
        tiempos.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(tiempos)


def run():
    conn = get_connection()

    print(f"Latencia promedio sobre {REPEATS} repeticiones (ms):\n")
    print(f"{'Query':<28}{'GIN full-text':>16}{'Inv. propio':>16}")
    print("-" * 60)
    gin_all, inv_all = [], []
    for q in QUERIES:
        gin = _avg_ms(buscar_fulltext, conn, q, REPEATS)
        inv = _avg_ms(buscar, conn, q, REPEATS)
        gin_all.append(gin)
        inv_all.append(inv)
        print(f"{q:<28}{gin:>14.2f}ms{inv:>14.2f}ms")
    print("-" * 60)
    print(f"{'PROMEDIO':<28}{statistics.mean(gin_all):>14.2f}ms{statistics.mean(inv_all):>14.2f}ms")

    print("\nEXPLAIN (ANALYZE, BUFFERS) de la consulta full-text:")
    print("(busca 'Bitmap Index Scan on idx_text_chunks_tsv' => usa el índice GIN)\n")
    cur = conn.cursor()
    cur.execute(
        """
        EXPLAIN (ANALYZE, BUFFERS)
        SELECT tc.chunk_id, ts_rank(tc.tsv, q) AS rank
        FROM text_chunks tc, plainto_tsquery('english', %s) AS q
        WHERE tc.tsv @@ q
        ORDER BY rank DESC LIMIT 100
        """,
        ("russia ukraine war",),
    )
    for (linea,) in cur.fetchall():
        print("   ", linea)
    cur.close()
    conn.close()


if __name__ == "__main__":
    run()
