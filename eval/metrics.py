"""
Métricas de memoria e I/O para la evaluación experimental (Fase 4)

Complementan a latencia/throughput/precisión con dos métricas que se miden desde
la propia base de datos, sin instrumentar el proceso Python:

  - memoria : tamaño en disco del índice (o de la tabla del índice invertido
              propio), vía pg_relation_size / pg_total_relation_size.
  - I/O     : nº de bloques (páginas de 8 KB) que toca una consulta, leído del plan
              de EXPLAIN (ANALYZE, BUFFERS): Shared Hit Blocks + Shared Read Blocks.

Ambas son nativas de PostgreSQL y reproducibles, así que sirven para comparar de
forma justa el índice invertido propio contra GIN/GiST y pgvector.
"""


def index_size_bytes(conn, relation: str) -> int:
    """
    Tamaño en disco de una relación (índice o tabla), en bytes.

    Para los índices nativos (GIN/GiST/HNSW/IVFFlat) se pasa el nombre del índice y
    se usa pg_relation_size. Para el índice invertido "propio" no hay un índice de
    Postgres: su estructura ES la tabla inverted_index_text/_image, así que se pasa
    el nombre de la tabla y se usa pg_total_relation_size (datos + índices de PK +
    TOAST), que es el equivalente honesto de "cuánto ocupa el índice".

    Devuelve 0 si la relación no existe (p.ej. el índice aún no se creó).
    """
    cur = conn.cursor()
    cur.execute("SELECT to_regclass(%s)", (relation,))
    if cur.fetchone()[0] is None:
        cur.close()
        return 0
    # Una tabla (índice invertido propio) se mide completa; un índice, solo su relación.
    func = "pg_total_relation_size" if _is_table(cur, relation) else "pg_relation_size"
    cur.execute(f"SELECT {func}(%s)", (relation,))
    size = cur.fetchone()[0]
    cur.close()
    return int(size)


def _is_table(cur, relation: str) -> bool:
    """True si la relación es una tabla ordinaria; False si es un índice."""
    cur.execute("SELECT relkind FROM pg_class WHERE oid = to_regclass(%s)", (relation,))
    row = cur.fetchone()
    return bool(row) and row[0] == "r"


def explain_buffers(conn, sql: str, params=None) -> int:
    """
    Nº de bloques compartidos que accede la consulta (hit + read), del plan real.

    Corre EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) y suma los contadores de buffers
    de todo el árbol del plan. Es el proxy de "accesos a I/O": cuántas páginas de
    8 KB tuvo que tocar el motor (estén en caché -hit- o vengan de disco -read-).

    La consulta se ejecuta de verdad (ANALYZE); usar solo la SQL de lectura del
    método, nunca DDL.
    """
    cur = conn.cursor()
    cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}", params)
    plan = cur.fetchone()[0][0]["Plan"]
    cur.close()
    return _sum_buffers(plan)


def _sum_buffers(node: dict) -> int:
    """Suma recursiva de Shared Hit/Read Blocks sobre todos los nodos del plan."""
    total = node.get("Shared Hit Blocks", 0) + node.get("Shared Read Blocks", 0)
    for child in node.get("Plans", []):
        total += _sum_buffers(child)
    return int(total)


def human_bytes(n: float) -> str:
    """Formatea bytes a KB/MB/GB legibles para las tablas del informe."""
    size = float(n)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
