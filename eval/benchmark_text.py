"""
Evaluación experimental - pipeline de texto (Fase 4)

Compara el índice invertido propio (SPIMI) contra los índices de texto completo
nativos de PostgreSQL (GIN y GiST) sobre el mismo conjunto de consultas y a distintas
cargas (1K / 10K / 20K chunks). Mide las cinco métricas del enunciado:

  - latencia por consulta, en caché fría (primera) y caliente (repeticiones).
  - throughput (consultas por segundo).
  - precisión@k : fracción del top-k cuya categoría coincide con la de la consulta
                  (ground-truth = categoría del artículo, el proxy de relevancia del
                  proyecto).
  - memoria     : tamaño en disco del índice (tabla inverted_index_text para el propio;
                  índice GIN/GiST para los baselines).
  - accesos I/O : nº de bloques de 8 KB que toca la consulta (EXPLAIN BUFFERS).

Cada carga restringe la búsqueda a sus primeros N chunks (max_chunks); las cargas que
la colección no alcanza se omiten con aviso. Cada baseline se mide con su índice
aislado (GIN o GiST): si coexisten, el planner elegiría uno solo. Semilla fija para
reproducir el muestreo de consultas.
"""
import json
import time
import random

from src.core.db import get_connection
from src.core.paths import ROOT
from src.indexing.text import search as propio
from src.indexing.text.extractor import preprocess
from src.indexing.text.search import _query_term_ids
from eval import text_bench as tb
from eval import metrics

SEED = 42
RESULTS_DIR = ROOT / "eval" / "results"
LOADS = [1000, 10000, 20000]
TOP_K = 10
N_QUERIES = 30
WARM_REPS = 3

# Relación cuyo tamaño en disco representa "el índice" de cada método.
RELACION_INDICE = {
    "propio": "inverted_index_text",
    "gin": "idx_text_gin",
    "gist": "idx_text_gist",
}


def buscar_metodo(nombre: str, query: str, conn, max_chunks: int):
    if nombre == "propio":
        return propio.buscar(query, top_n=TOP_K, conn=conn, max_chunks=max_chunks)
    return tb.buscar(query, top_n=TOP_K, conn=conn, max_chunks=max_chunks)


def total_chunks(conn) -> int:
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM text_chunks")
    n = cur.fetchone()[0]
    cur.close()
    return n


def query_docs(conn, n: int) -> list[tuple[int, str, str]]:
    """Muestra de (doc_id, categoría, texto de consulta) para usar como consultas.

    La consulta es el título del artículo y su categoría es el ground-truth. Solo se
    toman documentos con categoría y con al menos un chunk de texto indexado.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT d.doc_id, d.category, d.title
        FROM documents d
        WHERE d.category IS NOT NULL AND d.category <> ''
          AND d.title IS NOT NULL AND d.title <> ''
          AND EXISTS (SELECT 1 FROM text_chunks tc WHERE tc.doc_id = d.doc_id)
        """
    )
    docs = cur.fetchall()
    cur.close()
    random.Random(SEED).shuffle(docs)
    return docs[:n]


def precision_at_k(resultados: list[dict], categoria: str, doc_id: int) -> float:
    """Fracción del top-k (sin contar la propia consulta) con la misma categoría."""
    vecinos = [r for r in resultados if r["doc_id"] != doc_id][:TOP_K]
    if not vecinos:
        return 0.0
    aciertos = sum(1 for r in vecinos if r["category"] == categoria)
    return aciertos / len(vecinos)


def _io_blocks(nombre: str, query: str, conn, max_chunks: int) -> int:
    """Accesos a bloques de la consulta central del método (proxy de I/O)."""
    if nombre == "propio":
        # Acceso central del propio: leer las posting lists de los términos de la query.
        word_ids = list(_query_term_ids(conn, preprocess(query)))
        if not word_ids:
            return 0
        sql = "SELECT word_id, chunk_id, tf_idf FROM inverted_index_text WHERE word_id = ANY(%s)"
        params: list = [word_ids]
        if max_chunks is not None:
            sql += " AND chunk_id <= %s"
            params.append(max_chunks)
        return metrics.explain_buffers(conn, sql, params)
    # GIN/GiST: la consulta de ranking completa es su acceso central.
    sql, _ = tb.search_sql(max_chunks)
    params = tb._params(query, max_chunks, TOP_K * 16)
    return metrics.explain_buffers(conn, sql, params)


def evaluar_metodo(nombre: str, docs, conn, carga: int) -> dict:
    latencias, precisiones, io_muestras = [], [], []
    cold = None
    for doc_id, categoria, query in docs:
        try:
            t0 = time.perf_counter()
            resultados = buscar_metodo(nombre, query, conn, carga)
            dt = (time.perf_counter() - t0) * 1000
            if cold is None:
                cold = dt
            for _ in range(WARM_REPS):
                t0 = time.perf_counter()
                buscar_metodo(nombre, query, conn, carga)
                latencias.append((time.perf_counter() - t0) * 1000)
            precisiones.append(precision_at_k(resultados, categoria, doc_id))
            # I/O se mide en unas pocas consultas (EXPLAIN ANALYZE es caro).
            if len(io_muestras) < 5:
                io_muestras.append(_io_blocks(nombre, query, conn, carga))
        except Exception as e:
            conn.rollback()  # no arrastrar una transacción abortada al resto del benchmark
            print(f"[aviso] consulta {doc_id} omitida en {nombre}@{carga}: {e}")

    media = sum(latencias) / len(latencias) if latencias else 0.0
    io_medio = round(sum(io_muestras) / len(io_muestras)) if io_muestras else 0
    return {
        "metodo": nombre,
        "carga": carga,
        "latencia_fria_ms": round(cold, 2) if cold else 0.0,
        "latencia_caliente_ms": round(media, 2),
        "throughput_qps": round(1000 / media, 1) if media else 0.0,
        "precision_at_k": round(sum(precisiones) / len(precisiones), 3) if precisiones else 0.0,
        "io_bloques": io_medio,
        "consultas": len(docs),
    }


def run() -> dict:
    conn = get_connection()
    tb.ensure_tsv(conn)  # columna tsvector para que GIN/GiST puedan construirse
    n_chunks = total_chunks(conn)
    docs = query_docs(conn, N_QUERIES)

    cargas = [n for n in LOADS if n <= n_chunks]
    omitidas = [n for n in LOADS if n > n_chunks]
    if omitidas:
        print(f"[aviso] cargas omitidas por falta de datos ({n_chunks} chunks "
              f"indexados): {omitidas}")
    if not cargas:
        cargas = [n_chunks]
        print(f"[aviso] ninguna carga estándar alcanzable; se mide sobre los {n_chunks} chunks disponibles")

    resultados = []
    memoria = {}
    for nombre in ("propio", "gin", "gist"):
        if nombre in ("gin", "gist"):
            print(f"Construyendo índice de texto aislado: {nombre}")
            tb.create_index(nombre, conn=conn)
        memoria[nombre] = metrics.index_size_bytes(conn, RELACION_INDICE[nombre])
        for carga in cargas:
            print(f"Evaluando {nombre} @ {carga} chunks")
            r = evaluar_metodo(nombre, docs, conn, carga)
            r["memoria_bytes"] = memoria[nombre]
            resultados.append(r)
    tb.drop_indexes(conn)
    conn.close()

    salida = {
        "chunks_indexados": n_chunks,
        "cargas": cargas,
        "top_k": TOP_K,
        "resultados": resultados,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "text_benchmark.json").write_text(
        json.dumps(salida, indent=2, ensure_ascii=False)
    )
    _graficos(resultados, cargas)
    return salida


def _graficos(resultados: list[dict], cargas: list[int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metodos = ["propio", "gin", "gist"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    (ax1, ax2), (ax3, ax4) = axes
    carga_max = max(cargas)

    if len(cargas) == 1:
        carga = cargas[0]
        por_metodo = {r["metodo"]: r for r in resultados if r["carga"] == carga}
        ax1.bar(metodos, [por_metodo[m]["latencia_caliente_ms"] for m in metodos], color="#4C72B0")
        ax1.set_title(f"Latencia caliente · {carga} chunks"); ax1.set_ylabel("ms por consulta")
        ax2.bar(metodos, [por_metodo[m]["precision_at_k"] for m in metodos], color="#55A868")
        ax2.set_title(f"Precisión@{TOP_K} · {carga} chunks"); ax2.set_ylim(0, 1)
    else:
        for m in metodos:
            serie = sorted((r for r in resultados if r["metodo"] == m), key=lambda r: r["carga"])
            ax1.plot([r["carga"] for r in serie], [r["latencia_caliente_ms"] for r in serie], marker="o", label=m)
            ax2.plot([r["carga"] for r in serie], [r["precision_at_k"] for r in serie], marker="o", label=m)
        ax1.set_title("Latencia caliente vs carga"); ax1.set_xlabel("chunks"); ax1.set_ylabel("ms"); ax1.legend()
        ax2.set_title(f"Precisión@{TOP_K} vs carga"); ax2.set_xlabel("chunks"); ax2.set_ylim(0, 1); ax2.legend()

    # Memoria: tamaño del índice por método (constante respecto a la carga).
    mem_mb = {r["metodo"]: r["memoria_bytes"] / (1024 * 1024) for r in resultados}
    ax3.bar(metodos, [mem_mb.get(m, 0) for m in metodos], color="#C44E52")
    ax3.set_title("Tamaño del índice en disco"); ax3.set_ylabel("MB")

    # I/O: bloques accedidos por consulta, a la carga mayor.
    io_max = {r["metodo"]: r["io_bloques"] for r in resultados if r["carga"] == carga_max}
    ax4.bar(metodos, [io_max.get(m, 0) for m in metodos], color="#8172B3")
    ax4.set_title(f"Accesos I/O por consulta · {carga_max} chunks"); ax4.set_ylabel("bloques (8 KB)")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "text_benchmark.png", dpi=120)


if __name__ == "__main__":
    salida = run()
    print("\nResultados (texto):")
    for r in salida["resultados"]:
        print(f"  {r['metodo']:>7} @ {r['carga']:>6} · fría {r['latencia_fria_ms']:>7.2f}ms · "
              f"caliente {r['latencia_caliente_ms']:>6.2f}ms · {r['throughput_qps']:>6.1f} q/s · "
              f"P@{TOP_K}={r['precision_at_k']} · {metrics.human_bytes(r['memoria_bytes'])} · "
              f"{r['io_bloques']} bloques")
    print(f"\nGuardado en {RESULTS_DIR}/ (text_benchmark.json + text_benchmark.png)")
