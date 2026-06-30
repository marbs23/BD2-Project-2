"""
Evaluación experimental - pipeline de imagen (issue #26)

Compara el índice invertido propio contra los baselines de pgvector (HNSW, IVFFlat)
sobre el mismo conjunto de consultas y a distintas cargas (1K / 10K / 100K chunks).
Mide:

  - latencia por consulta, en caché fría (primera) y caliente (repetidas).
  - throughput (consultas por segundo).
  - precisión@k: fracción del top-k cuya categoría coincide con la de la consulta
    (ground-truth = categoría del artículo, el proxy de relevancia del proyecto).

Cada carga restringe la búsqueda a sus primeros N chunks (max_chunks); las cargas
que la colección no alcanza se omiten con aviso. Cada baseline pgvector se mide con
su índice aislado: si HNSW e IVFFlat coexisten, el planner elegiría uno solo y las
dos filas medirían lo mismo. Semilla fija para reproducir el muestreo de consultas.
"""
import os
import sys
import json
import time
import random
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_module.connection import get_connection
from image_module import search as propio
from image_module import pgvector_bench as pg

SEED = 42
RESULTS_DIR = Path("eval/results")
IMAGES_DIR = Path("data/raw/images")
LOADS = [1000, 10000, 100000]
TOP_K = 10
N_QUERIES = 30
WARM_REPS = 3


def buscar_metodo(nombre: str, path: str, conn, max_chunks: int):
    if nombre == "propio":
        return propio.buscar(path, top_n=TOP_K, conn=conn, max_chunks=max_chunks)
    return pg.buscar(path, top_n=TOP_K, conn=conn, max_chunks=max_chunks)


def total_chunks(conn) -> int:
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM image_chunks")
    n = cur.fetchone()[0]
    cur.close()
    return n


def query_docs(conn, n: int) -> list[tuple[int, str]]:
    """Muestra de (doc_id, categoría) con imagen en disco, para usar como consultas."""
    cur = conn.cursor()
    cur.execute("SELECT doc_id, category FROM documents WHERE image_url IS NOT NULL AND image_url <> ''")
    docs = [(d, c) for d, c in cur.fetchall() if (IMAGES_DIR / f"{d}.jpg").exists()]
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


def evaluar_metodo(nombre: str, docs, conn, carga: int) -> dict:
    latencias, precisiones = [], []
    cold = None
    for doc_id, categoria in docs:
        path = str(IMAGES_DIR / f"{doc_id}.jpg")
        try:
            t0 = time.perf_counter()
            resultados = buscar_metodo(nombre, path, conn, carga)
            dt = (time.perf_counter() - t0) * 1000
            if cold is None:
                cold = dt
            for _ in range(WARM_REPS):
                t0 = time.perf_counter()
                buscar_metodo(nombre, path, conn, carga)
                latencias.append((time.perf_counter() - t0) * 1000)
            precisiones.append(precision_at_k(resultados, categoria, doc_id))
        except Exception as e:
            conn.rollback()  # no arrastrar una transacción abortada al resto del benchmark
            print(f"[aviso] consulta {doc_id} omitida en {nombre}@{carga}: {e}")

    media = sum(latencias) / len(latencias) if latencias else 0.0
    return {
        "metodo": nombre,
        "carga": carga,
        "latencia_fria_ms": round(cold, 2) if cold else 0.0,
        "latencia_caliente_ms": round(media, 2),
        "throughput_qps": round(1000 / media, 1) if media else 0.0,
        "precision_at_k": round(sum(precisiones) / len(precisiones), 3) if precisiones else 0.0,
        "consultas": len(docs),
    }


def run() -> dict:
    conn = get_connection()
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
    for nombre in ("propio", "hnsw", "ivfflat"):
        if nombre in ("hnsw", "ivfflat"):
            print(f"Construyendo índice pgvector aislado: {nombre}")
            pg.create_index(nombre, conn=conn)
        for carga in cargas:
            print(f"Evaluando {nombre} @ {carga} chunks")
            resultados.append(evaluar_metodo(nombre, docs, conn, carga))
    pg.drop_indexes(conn)
    conn.close()

    salida = {
        "chunks_indexados": n_chunks,
        "cargas": cargas,
        "top_k": TOP_K,
        "resultados": resultados,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "benchmark.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    _graficos(resultados, cargas)
    return salida


def _graficos(resultados: list[dict], cargas: list[int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metodos = ["propio", "hnsw", "ivfflat"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    if len(cargas) == 1:
        # Una sola carga: barras por método.
        carga = cargas[0]
        por_metodo = {r["metodo"]: r for r in resultados if r["carga"] == carga}
        ax1.bar(metodos, [por_metodo[m]["latencia_caliente_ms"] for m in metodos], color="#4C72B0")
        ax1.set_title(f"Latencia caliente · {carga} chunks")
        ax1.set_ylabel("ms por consulta")
        ax2.bar(metodos, [por_metodo[m]["precision_at_k"] for m in metodos], color="#55A868")
        ax2.set_title(f"Precisión@{TOP_K} · {carga} chunks")
        ax2.set_ylim(0, 1)
    else:
        # Varias cargas: una línea por método contra el tamaño de la colección.
        for m in metodos:
            serie = sorted((r for r in resultados if r["metodo"] == m), key=lambda r: r["carga"])
            ax1.plot([r["carga"] for r in serie], [r["latencia_caliente_ms"] for r in serie], marker="o", label=m)
            ax2.plot([r["carga"] for r in serie], [r["precision_at_k"] for r in serie], marker="o", label=m)
        ax1.set_title("Latencia caliente vs carga"); ax1.set_xlabel("chunks"); ax1.set_ylabel("ms"); ax1.legend()
        ax2.set_title(f"Precisión@{TOP_K} vs carga"); ax2.set_xlabel("chunks"); ax2.set_ylim(0, 1); ax2.legend()

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "benchmark.png", dpi=120)


if __name__ == "__main__":
    salida = run()
    print("\nResultados:")
    for r in salida["resultados"]:
        print(f"  {r['metodo']:>8} @ {r['carga']:>6} · fría {r['latencia_fria_ms']:>7.2f}ms · "
              f"caliente {r['latencia_caliente_ms']:>6.2f}ms · "
              f"{r['throughput_qps']:>6.1f} q/s · P@{TOP_K}={r['precision_at_k']}")
    print(f"\nGuardado en {RESULTS_DIR}/ (benchmark.json + benchmark.png)")
