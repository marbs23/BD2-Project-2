"""
Corre TODOS los benchmarks de la Fase 4 con un solo comando.

Ejecuta de punta a punta los dos tracks de evaluación y deja los artefactos listos
para el informe del README:

  1. Texto : índice invertido propio (SPIMI) vs GIN vs GiST.
  2. Imagen: índice invertido propio vs pgvector (HNSW / IVFFlat).

Al terminar imprime una tabla resumen con las cinco métricas (latencia, throughput,
precisión@k, memoria e I/O) por método y carga, y la ruta de los JSON/PNG generados
en eval/results/.

Uso:
    python -m eval.run_all
    ./scripts/run_benchmarks.sh
"""
from eval import benchmark_text, benchmark, metrics
from src.core.paths import ROOT

RESULTS_DIR = ROOT / "eval" / "results"


def _tabla(titulo: str, salida: dict) -> None:
    print(f"\n=== {titulo} — {salida['chunks_indexados']} chunks indexados, "
          f"top_k={salida['top_k']} ===")
    cab = f"  {'método':>8} {'carga':>7} {'fría(ms)':>10} {'cal(ms)':>9} " \
          f"{'q/s':>7} {'P@k':>6} {'memoria':>10} {'I/O':>8}"
    print(cab)
    print("  " + "-" * (len(cab) - 2))
    for r in salida["resultados"]:
        print(f"  {r['metodo']:>8} {r['carga']:>7} {r['latencia_fria_ms']:>10.2f} "
              f"{r['latencia_caliente_ms']:>9.2f} {r['throughput_qps']:>7.1f} "
              f"{r['precision_at_k']:>6.3f} {metrics.human_bytes(r['memoria_bytes']):>10} "
              f"{r['io_bloques']:>8}")


def main() -> None:
    print("### Track de TEXTO: SPIMI propio vs GIN vs GiST ###")
    texto = benchmark_text.run()

    print("\n### Track de IMAGEN: índice propio vs pgvector (HNSW / IVFFlat) ###")
    imagen = benchmark.run()

    print("\n" + "=" * 72)
    print("RESUMEN DE RESULTADOS (Fase 4)")
    print("=" * 72)
    _tabla("TEXTO", texto)
    _tabla("IMAGEN", imagen)
    print(f"\nArtefactos en {RESULTS_DIR}/:")
    print("  - text_benchmark.json / text_benchmark.png")
    print("  - benchmark.json / benchmark.png")


if __name__ == "__main__":
    main()
