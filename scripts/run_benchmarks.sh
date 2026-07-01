#!/usr/bin/env bash
#
# run_benchmarks.sh — corre TODOS los benchmarks de la Fase 4 con un solo comando.
#
# Ejecuta los dos tracks de evaluación experimental y genera los artefactos del
# informe (JSON + gráficos) en eval/results/:
#   - Texto : índice invertido propio (SPIMI) vs GIN vs GiST.
#   - Imagen: índice invertido propio vs pgvector (HNSW / IVFFlat).
# Al final imprime una tabla resumen con las cinco métricas por método y carga.
#
# Uso:
#   ./scripts/run_benchmarks.sh
#   PYTHON=.venv/bin/python ./scripts/run_benchmarks.sh   # si no tienes el venv activado
#
# Requiere la BD levantada y poblada:  docker compose up -d db
#
set -euo pipefail
cd "$(dirname "$0")/.."   # raíz del proyecto

PY="${PYTHON:-python}"

echo "==> Corriendo benchmarks (texto + imagen)..."
$PY -m eval.run_all

echo
echo "✅ Listo. Resultados y gráficos en eval/results/"
echo "   - text_benchmark.json / text_benchmark.png   (SPIMI vs GIN vs GiST)"
echo "   - benchmark.json / benchmark.png             (propio vs pgvector)"
