#!/usr/bin/env bash
#
# setup.sh — deja la aplicación operativa a partir del CSV ya scrapeado.
#
# NO incluye scraping (cada integrante scrapea su parte por separado). Parte de
# data/articulos.csv ya generado y construye el pipeline de TEXTO de punta a punta
# sobre la base de datos: documentos -> chunks -> TF-IDF -> codebook -> índice SPIMI.
#
# Es idempotente: reinicia las tablas de texto antes de reconstruir, así se puede
# correr varias veces sin duplicar chunks.
#
# Uso:
#   ./scripts/setup.sh
#   PYTHON=.venv/bin/python ./scripts/setup.sh    # si no tienes el venv activado
#
# Requiere la BD levantada:  docker compose up -d db
#
set -euo pipefail
cd "$(dirname "$0")/.."   # raíz del proyecto

PY="${PYTHON:-python}"

echo "==> 0/5  Reiniciando tablas de texto (build limpio)..."
$PY - <<'PYEOF'
from db_module.connection import get_connection
conn = get_connection(); cur = conn.cursor()
cur.execute("TRUNCATE inverted_index_text, codebook_text, text_chunks RESTART IDENTITY CASCADE")
conn.commit(); cur.close(); conn.close()
print("   text_chunks / codebook_text / inverted_index_text reiniciadas")
PYEOF

echo "==> 1/5  Insertando documentos (data/articulos.csv -> documents)..."
$PY data/insert_documents.py

echo "==> 2/5  Split de artículos en chunks..."
$PY text_module/split.py

echo "==> 3/5  Extractor TF-IDF (estadísticas del corpus)..."
$PY text_module/extractor.py

echo "==> 4/5  Codebook top-k..."
$PY text_module/codebook.py

echo "==> 5/5  Índice invertido SPIMI (+ normas por chunk)..."
$PY text_module/spimi.py

echo
echo "✅ Texto operativo. Levanta la aplicación con:"
echo "   uvicorn backend.app:app --port 8000                 # API + Swagger en /docs"
echo "   python -m http.server 5500 --directory frontend     # frontend"
echo
echo "Nota: la búsqueda por imagen es del módulo del compañero (requiere sus imágenes,"
echo "      opencv/scikit-learn y su pipeline de indexado)."
