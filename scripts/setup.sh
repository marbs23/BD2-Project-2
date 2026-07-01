#!/usr/bin/env bash
#
# setup.sh — construye el índice de TEXTO a partir de data/articulos.csv (maintainer).
#
# NO incluye scraping. Parte de data/articulos.csv ya generado y construye el
# pipeline de texto de punta a punta sobre la base de datos:
#   documents -> chunks -> TF-IDF -> codebook -> índice SPIMI.
#
# Es idempotente: reinicia las tablas de texto antes de reconstruir.
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
from src.core.db import get_connection
conn = get_connection(); cur = conn.cursor()
cur.execute("TRUNCATE inverted_index_text, codebook_text, text_chunks RESTART IDENTITY CASCADE")
conn.commit(); cur.close(); conn.close()
print("   text_chunks / codebook_text / inverted_index_text reiniciadas")
PYEOF

echo "==> 1/5  Insertando documentos (data/articulos.csv -> documents)..."
$PY -m pipeline.ingest_documents

echo "==> 2/5  Split de artículos en chunks..."
$PY -m src.indexing.text.split

echo "==> 3/5  Extractor TF-IDF (estadísticas del corpus)..."
$PY -m src.indexing.text.extractor

echo "==> 4/5  Codebook top-k..."
$PY -m src.indexing.text.codebook

echo "==> 5/5  Índice invertido SPIMI (+ normas por chunk)..."
$PY -m src.indexing.text.spimi

echo
echo "✅ Texto operativo. Levanta la aplicación con:"
echo "   docker compose up        # API + frontend + BD"
echo
echo "Nota: la búsqueda por imagen requiere su pipeline de indexado "
echo "      (pipeline/scrape_images.py + src/indexing/image/*)."
