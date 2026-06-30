#!/usr/bin/env bash
#
# build_dataset.sh — regenera TODOS los datos y produce el dump distribuible (maintainer).
#
# Este script NO lo corre quien clona el repo. Lo corre quien mantiene los datos para
# (re)generar el dump que luego se sube al Drive. Pasos:
#   1. scrape de artículos        -> data/articulos.csv
#   2. pipeline de texto          -> documents/chunks/codebook/índice SPIMI
#   3. scrape de imágenes         -> data/raw/images/{doc_id}.jpg
#   4. pipeline de imagen         -> SIFT -> K-Means -> índice invertido visual
#   5. pg_dump (data-only)        -> db/seed/dump.sql.gz
#
# Requiere la BD levantada:  docker compose up -d db
# Requiere el dataset Kaggle en data/bbc_news.csv.
#
set -euo pipefail
cd "$(dirname "$0")/.."   # raíz del proyecto

PY="${PYTHON:-python}"

echo "==> 1/3  Scraping de artículos (data/bbc_news.csv -> data/articulos.csv)..."
$PY -m pipeline.scrape_text

echo "==> 2/3  Índice de texto (insert -> chunks -> TF-IDF -> codebook -> SPIMI)..."
./scripts/setup.sh

# --- Pipeline de IMAGEN desactivado por ahora ---------------------------------
# Descomenta este bloque cuando quieras incluir la modalidad de imagen en el dump.
# echo "==> Descarga de imágenes (documents.image_url -> data/raw/images)..."
# $PY -m pipeline.scrape_images
#
# echo "==> Pipeline de imagen (SIFT -> K-Means -> índice invertido)..."
# $PY -m src.indexing.image.extractor
# $PY -m src.indexing.image.codebook
# $PY -m src.indexing.image.index
# ------------------------------------------------------------------------------

echo "==> 3/3  Generando dump distribuible (db/seed/dump.sql.gz)..."
mkdir -p db/seed
docker exec bd2_postgres pg_dump --data-only --no-owner \
    -U "${POSTGRES_USER:-bd2user}" -d "${POSTGRES_DB:-bd2_multimodal}" \
    | gzip > db/seed/dump.sql.gz

echo
echo "✅ Listo. Sube db/seed/dump.sql.gz al Drive y actualiza el link en el README."
