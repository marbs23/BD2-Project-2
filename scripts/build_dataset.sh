#!/usr/bin/env bash
#
# build_dataset.sh — regenera los datos y produce el dump distribuible (maintainer).
#
# Este script NO lo corre quien clona el repo. Lo corre quien mantiene los datos para
# (re)generar el dump que luego se sube al Drive. Pasos:
#   1. insert de documentos desde data/articulos.csv
#   2. pipeline de texto   -> split -> TF-IDF -> codebook -> SPIMI
#   3. pipeline de imagen  -> split -> SIFT -> K-Means -> índice invertido
#   4. pg_dump             -> db/seed/dump.sql.gz
#
# Requiere la BD levantada:  docker compose up -d db
# Requiere data/articulos.csv ya generado (scraping de texto corrido aparte).
# Requiere data/raw/images/{doc_id}.jpg ya descargadas
#   (scraping de imágenes corrido aparte: python -m pipeline.scrape_images).
#
set -euo pipefail
cd "$(dirname "$0")/.."   # raíz del proyecto

PY="${PYTHON:-python}"

# Cargar .env para que bash (el pg_dump del paso 4) vea POSTGRES_USER/DB, igual que
# Python los lee con load_dotenv(). Sin esto, `set -u` aborta con "unbound variable".
if [ -f .env ]; then
  set -a           # exporta todo lo que se defina a continuación
  . ./.env
  set +a
fi

# Nº de imágenes a indexar. SIFT sobre las decenas de miles de imágenes del scrape
# no cabe en RAM; para la evaluación basta un subconjunto (ver README, Fase 4).
# Overridable:  IMAGE_LIMIT=3000 ./scripts/build_dataset.sh
IMAGE_LIMIT="${IMAGE_LIMIT:-5000}"

echo "==> 0/4  Arranque limpio (cache de disco + tablas de la BD)..."
rm -rf data/processed          # descriptors/corpus_stats/spimi_blocks (regenerables)
$PY - <<'PYEOF'
from src.core.db import get_connection
conn = get_connection(); cur = conn.cursor()
cur.execute("""
    TRUNCATE inverted_index_text, inverted_index_image,
             text_chunks, image_chunks,
             codebook_text, codebook_image, documents
    RESTART IDENTITY CASCADE
""")
conn.commit(); cur.close(); conn.close()
print("   tablas reiniciadas")
PYEOF

echo "==> 1/4  Insertando documentos desde articulos.csv..."
$PY -m pipeline.ingest_documents

echo "==> 2/4  Pipeline de TEXTO (split -> TF-IDF -> codebook -> SPIMI)..."
$PY -m src.indexing.text.split       # párrafos
$PY -m src.indexing.text.extractor   # TF-IDF (cachea estadísticas)
$PY -m src.indexing.text.codebook    # top-5000 términos
$PY -m src.indexing.text.spimi       # índice invertido

echo "==> 3/4  Pipeline de IMAGEN (tope ${IMAGE_LIMIT} imágenes: SIFT -> K-Means -> índice)..."
IMAGE_LIMIT="$IMAGE_LIMIT" $PY -m src.indexing.image.extractor  # SIFT (cachea descriptores)
$PY -m src.indexing.image.codebook   # K-Means 512 centroides
$PY -m src.indexing.image.index      # histogramas + índice

echo "==> 4/4  Generando dump distribuible (db/seed/dump.sql.gz)..."
mkdir -p db/seed
docker compose exec -T db pg_dump --username="${POSTGRES_USER}" --dbname="${POSTGRES_DB}" \
  | gzip > db/seed/dump.sql.gz

echo
echo "✅ Listo. Sube db/seed/dump.sql.gz al Drive y actualiza el link en el README."