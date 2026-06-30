# Sistema Multimodal de Recuperación y Búsqueda

**Curso:** Base de Datos 2 — UTEC 2026-1
**Opción implementada:** Búsqueda Multimodal en Documentos (Texto + Imagen)
**Dataset:** [BBC News RSS Feeds — Kaggle (gpreda)](https://www.kaggle.com/datasets/gpreda/bbc-news)

Sistema de recuperación de información que indexa artículos de BBC News combinando dos modalidades y
permite buscar por consulta textual, por imagen, o por una fusión de ambas:

- **Texto:** split por párrafos → TF-IDF → codebook lingüístico (top-k) → índice invertido con **SPIMI**.
- **Imagen:** patches → **SIFT** → codebook visual (**K-Means**, BoVW) → índice invertido por histogramas.

Ambas modalidades comparten el mismo modelo de espacio vectorial (ranking por similitud de coseno) y se
cruzan por `doc_id`, que une el texto y la imagen de un mismo artículo.

---

## Mapeo al checklist del proyecto

| Requisito | Modalidad | Implementación |
|-----------|-----------|----------------|
| Split | Texto / Imagen | `src/indexing/text/split.py` (párrafos) · `src/indexing/image/split.py` (patches 4×4) |
| Extractor | Texto / Imagen | TF-IDF en `src/indexing/text/extractor.py` · SIFT en `src/indexing/image/extractor.py` |
| Codebook | Texto / Imagen | top-k lingüístico en `src/indexing/text/codebook.py` · K-Means en `src/indexing/image/codebook.py` |
| Índice invertido | Texto / Imagen | **SPIMI** en `src/indexing/text/spimi.py` · histogramas en `src/indexing/image/index.py` |
| Persistencia | Compartida | PostgreSQL + pgvector (`db/schema.sql`): codebooks, histogramas, índices y metadatos |
| Búsqueda | Texto / Imagen | coseno en `src/indexing/{text,image}/search.py`; fusión en `src/api/main.py` |

---

## Estructura del repositorio

```
BD2-Project-2/
├── docker-compose.yml          # db (pgvector) + backend (FastAPI) + frontend (nginx)
├── Dockerfile                  # imagen del backend
├── requirements.txt
├── .env.example                # plantilla de configuración (sin secretos reales)
│
├── src/                        # código de la aplicación
│   ├── api/main.py             # API REST (endpoints de búsqueda)
│   ├── core/
│   │   ├── db.py               # conexión a PostgreSQL
│   │   └── paths.py            # rutas de datos ancladas a la raíz del repo
│   └── indexing/
│       ├── text/               # split, extractor, codebook, spimi, search
│       └── image/              # split, extractor, codebook, index, search, pgvec
│
├── pipeline/                   # SOLO maintainer: generación de datos
│   ├── scrape_text.py          # bbc_news.csv -> articulos.csv
│   ├── scrape_images.py        # documents.image_url -> data/raw/images
│   └── ingest_documents.py     # articulos.csv -> tabla documents
│
├── db/
│   ├── schema.sql              # esquema (7 tablas)
│   ├── restore.sh              # restaura el dump en el primer arranque
│   └── seed/                   # aquí va dump.sql.gz (descargado del Drive)
│
├── scripts/
│   ├── setup.sh                # construye el índice de texto desde el CSV
│   └── build_dataset.sh        # scrape + index + pg_dump (regenera el dump)
│
├── frontend/index.html         # SPA: modos texto / imagen / multimodal
├── eval/                       # benchmark del índice propio vs pgvector
└── tests/                      # pytest
```

---

## Puesta en marcha (usuario que consume)

El que clona el repo **no scrapea ni indexa**: la base de datos viene poblada en un dump que se descarga
aparte (pesa demasiado para git) y se restaura sola en el primer arranque del contenedor.

```bash
# 1. Clonar
git clone https://github.com/marbs23/BD2-Project-2.git
cd BD2-Project-2

# 2. Configurar entorno
cp .env.example .env            # rellena POSTGRES_USER / POSTGRES_PASSWORD

# 3. Descargar el dump de la BD y colocarlo en db/seed/
#    Link: <PEGAR_AQUÍ_EL_LINK_DE_GOOGLE_DRIVE>
#    -> db/seed/dump.sql.gz

# 4. Levantar todo
docker compose up --build
```

- Frontend: **http://localhost:5500**
- API (Swagger): **http://localhost:8000/docs**
- Health: **http://localhost:8000/health**

En el primer arranque Postgres ejecuta `db/schema.sql` y luego `db/restore.sh`, que carga
`db/seed/dump.sql.gz`. Para reimportar desde cero: `docker compose down -v && docker compose up`.

> Sin el dump en `db/seed/`, la app igual levanta pero con la BD vacía (solo el esquema).

---

## Regenerar los datos (maintainer)

Solo para quien mantiene el dataset. Requiere el dataset de Kaggle en `data/bbc_news.csv` y la BD
levantada (`docker compose up -d db`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

./scripts/build_dataset.sh      # scrape -> index (texto+imagen) -> db/seed/dump.sql.gz
```

Luego sube `db/seed/dump.sql.gz` a Google Drive y actualiza el link en este README.

El scraper es **reanudable** (continúa donde quedó) y pausa entre requests para no saturar BBC.
`scripts/setup.sh` reconstruye solo el índice de texto si necesitas iterar sobre esa parte.

---

## API

| Endpoint | Método | Parámetros |
|----------|--------|-----------|
| `/health` | GET | — |
| `/search/text` | POST | `query`, `top_n` |
| `/search/image` | POST | `file` (multipart), `top_n` |
| `/search/multimodal` | POST | `query`, `file` (opc.), `alpha`, `top_n` |

`alpha` pondera el texto y `(1-alpha)` la imagen en la fusión multimodal.

---

## Tests

```bash
pytest
```

---

## Equipo

| Integrante | Modalidad |
|------------|-----------|
| Martin | Texto (split, TF-IDF, codebook, SPIMI) |
| Marcelo | Imagen (SIFT, K-Means, histogramas) |
