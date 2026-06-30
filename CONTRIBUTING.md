# Cómo reproducir la aplicación desde cero

Este documento explica cómo un nuevo usuario puede clonar el repo, descargar los datos y tener la aplicación operativa **sin subir 100MB de archivos a GitHub**.

## Flujo general

```
1. Clonar repo (solo código + esquema)
   ↓
2. Descargar dataset de Kaggle (externo)
   ↓
3. Instalar deps + levantar BD
   ↓
4. Scrapear artículos (4-5 horas por 1000s URLs, o rápido con límite)
   ↓
5. Ejecutar setup.sh (índices: SPIMI, codebook, etc.)
   ↓
6. Levantar backend + frontend
```

## Paso 1: Clonar el repo

```bash
git clone https://github.com/marbs23/BD2-Project-2.git
cd BD2-Project-2
```

El repo contiene **solo código**: módulos, frontend, Dockerfile, esquema BD. **No contiene CSV ni imágenes** (están en `.gitignore`).

## Paso 2: Descargar el dataset de Kaggle

El dataset original (`bbc_news.csv`) viene del [Kaggle BBC News Dataset](https://www.kaggle.com/datasets/sunnysai12345/news-summary).

**Opción A: vía Kaggle CLI** (recomendado)

```bash
pip install kaggle
# Configura tu archivo ~/.kaggle/kaggle.json (obtén credenciales en tu cuenta de Kaggle)
kaggle datasets download -d sunnysai12345/news-summary
unzip news-summary.zip -d data/
mv data/bbc_news.csv data/bbc_news.csv  # asegurate que esté en data/
```

**Opción B: descarga manual**

1. Ve a https://www.kaggle.com/datasets/sunnysai12345/news-summary
2. Descarga `bbc_news.csv`
3. Colócalo en `data/bbc_news.csv`

## Paso 3: Instalar deps y levantar BD

```bash
# Crear venv (Python 3.12+)
python -m venv .venv
source .venv/bin/activate  # en Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Levantar BD en Docker (contiene el esquema)
docker compose up -d db

# Crear .env con credenciales (copia .env.example si existe, o este contenido)
cat > .env <<EOF
POSTGRES_USER=bd2user
POSTGRES_PASSWORD=bd2pass
POSTGRES_DB=bd2_multimodal
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
EOF
```

## Paso 4: Scrapear artículos

El script `data/scraper.py` extrae contenido de cada URL en `bbc_news.csv`.

**Opción A: Scrapear 200 artículos (rápido, ~5 minutos)** ← recomendado para pruebas

El scraper incluye un límite de `[:200]` URLs de forma predeterminada. Solo ejecuta:

```bash
python data/scraper.py
```

Genera `data/articulos.csv` (~300KB con 200 artículos).

**Opción B: Scrapear toda la base (lento, 4-5 horas)**

Edita `data/scraper.py` línea 96:

```python
# Cambiar esto:
pendientes = [u for u in urls if u not in procesadas][:200]

# A esto (quita el [:200]):
pendientes = [u for u in urls if u not in procesadas]
```

Luego ejecuta (es reanudable si se interrumpe):

```bash
python data/scraper.py
```

**Notas:**
- El scraper es **reanudable**: si se corta, vuelve a ejecutarse y continúa desde donde quedó.
- Pausa automática entre requests (0.3s) para no sobrecargar BBC.
- Filtra artículos con menos de 100 caracteres.

## Paso 5: Ejecutar el setup (construye índices)

Con `data/articulos.csv` ya generado, ejecuta:

```bash
./scripts/setup.sh
```

Este script hace todo de una pasada:

1. **Reinicia tablas de texto** (build limpio).
2. **Inserta documentos** → `documents` table.
3. **Split** → divide artículos en chunks por párrafos (30-word mínimo).
4. **Extractor TF-IDF** → calcula estadísticas.
5. **Codebook** → selecciona top-5000 palabras.
6. **SPIMI** → construye índice invertido + calcula normas.

Salida esperada:

```
✅ Texto operativo. Levanta la aplicación con:
   uvicorn backend.app:app --port 8000
   python -m http.server 5500 --directory frontend
```

## Paso 6: Levantar la aplicación

```bash
# Terminal 1: Backend (API REST + Swagger)
uvicorn backend.app:app --port 8000

# Terminal 2: Frontend (HTML/JS)
python -m http.server 5500 --directory frontend
```

Abre en el navegador:
- Frontend: **http://localhost:5500/index.html**
- Swagger (API docs): **http://localhost:8000/docs**
- Health check: **http://localhost:8000/health**

## Flujo alternativo: Docker Compose (sin scrapear, con data preexistente)

Si ya tienes `data/articulos.csv`, puedes levantar **toda la app en un contenedor**:

```bash
docker compose up --build
# db (Postgres) → :5432
# backend (FastAPI) → :8000
# frontend (nginx) → :5500
```

Nota: el contenedor arranca con la BD vacía; debes ejecutar los pasos 1-5 antes (o montar `data/articulos.csv` como volumen).

## Notas sobre la data

| Archivo | Tamaño | Qué es | ¿Va a Git? |
|---------|--------|--------|-----------|
| `bbc_news.csv` | ~13MB | Dataset original de Kaggle | ❌ No |
| `articulos.csv` | ~300KB–1MB | CSV scrapeado (depende del límite) | ❌ No |
| `data/raw/images/` | Variable | Imágenes descargadas (relación 1:1 con documentos) | ❌ No |
| `data/processed/spimi_blocks/` | ~50MB | Bloques temporales del SPIMI (se borra después) | ❌ No |
| Esquema BD (`db/init.sql`) | 2KB | Definición de tablas | ✅ Sí |
| Código (`text_module/`, `backend/`, etc.) | ~200KB | Módulos Python + frontend | ✅ Sí |

## Troubleshooting

**P: `python data/scraper.py` falla — "connection timeout"**
R: BBC puede bloquear requests muy seguidos. El script ya pausa 0.3s; si aún falla, incrementa la pausa en línea 117 de `data/scraper.py`.

**P: "ModuleNotFoundError: No module named 'cv2'"**
R: Normal si no necesitas la búsqueda por imagen. El backend degrada sin `opencv`/`sklearn` — solo texto funciona.

**P: BD tarda mucho en levantar**
R: PostgreSQL+pgvector es pesado. La primera vez descarga la imagen (~1GB). Paciencia.

**P: ¿Cómo borro la BD y empiezo de cero?**
R: `docker compose down -v` (borra el volumen). Luego `docker compose up -d db` levanta una vacía.

## Contacto

Si el scraper falla con un dominio específico o encuentras errores, reporta en GitHub issues.
