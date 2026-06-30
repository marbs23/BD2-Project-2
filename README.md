# Sistema Multimodal de Recuperación y Búsqueda
**Curso:** Base de Datos 2 — UTEC 2026-1  
**Opción implementada:** Búsqueda Multimodal en Documentos (Texto + Imagen)  
**Dataset:** [BBC News RSS Feeds — Kaggle (gpreda)](https://www.kaggle.com/datasets/gpreda/bbc-news)

---

## Descripción del sistema

Sistema de recuperación de información que indexa artículos de noticias de BBC combinando dos modalidades:

- **Texto:** pipeline TF-IDF + codebook lingüístico + índice invertido (SPIMI)
- **Imagen:** pipeline SIFT + codebook visual (K-Means) + índice invertido por histogramas

El usuario puede buscar por consulta textual o por imagen y el sistema retorna los documentos más similares combinando ambos scores.

---

## Arquitectura

```
Dataset BBC News
      ↓
  [SCRAPER]
  title, body, description, image_url, category
      ↓
  PostgreSQL → documents
      ↓
  ┌─────────────────────────────────────┐
  │         MODALIDAD TEXTO             │
  │  Split → TF-IDF → Codebook → SPIMI │
  └─────────────────────────────────────┘
  ┌─────────────────────────────────────┐
  │         MODALIDAD IMAGEN            │
  │  Split → SIFT → K-Means → Histog.  │
  └─────────────────────────────────────┘
      ↓
  [BACKEND FastAPI]
  /search/text  /search/image  /search/multimodal
      ↓
  [FRONTEND HTML/JS]
```

---

## Dataset

| Característica | Valor |
|----------------|-------|
| Fuente | BBC News RSS Feeds (Kaggle) |
| Filas originales | 42,115 |
| URLs únicas | 37,856 |
| Artículos scrapeados | ~37,000 (en proceso) |
| Columnas extraídas | url, title, description, body, image_url, category |
| Categorías | Europe, UK, Business, Technology, Science, Sport, Entertainment, entre otras |
| Tamaño promedio de body | ~750 palabras por artículo |

**Nota:** El dataset original contiene solo metadatos RSS (título, descripción corta, link). El body completo e imagen principal se obtienen mediante scraping de cada URL.

---

## Estructura del repositorio

```
BD2-Project-2/
├── docker-compose.yml
├── .env.example
├── README.md
├── db/
│   └── init.sql
├── data/
│   ├── scraper.py
│   └── insert_documents.py
├── text_module/
│   ├── split.py
│   ├── extractor.py        (TF-IDF — en desarrollo)
│   ├── codebook.py         (en desarrollo)
│   ├── spimi.py            (en desarrollo)
│   └── search.py           (en desarrollo)
├── image_module/           (en desarrollo — compañero)
├── db_module/
│   └── connection.py
├── backend/
│   └── app.py              (en desarrollo)
├── frontend/
│   └── app.py              (en desarrollo)
└── evaluation/
    └── benchmark.py        (en desarrollo)
```

---

## Instalación y uso

### Quick Start (usuarios clonadores)

```bash
# Clonar repo (solo código)
git clone https://github.com/marbs23/BD2-Project-2.git
cd BD2-Project-2

# Descargar dataset de Kaggle + scrapear + construir índices
# Ver CONTRIBUTING.md para instrucciones paso a paso
```

**👉 Lee [CONTRIBUTING.md](CONTRIBUTING.md) para el flujo completo** — incluye:
- Cómo descargar el dataset de Kaggle sin subir data a git
- Cómo scrapear artículos (rápido con 200, o lento con todo)
- Cómo ejecutar el pipeline de indexado con `./scripts/setup.sh`
- Cómo levantar la app (Docker o local)
- Troubleshooting y notas de reproducibilidad

### Requisitos

- Docker y Docker Compose
- Python 3.12+
- Pip + venv
- Acceso a Kaggle (para descargar el dataset BBC News)

### Flujo rápido (resumen)

```bash
# 1. Setup local
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. BD en Docker
docker compose up -d db

# 3. Descargar Kaggle + Scrapear (ver CONTRIBUTING.md para detalles)
python data/scraper.py                   # Crea articulos.csv

# 4. Construir índices
./scripts/setup.sh

# 5. Levantar app
uvicorn backend.app:app --port 8000      # Terminal 1: API
python -m http.server 5500 --directory frontend  # Terminal 2: UI
```

Abre **http://localhost:5500/index.html** — interfaz con 3 modos: texto, imagen y multimodal.

**Swagger (API docs):** http://localhost:8000/docs

---

## Detalles de implementación

### Módulo: Scraper (`data/scraper.py`)

Extrae de cada artículo de BBC:

| Campo | Fuente HTML |
|-------|-------------|
| `title` | `<meta property="og:title">` |
| `description` | `<meta property="og:description">` |
| `image_url` | `<meta property="og:image">` |
| `category` | `<meta property="article:section">` |
| `body` | Párrafos `<p>` dentro de `<article>` |

Características:
- Guardado incremental: cada artículo se escribe al CSV inmediatamente
- Reanudación automática por conjunto de URLs ya procesadas
- Limpieza de ruido: elimina frases de navegación, formularios de contacto y referencias externas de BBC
- Pausa de 300ms entre requests para no saturar los servidores

### Módulo: Split (`text_module/split.py`)

Divide el body de cada artículo en chunks de párrafo para su posterior indexación.

| Parámetro | Valor |
|-----------|-------|
| Separador | Doble salto de línea `\n\n` |
| Mínimo de palabras por chunk | 30 |
| Estrategia para párrafos cortos | Acumulación en buffer hasta alcanzar mínimo |

Resultados sobre muestra de 199 documentos:

| Métrica | Valor |
|---------|-------|
| Total chunks | ~3,800 |
| Promedio por documento | ~19 chunks |
| Mínimo palabras | 30 |
| Máximo palabras | 152 |

---

## Estado del proyecto

| Módulo | Estado |
|--------|--------|
| Setup Docker + pgvector | ✅ Completo |
| Scraper BBC News | ✅ Completo |
| Inserción en PostgreSQL | ✅ Completo |
| Split de texto | ✅ Completo |
| Extractor TF-IDF | ✅ Completo |
| Codebook lingüístico | ✅ Completo |
| Índice invertido SPIMI | ✅ Completo |
| Búsqueda por texto | ✅ Completo |
| Módulo imagen (SIFT + K-Means) | ✅ Completo |
| Backend FastAPI (texto/imagen/multimodal) | ✅ Completo |
| Frontend (texto/imagen/multimodal) | ✅ Completo |
| Evaluación comparativa (Fase 4) | 🔄 En curso |

---

## Equipo

| Integrante | Modalidad |
|------------|-----------|
| Martin | Texto (Split, TF-IDF, Codebook, SPIMI) |
| Marcelo | Imagen (SIFT, K-Means, Histogramas) |