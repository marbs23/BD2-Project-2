# Plan de trabajo — Pipeline de Imagen (Alejandro)

> **Proyecto 2 BD2 — Sistema Multimodal de Recuperación y Búsqueda (App 3: Búsqueda Multimodal en Documentos)**
> Dataset: BBC News. Modalidades: **texto (Martín)** + **imagen (Alejandro)**.
> Este documento es la guía de la rama de imagen. Cada tarea sale como **PR que Martín revisa y mergea**.

---

## 0. Resumen en una frase

Construir un buscador de imágenes de noticias usando **Bag of Visual Words (BoVW)**:
una imagen → se parte en patches → SIFT saca descriptores → K-Means arma un "diccionario visual" →
cada imagen se representa como un histograma de palabras visuales → se indexa y se busca por similitud.

La arquitectura de imagen es un **espejo exacto** de la de texto que ya hizo Martín. Esa simetría
es justo lo que el enunciado llama *"arquitectura unificada agnóstica a la modalidad"*.

```
TEXTO (hecho) IMAGEN (por hacer)
───────────────── ─────────────────────────
documento imagen de un documento
 → split en párrafos → split en patches
 → tokens/términos → descriptores SIFT
 → codebook (top-k) → codebook K-Means (k visual words)
 → histograma TF-IDF → histograma TF-IDF de visual words
 → índice invertido → índice invertido
 → búsqueda coseno → búsqueda coseno
 → dedup por documento → dedup por documento
```

---

## 1. Estado actual (auditado)

**Ya en `main` (Martín):**
- `data/scraper.py` — scrapea artículos BBC: `title, description, body, image_url, category`. **Solo guarda la URL de la imagen, no la descarga.** Tope actual `[:200]` → solo hay **199 artículos**.
- `data/insert_documents.py` — carga el CSV a la tabla `documents`.
- `text_module/` — pipeline de texto **completo y funcionando**: `split`, `extractor` (TF-IDF), `codebook` (top-k), `spimi` (índice invertido), `search` (coseno).
- `db/init.sql`, `db_module/connection.py`, `docker-compose.yml`, `.env.example`.

**Ya en `main` (Alejandro, issue #2):**
- `src/image/scraper.py` — descarga imágenes a disco. **Funciona** (19/19 artículos en prueba).

**Lo que falta:** todo el pipeline de imagen (#7 → #26).

---

## 2. Decisiones de diseño (LEER — esto define todo)

### D1 — Un "chunk" de imagen es un **patch**, no la imagen entera
Igual que en texto un chunk es un párrafo (hay varios por documento), en imagen un chunk es un patch
(hay varios por imagen). Así:
- la arquitectura queda simétrica con texto,
- llegamos a los **100K chunks** del benchmark de forma natural: `~6,250 imágenes × 16 patches = 100K`,
- la búsqueda recupera patches y **deduplica por documento** (idéntico a lo que ya hace `search.py` de texto).

### D2 — Histograma de visual words con **TF-IDF** (no conteo crudo)
BoVW + TF-IDF sobre visual words es estándar (Sivic & Zisserman, *Video Google*). Permite **reutilizar
literalmente** la lógica de coseno e índice invertido de texto. Mismo `inverted_index_*`, misma fórmula.

### D3 — La imagen por documento es la **hero image** (`og:image`), 1:1
Es la que Martín ya guarda en `documents.image_url`. Una imagen por documento mantiene el cruce
texto↔imagen limpio para la búsqueda multimodal (App 3). El split en patches genera la multiplicidad.

### D4 — Carpeta `image_module/` (no `src/image/`)
Para que sea simétrico con `text_module/`. En el Sprint 0 se migra el scraper actual.

### D5 — Tamaño del codebook `k = 512` (configurable)
Rango típico BoVW: 256–1024. Con 100K chunks, 512 es buen punto de partida. Se deja como constante.

### D6 — Split en grid **4×4 con solapamiento** sobre imagen redimensionada (lado mayor ~512px)
4×4 = 16 patches. El solapamiento evita perder keypoints en los bordes. Patches sin keypoints SIFT
(zonas planas: cielo, fondo liso) se descartan y no se indexan.

---

## 3. Esquema de base de datos (objetivo)

El `image_chunks` actual tiene `descriptor vector(128)` — **no sirve** para el enfoque índice invertido
+ histograma. Se rediseña espejo del de texto:

```sql
-- un chunk = un patch
CREATE TABLE image_chunks (
 chunk_id SERIAL PRIMARY KEY,
 doc_id INTEGER REFERENCES documents(doc_id),
 patch_index INTEGER, -- 0..15 dentro de la imagen
 norm FLOAT, -- ||d|| del histograma TF-IDF, precalculada
 histogram vector(512) -- para la comparativa pgvector (HNSW/IVF)
);

-- diccionario visual: k centroides de 128 dimensiones
CREATE TABLE codebook_image (
 word_id SERIAL PRIMARY KEY,
 centroid vector(128)
);

-- índice invertido propio (espejo de inverted_index_text)
CREATE TABLE inverted_index_image (
 word_id INTEGER REFERENCES codebook_image(word_id),
 chunk_id INTEGER REFERENCES image_chunks(chunk_id),
 tf_idf FLOAT,
 PRIMARY KEY (word_id, chunk_id)
);
```

> Toca `db/init.sql` (archivo de Martín) → **coordinar antes de mergear**.

---

## 4. Convenciones (copiadas del estilo de Martín, para que los PR pasen review)

- Módulos en `image_module/`, un archivo por etapa.
- Cada módulo: docstring que cita el enunciado + nº de issue, type hints, `if __name__ == "__main__"`
 con verificación real contra la BD, comentarios en **español**.
- Conexión siempre por `db_module.connection.get_connection()`.
- Caches y artefactos pesados en `data/processed/` (gitignored): descriptores `.npy`, codebook, etc.
- Commits chicos en español: `feat(image): extracción SIFT por patch`.
- **Una rama por issue** (GitHub la crea como `<nº>-<slug>` desde el issue) → PR a `main` → review de Martín.
- Tests con `pytest` en `tests/`. Un sprint no se cierra sin sus tests en verde.

---

## 5. Sprints

> Cada sprint termina con **goal cumplido + tests en verde** antes de pasar al siguiente.
> La cadena es casi lineal (cada paso necesita el anterior).

### Sprint 0 — Cimientos: datos + schema
**Goal:** base poblada con ~6–7K documentos (texto + imagen descargada) y schema de imagen listo.

- Escalar el scraping de artículos a ~7K (subir el tope `[:200]` en `data/scraper.py`) → `documents`.
- Migrar `src/image/scraper.py` → `image_module/scraper.py` y ajustarlo para bajar la **hero image**
 (`documents.image_url`) 1:1 por documento, guardando con el `doc_id`.
- Extender `db/init.sql` con `codebook_image`, `inverted_index_image` y el `image_chunks` rediseñado.

** Done:** `docker compose up` limpio · `SELECT count(*) FROM documents > 6000` · nº de imágenes en
disco == nº de docs con imagen · las 3 tablas existen.
**Issue:** nuevo — *"infra: schema de imagen + carga de datos"*. **Coordinar con Martín (toca su scraper e init.sql).**

---

### Sprint 1 — Extracción: patches + SIFT (#7, #8)
**Goal:** de cada imagen → 16 patches → descriptores SIFT, cacheados en disco.

- `image_module/split.py` (#7): redimensiona, parte en grid 4×4 con solape → patches.
- `image_module/extractor.py` (#8): `cv2.SIFT_create().detectAndCompute()` por patch → array `(N,128)`.
 Persiste todos los descriptores en `data/processed/descriptors.npy` (+ índice patch→doc).

** Done:** test de split (1 imagen → 16 patches, dimensiones correctas) · SIFT devuelve `(N,128)` ·
patches vacíos manejados sin crash · descriptores se guardan y recargan · `pytest` verde.
**Ramas:** `7-img-split-patches`, `8-img-extractor-sift`.

---

### Sprint 2 — Vocabulario visual + índice (#9, #10)
**Goal:** codebook entrenado + histograma TF-IDF por patch + índice invertido propio, todo en Postgres.

- `image_module/codebook.py` (#9): `MiniBatchKMeans(n_clusters=512)` sobre todos los descriptores →
 centroides → `codebook_image`. (Equivale al "codebook top-k" de texto.)
- `image_module/index.py` (#10): por patch, asigna cada descriptor a su visual word más cercano →
 histograma → TF-IDF → persiste en `inverted_index_image` + `image_chunks.histogram` +
 `image_chunks.norm`. (Mismo índice invertido y TF-IDF que texto; sin SPIMI por
 bloques porque el vocabulario visual es fijo y el histograma denso.)

** Done:** `codebook_image` tiene 512 filas · cada patch indexado tiene histograma dim 512 ·
**sanity check: un patch se recupera a sí mismo con similitud ≈ 1.0** · índice invertido no vacío.
**Ramas:** `9-img-codebook-kmeans`, `10-img-indice-invertido`.
> El K-Means corre largo (millones de descriptores). Correr **overnight**, no el último día.

---

### Sprint 3 — Búsqueda + comparativa pgvector (#15, #24)
**Goal:** búsqueda por imagen funcionando con **índice propio** y con **pgvector**.

- `image_module/search.py` (#15): query imagen → patches → SIFT → histograma → coseno sobre
 `inverted_index_image` → top-N → **dedup por documento**. (Espejo de `search.py` de texto.)
- `image_module/pgvector_bench.py` (#24): mismos histogramas como `vector(512)`, índices **HNSW** e
 **IVFFlat**, búsqueda KNN. Medir con `EXPLAIN (ANALYZE, BUFFERS)`.

** Done:** la query devuelve top-N coherente (misma categoría BBC en los primeros puestos) ·
índice propio vs pgvector dan resultados comparables · latencia básica medida.
**Ramas:** `15-img-busqueda-similitud`, `24-postgres-pgvector-hnsw`.

---

### Sprint 4 — Backend + Frontend (#17, #25)
**Goal:** demo viva end-to-end.

- `backend/` (#17): FastAPI con `POST /search/text` (usa `text_module.search.buscar`),
 `POST /search/image` (usa `image_module.search`), y fusión `score = α·texto + (1−α)·imagen` por `doc_id`.
- `frontend/` (#25): formulario de texto + upload de imagen → muestra top-N con título, categoría y thumbnail.

** Done:** `curl` a los endpoints devuelve JSON · subir una imagen real devuelve resultados ·
la UI carga y muestra · demo end-to-end funciona.
**Ramas:** `17-backend-api-rest`, `25-frontend-ui`.

---

### Sprint 5 — Evaluación + informe (#26)
**Goal:** benchmark completo con gráficos para el informe.

- `eval/benchmark.py` (#26): latencia, throughput, precisión@10, memoria e I/O en **1K / 10K / 100K**
 chunks × **3 métodos** (índice propio, HNSW, IVFFlat). Caché fría y caliente. Semillas fijas.
- Gráficos comparativos + sección de imagen en el README/Wiki.

** Done:** tablas y gráficos para las 3 cargas · precisión con ground-truth = misma categoría BBC ·
reproducible con seed · sección de informe redactada.
**Rama:** `26-eval-benchmarking`.

---

## 6. Cadena de dependencias

```
Sprint 0 ──> 1 ──> 2 ──> 3 ──> 4 ──> 5
 datos SIFT codebook búsqueda demo eval
 (#24 ∥ #15)
```
Único paralelizable: #24 (pgvector) junto a #15. Todo lo demás es secuencial.

---

## 7. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Scraping de 7K artículos lento / bloqueos BBC | resume incremental (ya lo tiene el scraper), delay ≥0.3s, correr por bloques |
| Patches sin keypoints (cielo, fondos planos) | descartarlos; grid con solape; redimensionar a 512px |
| K-Means tarda horas | `MiniBatchKMeans`, correr overnight, cachear centroides |
| 100K chunks en texto difícil (descripciones cortas) | es de Martín; coordinar si la carga "grande" se reduce |
| Schema toca archivo de Martín | coordinar antes de cada PR que modifique `db/init.sql` o `data/scraper.py` |

---

## 8. Definición de "segunda aplicación"

El enunciado pide **al menos dos aplicaciones**. App 3 se entrega como dos modos de búsqueda sobre el
mismo backend: **(a) búsqueda por texto** → artículos relevantes, **(b) búsqueda por imagen** → artículos
con imágenes similares. Confirmar con el profe/equipo que esto cuenta como dos, o sumar una tercera vista
(p. ej. búsqueda multimodal combinada) que ya sale gratis de la fusión del backend (#17).
