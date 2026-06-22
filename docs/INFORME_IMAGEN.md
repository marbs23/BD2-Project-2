# Informe - Pipeline de Imagen (Bag of Visual Words)

Sección de imagen del sistema multimodal. Sigue la arquitectura unificada del
proyecto, `split → extractor → codebook → índice invertido`, como un espejo del
pipeline de texto: lo que ahí es una palabra, aquí es una *palabra visual*.

## Arquitectura

```
imagen del documento
 → split : 4x4 = 16 patches solapados (image_module/split.py)
 → extractor : descriptores SIFT de 128 dims por patch (image_module/extractor.py)
 → codebook : MiniBatchKMeans, k=512 palabras visuales (image_module/codebook.py)
 → índice : histograma TF-IDF de palabras visuales por patch (image_module/index.py)
 → búsqueda : coseno sobre el índice invertido + dedup por documento (image_module/search.py)
```

Un **chunk** de imagen es un **patch**, igual que un chunk de texto es un párrafo:
hay varios por documento. Esto hace que el conteo de chunks crezca de forma natural
hacia las cargas de evaluación y mantiene la simetría con el pipeline de texto.

## Detalle por módulo

| Módulo | Rol | Equivalente en texto |
|---|---|---|
| `split.py` | divide la imagen en patches | split en párrafos |
| `extractor.py` | descriptores SIFT por patch | TF-IDF / tokenización |
| `codebook.py` | K-Means → palabras visuales | top-k palabras |
| `index.py` | histograma TF-IDF + índice invertido | índice invertido / TF-IDF (sin SPIMI) |
| `search.py` | coseno + dedup por documento | búsqueda por coseno |
| `pgvector_bench.py` | baseline HNSW / IVFFlat | GIN / GiST |

El índice de imagen usa el mismo modelo de índice invertido y la misma ponderación
TF-IDF que el de texto, pero no aplica SPIMI por bloques: el vocabulario visual es
fijo (k) y el histograma es denso, de modo que el spill a disco y el merge externo
no aportan.

La ponderación es la misma del curso: `tf_w = 1 + log10(tf)`, `idf = log10(N/df)`,
`peso = tf_w · idf`. La similitud es de coseno, con la norma del chunk precalculada
al indexar (`image_chunks.norm`), igual que en texto.

## Persistencia

- `codebook_image(word_id, centroid vector(128))` - las k palabras visuales.
- `image_chunks(chunk_id, doc_id, patch_index, norm, histogram vector(512))` - un patch
 por fila; el histograma denso alimenta la comparativa pgvector.
- `inverted_index_image(word_id, chunk_id, tf_idf)` - las posting lists del método propio.

## Comparativa: índice propio vs pgvector

Los mismos histogramas se consultan de dos formas: con las posting lists propias y
con los índices vectoriales nativos de PostgreSQL (HNSW e IVFFlat), ambos con
distancia de coseno para que la comparación sea justa.

### Resultados (carga 1K, colección de desarrollo: 3.035 patches, 290.066 descriptores, k=512)

| Método | Latencia fría | Latencia caliente | Throughput | Precisión@10 |
|---|---|---|---|---|
| Índice propio | 194.6 ms | 108.7 ms | 9.2 q/s | 0.089 |
| pgvector HNSW | 74.0 ms | 64.3 ms | 15.5 q/s | 0.089 |
| pgvector IVFFlat | 56.3 ms | 64.3 ms | 15.6 q/s | 0.089 |

> Métrica de relevancia: misma categoría de artículo BBC (proxy de ground-truth).
> Consultas: 30 imágenes muestreadas con semilla fija. Caché fría y caliente.
> Cada baseline pgvector se mide con su índice aislado (si HNSW e IVFFlat coexisten,
> el planner elegiría uno solo y ambas filas medirían lo mismo).

### Análisis de trade-offs

- **pgvector es ~1.7× más rápido** que el índice propio: la búsqueda aproximada
 evita recorrer y puntuar todas las posting lists candidatas.
- **La precisión@10 empata** a esta escala: con 1K chunks y categoría como
 ground-truth, los tres métodos recuperan vecinos de calidad equivalente; la
 diferencia entre HNSW e IVFFlat aparece sobre todo en latencia.
- Es el trade-off **exactitud/latencia** que pide el enunciado: el método propio
 hace el coseno exacto sobre el índice invertido; pgvector gana en velocidad.
- A escala de desarrollo (3.035 chunks) las cargas 10K y 100K no son alcanzables;
 el benchmark las omite con aviso y corre solo la carga 1K real.

## Reproducir

```bash
docker compose up -d
pip install -r requirements.txt

python data/insert_documents.py # documentos -> tabla documents
python -m image_module.scraper # descarga las imágenes hero
python -m image_module.extractor # SIFT -> data/processed/descriptors.npy
python -m image_module.codebook # K-Means -> codebook_image
python -m image_module.index # histogramas + inverted_index_image
python -m eval.benchmark # comparativa -> eval/results/
```

## Escala

La colección de desarrollo tiene 3.035 patches, suficiente para la carga pequeña
(1K). Para las cargas mediana (10K) y grande (100K) se amplía el scraping de
artículos; el código es agnóstico al número de documentos y el benchmark omite con
aviso las cargas que la colección no alcanza (sin recortes silenciosos).
