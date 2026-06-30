"""
API REST de búsqueda (issue #17)

Expone las dos aplicaciones sobre el mismo motor:
  - POST /search/text   : búsqueda full-text (índice invertido de texto).
  - POST /search/image  : búsqueda por imagen (índice invertido visual).
  - POST /search/multimodal : fusiona ambos scores -> alpha*texto + (1-alpha)*imagen.

Los resultados de las dos modalidades se cruzan por doc_id, que es lo que une el
texto y la imagen de un mismo artículo.
"""
import os
import sys
import tempfile
from collections import defaultdict

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_module.connection import get_connection
from text_module import search as text_search
from image_module import search as image_search

app = FastAPI(title="BD2 - Búsqueda multimodal")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _enrich(doc_ids: list[int]) -> dict[int, dict]:
    """Metadata de documentos por doc_id, para uniformar la respuesta."""
    if not doc_ids:
        return {}
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT doc_id, title, url, category, image_url FROM documents WHERE doc_id = ANY(%s)",
        (doc_ids,),
    )
    meta = {
        doc_id: {"doc_id": doc_id, "title": title, "url": url,
                 "category": category, "image_url": image_url}
        for doc_id, title, url, category, image_url in cur.fetchall()
    }
    cur.close()
    conn.close()
    return meta


def _normalize(results: list[dict]) -> dict[int, float]:
    """Score por doc_id reescalado a [0,1] por el máximo, para poder fusionar."""
    by_doc = {r["doc_id"]: r["score"] for r in results}
    if not by_doc:
        return {}
    top = max(by_doc.values())
    return {d: s / top for d, s in by_doc.items()} if top > 0 else by_doc


async def _buscar_imagen(file: UploadFile, top_n: int) -> list[dict]:
    """Guarda el upload en un temporal (buscar lee por ruta) y lo borra siempre."""
    suffix = os.path.splitext(file.filename or "q.jpg")[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(await file.read())
        tmp.close()
        return image_search.buscar(tmp.name, top_n=top_n)
    finally:
        os.unlink(tmp.name)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search/text")
def search_text(query: str = Form(...), top_n: int = Form(10)):
    return {"results": text_search.buscar(query, top_n=top_n)}


@app.post("/search/image")
async def search_image(file: UploadFile = File(...), top_n: int = Form(10)):
    return {"results": await _buscar_imagen(file, top_n)}


@app.post("/search/multimodal")
async def search_multimodal(
    query: str = Form(""),
    file: UploadFile | None = File(None),
    alpha: float = Form(0.5),
    top_n: int = Form(10),
):
    """Combina texto e imagen. alpha pondera el texto; (1-alpha) la imagen."""
    text_scores: dict[int, float] = {}
    image_scores: dict[int, float] = {}

    if query.strip():
        text_scores = _normalize(text_search.buscar(query, top_n=top_n * 3))

    if file is not None:
        image_scores = _normalize(await _buscar_imagen(file, top_n * 3))

    combined: dict[int, float] = defaultdict(float)
    for doc_id, s in text_scores.items():
        combined[doc_id] += alpha * s
    for doc_id, s in image_scores.items():
        combined[doc_id] += (1 - alpha) * s

    meta = _enrich(list(combined))
    ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    results = [{**meta[doc_id], "score": score} for doc_id, score in ranked if doc_id in meta]
    return {"results": results}
