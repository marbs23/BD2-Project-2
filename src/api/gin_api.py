"""
Endpoint de búsqueda full-text con índice GIN (issue #23, Fase 3)

App FastAPI mínima y autocontenida para demostrar la búsqueda full-text nativa de
PostgreSQL. Vive en un archivo aparte para no chocar con backend/app.py (que está
en la rama del backend, aún sin mergear); al integrar el backend, esta ruta
`/search/fulltext` se mueve a app.py.

Levantar:  uvicorn backend.gin_api:app --reload
Probar:    curl -X POST localhost:8000/search/fulltext -H "Content-Type: application/json" \
                -d '{"query": "russia ukraine war", "top_n": 5}'
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel

from text_module.gin_search import buscar_fulltext

app = FastAPI(title="BD2 — Full-text GIN (comparación PostgreSQL)", version="0.1")


class FullTextQuery(BaseModel):
    query: str
    top_n: int = 10


class SearchResult(BaseModel):
    doc_id: int
    chunk_id: int
    title: str | None = None
    url: str | None = None      # link al artículo para redirigir al usuario
    score: float
    snippet: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search/fulltext", response_model=list[SearchResult])
def search_fulltext(q: FullTextQuery):
    """Búsqueda full-text nativa de PostgreSQL usando el índice GIN."""
    return buscar_fulltext(q.query, top_n=q.top_n)
