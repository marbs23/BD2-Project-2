"""
Corpus de evaluación a escala (fallback opcional de la Fase 4)

El corpus principal de la evaluación sale del scrape completo (scripts/build_dataset.sh):
artículos con body real, split por párrafos y categoría de article:section. Este script
es un **respaldo**: si tras el scrape no se alcanzan las cargas grandes (10K/20K chunks),
rellena la colección ingiriendo filas de data/bbc_news.csv (el feed RSS de Kaggle).

Cada fila del CSV se convierte en un documento y un único chunk de texto
(`title + ". " + description`); las descripciones son cortas, así que se insertan como
chunk directamente (sin el split por párrafos, que descartaría los textos < 30 palabras).
La categoría (proxy de relevancia) se parsea del path del enlace BBC.

⚠️  DESTRUCTIVO: reemplaza el contenido de documents / text_chunks / codebook_text /
inverted_index_text. Para volver al dataset curado de la demo, re-restaura el dump:
    docker compose down -v && docker compose up

Uso:
    python -m eval.build_eval_corpus --limit 20000
    python -m eval.build_eval_corpus              # por defecto 20000 filas
"""
import re
import sys
import argparse
import subprocess
from urllib.parse import urlparse

import pandas as pd
from psycopg2.extras import execute_values

from src.core.db import get_connection
from src.core.paths import DATA_DIR

CSV_PATH = DATA_DIR / "bbc_news.csv"
DEFAULT_LIMIT = 20000


def parse_category(link: str) -> str:
    """
    Deriva una categoría del path del enlace BBC como proxy de relevancia.

    Ejemplos:
      /news/world-europe-60638042 -> world-europe
      /news/technology-12345      -> technology
      /sport/football/12345       -> sport
    Si no se puede inferir, devuelve 'Unknown'.
    """
    try:
        segs = [s for s in urlparse(str(link)).path.split("/") if s]
    except Exception:
        return "Unknown"
    if not segs:
        return "Unknown"
    if segs[0] == "news" and len(segs) > 1:
        return re.sub(r"-\d+$", "", segs[1]) or "Unknown"
    return segs[0] or "Unknown"


def cargar_filas(csv_path, limit: int) -> pd.DataFrame:
    """Lee el CSV, limpia y deduplica por url, devolviendo hasta `limit` filas."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["title", "description"])
    df = df.drop_duplicates(subset=["guid"]).head(limit).copy()
    df["category"] = df["link"].map(parse_category)
    df["content"] = df["title"].str.strip() + ". " + df["description"].str.strip()
    return df


def poblar(df: pd.DataFrame) -> int:
    """Reinicia las tablas de texto e inserta documentos + un chunk por fila."""
    conn = get_connection()
    cur = conn.cursor()

    print("Reiniciando tablas de texto (build limpio)...")
    cur.execute("TRUNCATE inverted_index_text, codebook_text, text_chunks, documents "
                "RESTART IDENTITY CASCADE")

    doc_rows = [
        (r.guid, r.title, r.description, r.content, "", r.category)
        for r in df.itertuples(index=False)
    ]
    doc_ids = execute_values(
        cur,
        """
        INSERT INTO documents (url, title, description, body, image_url, category)
        VALUES %s RETURNING doc_id
        """,
        doc_rows,
        fetch=True,
    )

    chunk_rows = [
        (doc_id[0], content, 0, len(content.split()))
        for doc_id, content in zip(doc_ids, df["content"])
    ]
    execute_values(
        cur,
        "INSERT INTO text_chunks (doc_id, content, position, word_count) VALUES %s",
        chunk_rows,
    )

    conn.commit()
    cur.close()
    conn.close()
    return len(chunk_rows)


def reindexar() -> None:
    """Reconstruye TF-IDF, codebook y el índice invertido SPIMI reutilizando los módulos."""
    for modulo in ("src.indexing.text.extractor",
                   "src.indexing.text.codebook",
                   "src.indexing.text.spimi"):
        print(f"\n==> {modulo}")
        subprocess.run([sys.executable, "-m", modulo], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Corpus de evaluación desde bbc_news.csv (fallback)")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help=f"nº de filas a ingerir (por defecto {DEFAULT_LIMIT})")
    ap.add_argument("--csv", default=str(CSV_PATH), help="ruta del CSV de entrada")
    args = ap.parse_args()

    df = cargar_filas(args.csv, args.limit)
    n = poblar(df)
    print(f"\n✅ {n} chunks insertados (categorías proxy desde el enlace BBC).")
    reindexar()
    print(f"\n✅ Corpus de evaluación listo ({n} chunks). Ahora corre:")
    print("   ./scripts/run_benchmarks.sh")
    print("\n⚠️  Esto reemplazó el dataset de demo. Para recuperarlo: "
          "docker compose down -v && docker compose up")


if __name__ == "__main__":
    main()
