"""
Scraper de artículos (maintainer)

Lee el dataset original de Kaggle (data/bbc_news.csv), descarga el HTML de cada
URL y extrae título, descripción, cuerpo, imagen y categoría a data/articulos.csv.
Guarda de forma incremental y reanuda si se interrumpe.

Corre en paralelo (ThreadPoolExecutor) con un pequeño sleep por worker y backoff
simple ante 429/503, para no saturar a BBC ni terminar bloqueado a mitad del scraping.

NO lo corre quien clona el repo: el dataset ya viene poblado vía el dump de la BD.
"""
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
import pandas as pd

from src.core.paths import DATA_DIR

INPUT = DATA_DIR / "bbc_news.csv"
OUTPUT = DATA_DIR / "articulos.csv"
COLUMNAS = ["url", "title", "description", "body", "image_url", "category"]

MAX_WORKERS = 6           # no subir mucho más para no gatillar rate limiting de BBC
SLEEP_POR_REQUEST = 0.15  # pequeño respiro por hilo (no bloquea a los demás workers)
BACKOFF_SEGUNDOS = 5      # espera si BBC responde 429/503


def clave(url) -> str:
    """
    Clave estable para reanudar: el link del RSS trae params de tracking
    (?at_medium=RSS&at_campaign=...) que la url canónica guardada no tiene. Sin
    quitarlos, ningún link del input coincidiría con lo ya scrapeado y se
    re-scrapearía todo desde cero.
    """
    return str(url).split("?")[0]


RUIDO = [
    "This video can not be played",
    "To play this video you need to enable JavaScript",
    "Figure caption,",
    "Share page",
    "Copy link",
    "About sharing",
    ",external",
    "haveyoursay@bbc.co.uk",
    "HaveYourSay@bbc.co.uk",
    "WhatsApp:+44 7756 165803",
    "Tweet:@BBC_HaveYourSay",
    "Please read ourterms & conditionsandprivacy policy",
    "Please include a contact number",
    "Please include your name, age and location",
    "If you are reading this page and can't see the form",
    "visit the mobile version of theBBC website",
    "Are you personally affected by the issues raised in this story",
    "Upload pictures or video",
    "Please include a contact number if you are willing to speak",
]


def limpiar_body(texto):
    for frase in RUIDO:
        texto = texto.replace(frase, "")

    # Limpiar espacios dentro de cada párrafo pero preservar \n\n
    parrafos = texto.split("\n\n")
    parrafos_limpios = [" ".join(p.split()) for p in parrafos]
    parrafos_limpios = [p for p in parrafos_limpios if p.strip()]

    return "\n\n".join(parrafos_limpios)


def _get(url):
    """GET con un reintento simple si BBC responde 429/503 (rate limiting)."""
    r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code in (429, 503):
        time.sleep(BACKOFF_SEGUNDOS)
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    return r


def scrape_articulo(url):
    time.sleep(SLEEP_POR_REQUEST)  # respiro por request, no serializa a los demás workers
    try:
        r = _get(url)
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, "html.parser")

        def get_meta(prop):
            tag = soup.find("meta", property=prop)
            return tag["content"] if tag else None

        title    = get_meta("og:title")
        desc     = get_meta("og:description")
        image    = get_meta("og:image")
        category = get_meta("article:section")
        canon    = get_meta("og:url")

        article = soup.find("article")
        if not article:
            return None

        parrafos = article.find_all("p")
        body = "\n\n".join(p.get_text(strip=True) for p in parrafos)
        body = limpiar_body(body)

        if len(body) < 100:
            return None

        return {
            "url":         canon or url,
            "title":       title,
            "description": desc,
            "body":        body,
            "image_url":   image,
            "category":    category,
        }
    except Exception:
        return None


def run():
    # Cargar URLs ya procesadas para reanudar si se interrumpe
    procesadas = set()
    if OUTPUT.exists():
        df_existente = pd.read_csv(OUTPUT)
        procesadas = {clave(u) for u in df_existente["url"].dropna()}
        print(f"Reanudando — ya procesadas: {len(procesadas)}")

    # Cargar dataset original
    df = pd.read_csv(INPUT)
    urls = df["link"].dropna().unique().tolist()
    pendientes = [u for u in urls if clave(u) not in procesadas]
    print(f"URLs pendientes: {len(pendientes)}")

    if not pendientes:
        print("\n✅ Nada por hacer, ya está todo scrapeado")
        return

    ok = 0
    fail = 0

    # Scraping en paralelo con guardado incremental
    with open(OUTPUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS)

        # escribir header solo si el archivo es nuevo
        if len(procesadas) == 0:
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(scrape_articulo, u): u for u in pendientes}

            for i, fut in enumerate(as_completed(futures), 1):
                url = futures[fut]
                resultado = fut.result()

                if resultado:
                    writer.writerow(resultado)
                    f.flush()  # guardar inmediatamente en disco
                    ok += 1
                    print(f"[{i}/{len(pendientes)}] ✅ {resultado['title'][:60]}")
                else:
                    fail += 1
                    print(f"[{i}/{len(pendientes)}] ❌ {url[:60]}")

    print(f"\n✅ Scraping completado — ok: {ok}, fallidos: {fail}")


if __name__ == "__main__":
    run()