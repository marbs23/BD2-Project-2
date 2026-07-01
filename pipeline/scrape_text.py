"""
Scraper de artículos (maintainer)

Lee el dataset original de Kaggle (data/bbc_news.csv), descarga el HTML de cada
URL y extrae título, descripción, cuerpo, imagen y categoría a data/articulos.csv.
Guarda de forma incremental y reanuda si se interrumpe.

NO lo corre quien clona el repo: el dataset ya viene poblado vía el dump de la BD.
"""
import csv
import time

import requests
from bs4 import BeautifulSoup
import pandas as pd

from src.core.paths import DATA_DIR

INPUT = DATA_DIR / "bbc_news.csv"
OUTPUT = DATA_DIR / "articulos.csv"
COLUMNAS = ["url", "title", "description", "body", "image_url", "category"]


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


def scrape_articulo(url):
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
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

    # Scraping con guardado incremental
    with open(OUTPUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS)

        # escribir header solo si el archivo es nuevo
        if len(procesadas) == 0:
            writer.writeheader()

        for i, url in enumerate(pendientes, 1):
            resultado = scrape_articulo(url)

            if resultado:
                writer.writerow(resultado)
                f.flush()  # guardar inmediatamente en disco
                print(f"[{i}/{len(pendientes)}] ✅ {resultado['title'][:60]}")
            else:
                print(f"[{i}/{len(pendientes)}] ❌ {url[:60]}")

            time.sleep(0.3)  # pausa para no saturar BBC

    print("\n✅ Scraping completado")


if __name__ == "__main__":
    run()
