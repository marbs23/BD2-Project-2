import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv
import os
import time

RUIDO = [
    "This video can not be played",
    "To play this video you need to enable JavaScript",
    "Figure caption,",
    "Share page",
    "Copy link",
    "About sharing",
]

def limpiar_body(texto):
    for frase in RUIDO:
        texto = texto.replace(frase, "")
    return " ".join(texto.split())

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
        body = " ".join(p.get_text(strip=True) for p in parrafos)
        body = limpiar_body(body)

        if len(body) < 100:
            return None

        return {
            "url":         canon or url,
            "title":       title,
            "description": desc,
            "body":        body,
            "image_url":   image,
            "category":    category
        }
    except:
        return None

# ── PIPELINE PRINCIPAL ──────────────────────────────────────

OUTPUT = "articulos.csv"
COLUMNAS = ["url", "title", "description", "body", "image_url", "category"]

# Cargar URLs ya procesadas para reanudar si se interrumpe
procesadas = set()
if os.path.exists(OUTPUT):
    df_existente = pd.read_csv(OUTPUT)
    procesadas = set(df_existente["url"].dropna().tolist())
    print(f"Reanudando — ya procesadas: {len(procesadas)}")

# Cargar dataset original
df = pd.read_csv("bbc_news.csv")
urls = df['link'].dropna().unique().tolist()
pendientes = [u for u in urls if u not in procesadas][:200]
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