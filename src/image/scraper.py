import csv
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BBC_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
BBC_CDN = "ichef.bbci.co.uk"


def extract_article_id(url: str) -> str | None:
    m = re.search(r"-(\d{7,})(?:\?|$)", url)
    return m.group(1) if m else None


def extract_category(url: str) -> str:
    clean = url.split("?")[0].replace("https://www.bbc.co.uk/", "")
    parts = clean.split("/")
    if len(parts) < 2:
        return "unknown"
    sub = parts[1]
    m = re.match(r"^([a-z][a-z\-]+?)-\d", sub)
    return m.group(1) if m else parts[0]


def best_src_from_srcset(srcset: str) -> str | None:
    """Picks highest-resolution URL from a srcset attribute."""
    candidates = []
    for part in srcset.split(","):
        part = part.strip()
        tokens = part.split()
        if not tokens:
            continue
        url = tokens[0]
        w = int(tokens[1].replace("w", "")) if len(tokens) > 1 else 0
        candidates.append((w, url))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def scrape_images(article_url: str, out_dir: Path, max_imgs: int = 5) -> list[str]:
    """
    Descarga imágenes del artículo BBC. Devuelve lista de rutas guardadas.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(article_url, headers=BBC_HEADERS, timeout=12)
        if resp.status_code != 200:
            return []
    except requests.RequestException:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    img_urls: list[str] = []

    for img in soup.find_all("img"):
        src = img.get("src", "")
        srcset = img.get("srcset", "")
        if BBC_CDN not in src and BBC_CDN not in srcset:
            continue
        if srcset:
            url = best_src_from_srcset(srcset) or src
        else:
            url = src
        if url and url not in img_urls:
            img_urls.append(url)
        if len(img_urls) >= max_imgs:
            break

    saved: list[str] = []
    for i, img_url in enumerate(img_urls):
        try:
            r = requests.get(img_url, headers=BBC_HEADERS, timeout=10)
            content_type = r.headers.get("Content-Type", "")
            if r.status_code == 200 and "image" in content_type:
                ext = "jpg" if "jpeg" in content_type or "jpg" in content_type else "png"
                path = out_dir / f"img_{i}.{ext}"
                path.write_bytes(r.content)
                saved.append(str(path))
        except requests.RequestException:
            continue

    return saved


def run(
    csv_path: str,
    images_dir: str,
    meta_path: str,
    limit: int | None = None,
    delay: float = 1.0,
    only_news: bool = True,
) -> None:
    """
    Lee bbc_news.csv, scrapea imágenes y guarda metadata.json.

    Args:
        csv_path:   ruta al CSV del dataset
        images_dir: carpeta raíz donde guardar imágenes (data/raw/images/)
        meta_path:  ruta del JSON de salida con metadata
        limit:      máximo de artículos a procesar (None = todos)
        delay:      segundos entre requests (no bajar de 0.5)
        only_news:  si True, salta artículos que no sean /news/
    """
    images_root = Path(images_dir)
    metadata: list[dict] = []
    ok = failed = no_images = skipped = 0

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if limit:
        rows = rows[:limit]

    total = len(rows)
    print(f"Procesando {total} artículos...")

    for i, row in enumerate(rows):
        url = row["link"].split("?")[0]  # quitar params de tracking

        if only_news and "/news/" not in url:
            skipped += 1
            continue

        article_id = extract_article_id(url)
        if not article_id:
            skipped += 1
            continue

        category = extract_category(url)
        out_dir = images_root / article_id

        if i % 100 == 0:
            print(f"  [{i}/{total}] ok={ok} failed={failed} no_img={no_images}")

        saved = scrape_images(url, out_dir)

        record = {
            "article_id": article_id,
            "title": row["title"],
            "description": row["description"],
            "category": category,
            "url": url,
            "pub_date": row["pubDate"],
            "images": saved,
            "status": "ok" if saved else ("failed" if out_dir.exists() else "no_images"),
        }
        metadata.append(record)

        if saved:
            ok += 1
        else:
            no_images += 1

        time.sleep(delay)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nFinalizado: {ok} con imágenes | {no_images} sin imágenes | {failed} fallidos | {skipped} saltados")
    print(f"Metadata guardada en: {meta_path}")
